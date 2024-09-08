import argparse
import time
from typing import Type, Union, Callable
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from monai.data import MetaTensor
from monai import transforms as MT
import wandb

from engine.utils import (
    AverageMeter, RandAugmentor,
    distributed_all_gather, save_checkpoint,
    select_random_ids, find_executable_batch_size
)
from utils import model_input as ModelInputer
from utils import terminate as Terminate
from utils.data_utils import get_transforms as default_augmentor
from logger import WorstDataRecord


def iter_slice_patch(
        batch_size: int,
        slice_ids: Type[torch.Tensor] | Type[MetaTensor],
        inputs_l: Type[torch.Tensor] | Type[MetaTensor],
        labels_l: Type[torch.Tensor] | Type[MetaTensor],
        model: Type[torch.nn.Module], optimizer: Type[torch.optim.Optimizer],
        scaler: Type[GradScaler] | None, image_only: bool, loss_func: Callable,
        args: Type[argparse.Namespace], **kwargs
):
    """

    :param slice_ids:type np.ndarray: contains which index at slice-axis
    :param inputs_l:type torch.Tensor: the shape is Gs x H x W x z_roi
    :param labels_l:type torch.Tensor: the shape is S x H x W
    :param model:type torch.nn.Module:
    :param optimizer :type torch.optim.Optimizer:
    :param scaler :type torch.optim.GradScaler:
    :param loss_func :type Callable:
    :param args:
    :return:
    """
    slice_iter_loss = torch.as_tensor(.0).cuda(args.rank)

    do_vae = args.vae
    pseudo_bs = batch_size
    seq_slice_ids = slice_ids.split(pseudo_bs)
    step_cnt = kwargs.get('step_cnt', 0)

    for adpt_pseudo_bs, slice_idx in zip(map(len, seq_slice_ids), seq_slice_ids):
        step_cnt += adpt_pseudo_bs
        inputs, labels = inputs_l[slice_idx], labels_l[slice_idx]
        data, target, target_original, skip = ModelInputer.prepare_sam_training_input(
            inputs.cuda(args.rank), labels.cuda(args.rank), args, model
        )
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            outputs = model(data, is_train=True)
        if image_only:  # Only reconstruct loss exist.
            loss = outputs[0]['vae_loss']

            if pseudo_bs > 1:
                loss /= adpt_pseudo_bs
                for pack in outputs[1:]:
                    loss += pack['vae_loss'] / adpt_pseudo_bs

        else:
            if adpt_pseudo_bs > 1:
                pred_mask = torch.cat([_pack['low_res_logits'].permute(1, 0, 2, 3) for _pack in outputs], dim=0)
                loss = loss_func(pred_mask, target)
            else:
                loss = loss_func(outputs[0]['low_res_logits'].permute(1, 0, 2, 3).contiguous(), target)

            if do_vae and adpt_pseudo_bs > 1:
                for pack in outputs:
                    loss += .1 * (pack['vae_loss'] / adpt_pseudo_bs)
            elif do_vae and adpt_pseudo_bs == 1:
                loss += .1 * outputs[0]['vae_loss']
            else:
                pass

        loss *= .0 if skip else 1.
        if args.amp:
            scaler.scale(loss).backward()
            if (limit := args.clip) is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), limit)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if (limit := args.clip) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), limit)
            optimizer.step()
        slice_iter_loss += loss.detach()
    args.quasi_batch_size = pseudo_bs
    return slice_iter_loss / len(slice_ids)


def train_epoch(
        model: Type[torch.nn.Module], loader: Type[torch.utils.data.DataLoader], optimizer: Type[torch.optim.Optimizer],
        scaler: Type[GradScaler], epoch: int, loss_func: Callable, args: Type[argparse.Namespace], **kwargs
):
    print(f'Rank {args.rank} Epoch: {epoch} | Start Initial Training...')
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    bad_record = WorstDataRecord(args, just_name=True)
    permuter: Callable = kwargs.get('permuter')
    augmentor: Callable = kwargs.get('augmenter', kwargs.get('augmentor', RandAugmentor(
        default_augmentor(keys=['image', 'label', 'plaque'], args=args)
    )))
    assert args.roi_z_iter % 2 == 1
    n_slice = args.roi_z_iter
    pd = (n_slice // 2, n_slice // 2)
    step_cnt = 0
    final_epoch = args.iterative_training_warm_up_epoch
    adpt_iter_slice_patch = find_executable_batch_size(iter_slice_patch, args.quasi_batch_size)

    for step, batch_data in enumerate(loader):
        batch_data = augmentor(batch_data)
        batch_data: dict[str, Union[MetaTensor, str, range]]
        # only take 1 batch
        inputs_l = batch_data["image"]
        select_range: range = batch_data.get('range', range(0, inputs_l.shape[-1]))
        only_image = 'label' not in batch_data
        labels_l = batch_data.get("label", torch.zeros_like(inputs_l))
        inputs_l, labels_l = permuter(inputs_l, labels_l)
        # Remove original batch_size and the channel axes. Then swap the slice-axis at first
        labels_l = labels_l.squeeze().permute(2, 0, 1).contiguous()
        # Only image need padding for make sure its shape is same as original shape.
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        inputs_l = inputs_l.squeeze().unfold(-1, n_slice, 1).permute(2, 3, 0, 1).contiguous()
        random_ids: torch.Tensor = select_random_ids(
            select_range, args
        )
        _loss = adpt_iter_slice_patch(
            random_ids, inputs_l, labels_l, model,
            optimizer, scaler, only_image, loss_func, args,
            step=step_cnt
        )
        bad_record.add(_loss, batch_data['image_name'], batch_data['label_name'])
        if args.distributed:
            loss_list = distributed_all_gather(
                [_loss],
                out_numpy=True,
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(_loss.item(), n=len(random_ids))
        Terminate.show_training_info(epoch, step, len(loader), run_loss.avg, start_time, args)
        start_time = time.time()
    # I suggest this function is like optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None
    bad_record.store(epoch)
    return run_loss.avg
