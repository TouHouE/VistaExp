import json
import os
import random
import time
from copy import deepcopy
from argparse import Namespace


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from monai.data import decollate_batch, MetaTensor
from monai.metrics import compute_dice
from monai import transforms as MF
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from accelerate.utils import find_executable_batch_size
from icecream import ic
import wandb

from engine.utils import AverageMeter, distributed_all_gather, save_checkpoint
from utils import model_input as ModelInputer, assign_device
from utils import terminate as Terminate
from utils.decorator import show_exception_file


def prompt_adjust_mask(image_embedding, data, target, target_original: torch.Tensor, model, loss_func, args):
    loss = .0
    for iter_step_cnt in range(args.num_iterative_step):
        with autocast(enabled=args.amp):
            if args.distributed:
                outputs = model.module.get_mask_prediction(data, image_embedding)
            else:
                outputs = model.get_mask_prediction(data, image_embedding)
        pred_mask = torch.cat([_out['low_res_logits'].permute(1, 0, 2, 3) for _out in outputs], dim=0).contiguous()
        loss += loss_func(pred_mask, target)

        previous_point_coords = list()
        previous_point_labels = list()

        for i, _out in enumerate(outputs):            
            data[i]["mask_inputs"] = _out["low_res_logits"].detach()
            previous_point_labels.append(data[i].get('point_labels', None))
            previous_point_coords.append(data[i].get('point_coords', None))
        previous_pred = torch.cat([F.sigmoid(_out['high_res_logits'].detach()) > .5 for _out in outputs], dim=1).float()
        ic(previous_pred.shape)
        ic(target_original.shape)
        point_coords = list()
        point_labels = list()

        for _target_original, _previous_pred in zip(target_original, previous_pred.permute(1, 0, 2, 3).contiguous()):
            _point_coords, _point_labels = ModelInputer.generate_point_prompt(
                _target_original, args=args, points_pos=1, points_neg=1, previous_pred=_previous_pred
            )
            point_labels.append(_point_labels)
            point_coords.append(_point_coords)
        ic(list(any(torch.isnan(_x.view(-1))) for _x in point_labels))
        ic(list(any(torch.isnan(_x.view(-1))) for _x in point_coords))

        for bidx in range(len(data)):
            if previous_point_coords[bidx] is None:
                data[bidx]['point_coords'] = point_coords[bidx]
                data[bidx]['point_labels'] = point_labels[bidx]
            else:
                data[bidx]['point_coords'] = torch.cat([previous_point_coords[bidx], point_coords[bidx]], dim=1)
                data[bidx]['point_labels'] = torch.cat([previous_point_labels[bidx], point_labels[bidx]], dim=1)
    
    return loss


def iter_slice_patch(
        slice_ids: torch.Tensor, inputs_l: torch.Tensor, labels_l: torch.Tensor,
        model, optimizer, scaler, image_only, loss_func,
        args, **kwargs
):
    _loss = assign_device(torch.tensor(0.0), args.rank)
    do_vae = args.vae
    pseudo_bs = args.quasi_batch_size
    seq_slice_ids = slice_ids.split(pseudo_bs)

    for adpt_pseudo_bs, slice_idx in zip(map(len, seq_slice_ids), seq_slice_ids):
        inputs, labels = inputs_l[slice_idx], labels_l[slice_idx]
        ic(inputs.shape)
        ic(labels.shape)
        data, target, target_original, skip = ModelInputer.prepare_sam_training_input(
            inputs.cuda(args.rank), labels.cuda(args.rank), args, model
        )
        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            if args.distributed:
                image_embeddings = model.module.get_image_embeddings(data)
            else:
                image_embeddings = model.get_image_embeddings(data)

        loss = prompt_adjust_mask(
            image_embeddings, data, target, target_original, model, loss_func, args
        )

        if args.amp:
            scaler.scale(loss).backward()
            if args.clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        _loss += loss.detach() / args.num_iterative_step
    return _loss


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, run, args):
    print(f'Prompt Adjust training...')
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # we need to make sure the number of 2.5D input is an odd number.
    assert args.roi_z_iter % 2 == 1
    n_slice = args.roi_z_iter
    pd = (n_slice // 2, n_slice // 2)

    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data['image']
        image_only = 'label' not in batch_data
        labels_l: torch.Tensor = batch_data.get('label', torch.zeros_like(inputs_l))
        inputs_l = F.pad(inputs_l, pd, "constant", 0).squeeze()
        labels_l = labels_l.squeeze().permute(2, 0, 1).contiguous()
        inputs_patch = inputs_l.unfold(-1, n_slice, 1)
        ic(inputs_patch.shape)
        inputs_patch = inputs_patch.permute(2, 3, 0, 1).contiguous()
        n_inputs_patch = inputs_patch.shape[0]
        ids_size = min(args.num_patch, n_inputs_patch)
        random_ids = torch.from_numpy(np.random.choice(n_inputs_patch, size=ids_size, replace=False))

        _loss = iter_slice_patch(
            random_ids, inputs_patch, labels_l, model, optimizer, scaler, image_only, loss_func, args
        )

        _loss /= min(args.num_patch, n_inputs_patch)
        if args.distributed:
            loss_list = distributed_all_gather(
                [_loss],
                out_numpy=True,
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(_loss.item(), n=args.num_patch)
        Terminate.show_training_info(epoch, idx, len(loader), run_loss.avg, start_time, args)
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg
