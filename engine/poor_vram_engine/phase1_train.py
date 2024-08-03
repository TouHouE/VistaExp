import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
import wandb

from engine.utils import AverageMeter, distributed_all_gather, save_checkpoint
from utils import model_input as ModelInputer
from utils import terminate as Terminate


def iter_slice_patch(
        slice_ids: np.ndarray, inputs_l: torch.Tensor, labels_l: torch.Tensor,
        model, optimizer, scaler, image_only, loss_func,
        args, batch_pack, **kwargs
):
    """

    :param slice_ids:type np.ndarray: contains which index at slice-axis
    :param inputs_l:type torch.Tensor: the shape is Gs x H x W x z_roi
    :param labels_l:type torch.Tensor: the shape is S x H x W
    :param model:
    :param optimizer:
    :param scaler:
    :param epoch:
    :param loss_func:
    :param args:
    :return:
    """
    slice_iter_loss = torch.as_tensor(.0).cuda(args.rank)

    do_vae = args.vae
    pseudo_bs = args.quasi_batch_size
    seq_slice_ids = slice_ids.split(pseudo_bs)
    step_cnt = kwargs.get('step_cnt', 0)

    for adpt_pseudo_bs, slice_idx in zip(map(len, seq_slice_ids), seq_slice_ids):
        # slice_idx = slice_ids[start_idx: start_idx + pseudo_bs]
        step_cnt += adpt_pseudo_bs
        inputs, labels = inputs_l[slice_idx], labels_l[slice_idx]
        # ic(inputs.shape)
        # ic(labels.shape)
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
            # ic(outputs[0]['low_res_logits'].shape)
            # ic(target.shape)
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
    return slice_iter_loss / len(slice_ids)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, **kwargs):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    assert args.roi_z_iter % 2 == 1
    n_slice = args.roi_z_iter
    pd = (n_slice // 2, n_slice // 2)
    step_cnt = 0

    for step, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data["image"]
        only_image = 'label' not in batch_data
        labels_l = batch_data.get("label", torch.zeros_like(inputs_l))
        # Remove original batch_size and the channel axes. Then swap the slice-axis at first
        labels_l = labels_l.squeeze().permute(2, 0, 1).contiguous()
        # Only image need padding for make sure its shape is same as original shape.
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        inputs_l = inputs_l.squeeze().unfold(-1, n_slice, 1).permute(2, 3, 0, 1).contiguous()
        # ic(inputs_l.shape)

        if (bs_size := args.num_patch) >= (num_group := inputs_l.shape[0]):
            random_ids = torch.arange(num_group)
        else:
            random_ids = torch.from_numpy(np.random.choice(num_group, size=bs_size, replace=False))

        _loss = iter_slice_patch(
            random_ids, inputs_l, labels_l, model, optimizer, scaler, only_image, loss_func, args, batch_data,
            step=step_cnt
        )

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
    return run_loss.avg
