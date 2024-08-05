import time

import torch
from torch import autocast
from torch.nn import functional as F
import numpy as np
from monai.data import MetaTensor

from engine.utils import AverageMeter, distributed_all_gather
from utils import model_input as ModelInputer
from utils import terminate as Terminate


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    print(f" Rank: {args.rank} Single-step Training")
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    assert args.roi_z_iter % 2 == 1
    enable_vae = args.vae
    n_slice = args.roi_z_iter
    pd = (n_slice // 2, n_slice // 2)

    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data["image"]
        only_image = 'label' not in batch_data
        labels_l = batch_data.get("label", torch.zeros_like(inputs_l))
        # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        _loss = torch.tensor(0.0).cuda(args.rank)
        _tot_vae_loss = torch.tensor(.0).cuda(args.rank)
        inputs = inputs_l.squeeze().unfold(-1, n_slice, 1).permute(2, 3, 0, 1).contiguous()
        random_ids = torch.from_numpy(np.random.choice(inputs.shape[0], size=args.num_patch, replace=False))
        inputs = inputs[random_ids]
        labels = labels_l.squeeze()[..., random_ids].permute(2, 0, 1).contiguous()
        data, target, target_original, skip = ModelInputer.prepare_sam_training_input(
            inputs.cuda(args.rank), labels.cuda(args.rank), args, model
        )

        with autocast(enabled=args.amp):
            outputs = model(data, is_train=True)
        # not sure this operation is correct or not, i trying to cat at channels axis(maybe)
        pred_mask = torch.cat([_out['low_res_logits'] for _out in outputs], dim=1)
        pred_mask = pred_mask.permute(1, 0, 2, 3)
        if enable_vae:
            for cnt, _out in enumerate(outputs):
                _tot_vae_loss += _out['vae_loss']
            _tot_vae_loss /= cnt + 1

        if only_image:
            loss = _tot_vae_loss
        else:
            loss = loss_func(pred_mask, target)
            loss += .1 * _tot_vae_loss

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
        _loss += loss.detach()
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
    # I suggest this function is like optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None
    return run_loss.avg
