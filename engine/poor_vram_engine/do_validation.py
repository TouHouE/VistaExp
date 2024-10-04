import argparse
import gc
from typing import Type, Callable, Optional, Any, Iterable
import time

from monai.metrics import compute_dice
from monai.data import MetaTensor, decollate_batch
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from icecream import ic

from engine.utils import distributed_all_gather, AverageMeter, find_executable_batch_size
from utils import model_input as ModelInputer
from utils import terminate as Terminate

def iter_slice_patch(
        batch_size: int,
        slice_ids: Type[torch.Tensor] | Type[MetaTensor],
        inputs_l: Type[torch.Tensor] | Type[MetaTensor],
        labels_l: Type[torch.Tensor] | Type[MetaTensor],
        model: Type[torch.nn.Module], post_pred: Callable,
        args: Type[argparse.Namespace], **kwargs
):
    """
    :param batch_size: this argument is for huggingface's accelerate package
    :param slice_ids:type np.ndarray: contains which index at slice-axis
    :param inputs_l:type torch.Tensor: the shape is Gs x H x W x z_roi
    :param labels_l:type torch.Tensor: the shape is S x H x W
    :param model:type torch.nn.Module:
    :param scaler :type torch.optim.GradScaler:
    :param loss_func :type Callable:
    :param args:
    :return: all class dice score and val_losses
    """
    pseudo_bs = batch_size
    seq_slice_ids = slice_ids.split(pseudo_bs)
    step_cnt = kwargs.get('step_cnt', 0)
    pred_record = list()
    target_record = list()

    for adpt_pseudo_bs, slice_idx in zip(map(len, seq_slice_ids), seq_slice_ids):
        step_cnt += adpt_pseudo_bs
        inputs, labels = inputs_l[slice_idx], labels_l[slice_idx]
        data, target, _ = ModelInputer.prepare_sam_val_input_cp_only(
            inputs.cuda(args.rank), labels.cuda(args.rank), args
        )
        with autocast(enabled=args.amp):
            outputs = model(data, is_train=False)

            # Let shape from N_class_propmt x B x H x W -> B x N_class_prompt x H x W

        if adpt_pseudo_bs > 1:
            pred_mask = torch.cat([_pack['high_res_logits'].permute(1, 0, 2, 3) for _pack in outputs], dim=0)
        else:
            pred_mask = outputs[0]['high_res_logits'].permute(1, 0, 2, 3).contiguous()

        pred_mask = torch.stack(post_pred(decollate_batch(pred_mask)), dim=1)
        pred_record.append(pred_mask.cpu())
        target_record.append(target.cpu())

        del inputs, labels, pred_mask, target
        gc.collect()
        torch.cuda.empty_cache()

    # pred shape is: N_class x N_patch x H x W
    quasi3d_pred = torch.cat(pred_record, dim=1)[None]  # (B=1) x N_class x N_patch x H x W
    # target shape is: N_patch x N_class x H x W -> (B=1) x N_class x N_patch x H x W
    quasi3d_label = torch.cat(target_record, dim=0)[None].permute(0, 2, 1, 3, 4)  # the shape like quasi3d_pred
    breakpoint()
    sampling_acc = compute_dice(quasi3d_pred, quasi3d_label)
    args.quasi_batch_size = pseudo_bs
    return sampling_acc


@torch.no_grad()
def val_epoch(
        model: torch.nn.Module, loader: torch.utils.data.DataLoader, epoch: int,
        acc_func: Optional[Callable], args: argparse.Namespace, iterative: Optional[bool] = False,
        post_label: Optional[Any] = None, post_pred=None, **kwargs
) -> AverageMeter:
    model.eval()
    run_acc = AverageMeter(args=args)
    start_time = time.time()
    n_slice = args.roi_z_iter
    hf_slice = n_slice // 2
    # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
    pd = (hf_slice, hf_slice)
    num_half_patch: int = args.num_patch_val // 2
    adpt_all_patch = find_executable_batch_size(iter_slice_patch, starting_batch_size=args.quasi_batch_size)

    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        f_name = batch_data["image"].meta["filename_or_obj"]
        inputs_l = batch_data["image"].squeeze()
        labels_l = batch_data["label"].squeeze()
        # padding at last axis (z-axis), the goal in this step like convolution padding
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        inputs_l = inputs_l.unfold(-1, n_slice, 1).permute(2, 3, 0, 1).contiguous()
        labels_l = labels_l.permute(2, 0, 1).contiguous()

        if (n_group_patch := inputs_l.shape[0]) > (num_val_patch := args.num_patch_val):
            middle_group_index = n_group_patch // 2
            val_patch_ids = torch.arange(middle_group_index - num_half_patch, middle_group_index + num_half_patch + 1)
        else:
            val_patch_ids = torch.arange(n_group_patch)
        inputs_l, labels_l = inputs_l.as_tensor(), labels_l.as_tensor()
        class_dice = adpt_all_patch(
            val_patch_ids, inputs_l, labels_l, model, args=args, post_pred=post_pred
        )
        not_nans = args.nc - 1 - torch.isnan(class_dice).float().sum()
        acc = torch.nansum(class_dice) / not_nans
        print(f"Rank: {args.rank}, Case: {f_name}, Acc: {acc:.4f}, N_prompts: {int(not_nans)} ")
        acc = torch.tensor(acc).cuda(args.rank)
        not_nans = torch.tensor(not_nans).cuda(args.rank)

        if args.distributed:
            b_class_dice = distributed_all_gather([class_dice], out_numpy=True)
            for class_dice in b_class_dice:
                tot_dice = torch.nansum(class_dice)
                not_nan_class = args.nc - 1 - (torch.isnan(class_dice).float().item())
                run_acc.update(tot_dice, n=not_nan_class)
                run_acc.update_dice(class_dice)
        else:
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            run_acc.update_dice(class_dice)
        Terminate.show_validing_info(epoch, run_acc.avg, idx, len(loader), start_time, args)
        start_time = time.time()
    run_acc.log_worst(kwargs.get('run'), epoch)
    return run_acc