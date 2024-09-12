import json
import os
import random
import time
from typing import Callable
import gc

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from monai.data import decollate_batch, MetaTensor
from monai.metrics import compute_dice
from monai import losses as ML
from monai import transforms as MT
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import wandb
from icecream import ic

from engine import utils as EU
from engine.utils import (
    AverageMeter, TrainingAlgoManager, RandAugmentor, RandomPermute,
    distributed_all_gather, save_checkpoint
)
from engine.poor_vram_engine import phase1_train as PT1
from engine.poor_vram_engine import phase2_train as PT2
from engine.poor_vram_engine import phase3_train as PT3
from utils import model_input as ModelInputer
from utils import terminate as Terminate
from utils.losses import GapLoss

@torch.no_grad()
def val_epoch(model, loader, epoch, acc_func, args, iterative=False, post_label=None, post_pred=None, **kwargs):
    model.eval()
    run_acc = AverageMeter(args=args)
    start_time = time.time()
    n_slice = args.roi_z_iter
    hf_slice = n_slice // 2
    # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
    pd = (hf_slice, hf_slice)
    num_half_patch: int = args.num_patch_val // 2

    for idx, batch_data in enumerate(loader):
        buf_image, buf_pred, buf_label, buf_slice_loc = list(), list(), list(), list()
        # only take 1 batch
        inputs_l = batch_data["image"].squeeze()
        labels_l = batch_data["label"].squeeze()
        B = inputs_l.shape[0]
        # padding at last axis (z-axis), the goal in this step like convolution padding
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        inputs_l = inputs_l.unfold(-1, n_slice, 1).permute(2, 3, 0, 1).contiguous()

        labels_l = labels_l.permute(2, 0, 1).contiguous()
        n_group_patch = inputs_l.shape[0]
        if (n_group_patch := inputs_l.shape[0]) > (num_val_patch := args.num_patch_val):
            middle_group_index = n_group_patch // 2
            val_patch_ids = torch.arange(middle_group_index - num_half_patch, middle_group_index + num_half_patch + 1)
        else:
            val_patch_ids = torch.arange(n_group_patch)
        
        acc_sum_total = 0.0
        not_nans_total = 0.0
        # We only loop the center args.num_patch_val slices to save val time
        for patch_idx in val_patch_ids:
            inputs = inputs_l[patch_idx].unsqueeze(0)
            # we only need the label for the center slice
            labels = labels_l[patch_idx].unsqueeze(0)
            # Collect wandb.log element
            ic(inputs.shape)
            buf_image.append(inputs[0, hf_slice].cpu().numpy())
            buf_label.append(labels[0].cpu().numpy())

            data, target, _ = ModelInputer.prepare_sam_val_input_cp_only(
                ModelInputer.assign_device(inputs, args.rank), ModelInputer.assign_device(labels, args.rank), args
            )
            ic(len(data))
            
            with autocast(enabled=args.amp):
                outputs = model(data)
                ic(outputs[0]['high_res_logits'].shape)
                logit = torch.cat([_out['high_res_logits'] for _out in outputs], dim=0)
            ic(data[0]['image'].shape)
            ic(logit.shape)
            y_pred = torch.stack(post_pred(decollate_batch(logit)), 1)
            # Collect wandb.log element
            buf_pred.append(y_pred.cpu().numpy())
            # All wandb.log element collection are done
            ic(y_pred.shape)
            ic(target.shape)
            acc_batch = compute_dice(y_pred=y_pred, y=target)
            acc_sum, not_nans = (
                torch.nansum(acc_batch).item(),
                args.nc - 1 - torch.sum(torch.isnan(acc_batch).float()).item(),
            )
            acc_sum_total += acc_sum
            not_nans_total += not_nans
            del inputs, labels, y_pred, target
            gc.collect()
            torch.cuda.empty_cache()


        acc, not_nans = acc_sum_total / not_nans_total, not_nans_total
        f_name = batch_data["image"].meta["filename_or_obj"]
        print(f"Rank: {args.rank}, Case: {f_name}, Acc: {acc:.4f}, N_prompts: {int(not_nans)} ")
        # prepare some element for update, I take those element as "Bad" quality data.
        run_acc.add(
            acc, sample_name=[batch_data['image_name'], batch_data['label_name']],
            buf_image=buf_image, buf_label=buf_label, buf_pred=buf_pred
        )
        acc = torch.tensor(acc).cuda(args.rank)
        not_nans = torch.tensor(not_nans).cuda(args.rank)

        if args.distributed:
            acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True)
            for al, nl in zip(acc_list, not_nans_list):
                run_acc.update(al, n=nl)

        else:
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
        Terminate.show_validing_info(epoch, run_acc.avg, idx, len(loader), start_time, args)
        start_time = time.time()
    run_acc.log_worst(kwargs.get('run'), epoch)
    return run_acc.avg


def run_training(
        model, train_loader, val_loader, optimizer, loss_func, acc_func, args,
        scheduler=None, start_epoch=0, post_label=None, post_pred=None,
):
    writer, run, scaler, stage, train_function = None, None, None, 'init', PT1.train_epoch
    axesPermuter = RandomPermute(args)
    algoManager = TrainingAlgoManager(algo_map={
        1: PT1.train_epoch,
        2: PT2.train_epoch,
        3: PT3.train_epoch
    }, patience=5, mode='max', enable_eta=1e-4, cooldown=15, init_stage=1)
    step_cnt = 0

    if args.logdir is not None and args.rank == 0 and not args.test_mode:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
        if args.wandb:
            print(f'Initializing wandb')
            entity = getattr(args, 'id')
            run = wandb.init(
                project=args.project, name=args.name, id=entity, config=args, dir=args.logdir,
                resume='allow'
            )
    if args.amp:
        scaler = GradScaler()

    val_acc_max = 0.0
    val_avg_acc = .0
    best_epoch = -1
    val_MA = None
    best_log = {}
    train_function: Callable = algoManager.current_algo()

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        Terminate.hint_lr(args, epoch, optimizer, scheduler, run)

        # Used to change probability.
        if EU.change_drop_prob(args, epoch):
            Terminate.show_prob(args)
        if algoManager.current_stage == 3:
            loss_func = GapLoss(loss_map_func=ML.DiceFocalLoss(softmax=True, reduction='none', batch=True))
        # Start Training
        # we don't perform iterative training for the first args.iterative_training_warm_up_epoch epochs
        # if epoch > args.iterative_training_warm_up_epoch and stage == 'init':
        #     train_function = PT2.train_epoch
        #     stage = 'adjust'

        train_loss = train_function(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args,
            run=run, permuter=axesPermuter
        )
        # Training Done.

        Terminate.show_trained_info(args, epoch, train_loss, epoch_time, writer, run)
        # Show Training information done.

        # Check running validation or not
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            Terminate.show_before_valid_info(args)
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model, val_loader,
                iterative=False, epoch=epoch, acc_func=acc_func,
                args=args, post_label=post_label, post_pred=post_pred, run=run
            )

            val_avg_acc = np.mean(val_avg_acc)
            val_MA = val_avg_acc if val_MA is None else .9 * val_MA + .1 * val_avg_acc

            if args.rank == 0:
                Terminate.show_valided_info(epoch, val_avg_acc, val_MA, best_epoch, val_acc_max, epoch_time, args)
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if run is not None:
                    run.log({'val_acc': val_avg_acc, 'epoch': epoch})

                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    best_log[epoch] = float(val_acc_max)
                    best_epoch = epoch
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint and not args.test_mode:
                        save_checkpoint(
                            model, epoch, args,
                            best_acc=val_acc_max,
                            filename="model_best.pt",
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                if not args.test_mode:
                    with open(os.path.join(args.logdir, "train.log"), "w") as f:
                        json.dump(best_log, f)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint and not args.test_mode:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if scheduler is not None:
            scheduler.step()
        train_function = algoManager.next_algo(val_avg_acc)
    if args.rank == 0 and writer is not None:
        writer.close()

    print("Training Finished !, Best Accuracy: ", val_acc_max, "at epoch", best_epoch)

    return val_acc_max
