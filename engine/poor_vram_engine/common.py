import json
import os
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from monai.data import decollate_batch, MetaTensor
from monai.metrics import compute_dice
from monai import transforms as MT
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import wandb

from engine.utils import AverageMeter, distributed_all_gather, save_checkpoint
from engine.poor_vram_engine import phase1_train as PT1
from engine.poor_vram_engine import phase2_train as PT2
from utils import model_input as ModelInputer
from utils import terminate as Terminate


@torch.no_grad()
def val_epoch(model, loader, epoch, acc_func, args, iterative=False, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    n_slice = args.roi_z_iter
    # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
    pd = (n_slice // 2, n_slice // 2)
    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data["image"]
        labels_l = batch_data["label"]
        B = inputs_l.shape[0]
        inputs_l = inputs_l.squeeze()
        labels_l = labels_l.squeeze()

        # padding at last axis (z-axis), the goal in this step like convolution padding
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        labels_l = F.pad(labels_l, pd, "constant", 0)
        n_z_after_pad = labels_l.shape[-1]

        acc_sum_total = 0.0
        not_nans_total = 0.0
        start = n_z_after_pad // 2 - args.num_patch_val // 2
        end = n_z_after_pad // 2 + args.num_patch_val // 2
        # We only loop the center args.num_patch_val slices to save val time
        for start_idx in range(start, end):
            left_ptr = start_idx - n_slice // 2
            right_ptr = start_idx + n_slice // 2 + 1
            if B == 1:
                inputs = inputs_l[..., left_ptr: right_ptr].permute(2, 0, 1)
            else:
                inputs = inputs_l[..., left_ptr: right_ptr].permute(0, 1, 4, 2, 3)

            # we only need the label for the center slice
            labels = labels_l[..., left_ptr: right_ptr][..., n_slice // 2]
            data, target, _ = ModelInputer.prepare_sam_val_input_cp_only(
                inputs.cuda(args.rank), labels.cuda(args.rank), args
            )
            with autocast(enabled=args.amp):
                outputs = model(data)
                logit = torch.cat([_out['high_res_logits'] for _out in outputs], dim=0)

            y_pred = torch.stack(post_pred(decollate_batch(logit)), 0)

            # TODO: we compute metric for each prompt for simplicity in validation.
            acc_batch = compute_dice(y_pred=y_pred, y=target)
            acc_sum, not_nans = (
                torch.nansum(acc_batch).item(),
                args.nc - 1 - torch.sum(torch.isnan(acc_batch).float()).item(),
            )
            acc_sum_total += acc_sum
            not_nans_total += not_nans

        acc, not_nans = acc_sum_total / not_nans_total, not_nans_total
        f_name = batch_data["image"].meta["filename_or_obj"]
        print(f"Rank: {args.rank}, Case: {f_name}, Acc: {acc:.4f}, N_prompts: {int(not_nans)} ")

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
    return run_acc.avg


def run_training(
    model, train_loader, val_loader, optimizer, loss_func, acc_func, args,
    scheduler=None, start_epoch=0, post_label=None, post_pred=None,
):
    writer = None
    run = None
    scaler = None
    train_function: Callable = PT1.train_epoch
    step_cnt = 0

    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
        if args.wandb:
            print(f'Initializing wandb')
            run = wandb.init(project=args.project, name=args.name, config=args)
    if args.amp:
        scaler = GradScaler()

    val_acc_max = 0.0
    best_epoch = -1
    val_MA = None
    best_log = {}
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        # Used to change learning rate
        if args.rank == 0:
            lr = scheduler.get_last_lr() if scheduler is not None else optimizer.param_groups[0]['lr']
            print("Current lr:", lr)
            if run is not None:
                run.log({'lr': lr, 'epoch': epoch})
        # Used to change probability.
        if args.label_prompt and args.point_prompt:
            if epoch < args.label_prompt_warm_up_epoch:
                # during warm up, we drop class label prompt embedding with less prob,
                # since class label prompt embedding layer is trained from scratch.
                args.drop_label_prob = 0.2
                args.drop_point_prob = 0.5
            else:
                # after warmp up, we evenly drop two kinds of prompts
                args.drop_label_prob = 0.5
                args.drop_point_prob = 0.5
            Terminate.show_prob(args)

        # Start Training
        # we don't perform iterative training for the first args.iterative_training_warm_up_epoch epochs
        if epoch > args.iterative_training_warm_up_epoch:
            train_function = PT2.train_epoch
        #     if args.reuse_img_embedding:
        #         if args.rank == 0:
        #             print("Iterative Training: Reuse image embedding!")
        #         train_loss = train_epoch_iterative(
        #             model, train_loader, optimizer,
        #             scaler=scaler, epoch=epoch, loss_func=loss_func,
        #             run=run, args=args
        #         )
        #     else:
        #         if args.rank == 0:
        #             print("Iterative Training: Don't reuse image embedding!")
        #         raise NotImplementedError
        # else:
        #     pass
        train_loss = train_function(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, run=run
        )
            # print(f" Rank: {args.rank} Single-step Training")
            # train_loss = train_epoch(
            #     model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, run=run
            # )
        # Training Done.

        if args.rank == 0:
            Terminate.show_trained_info(epoch, train_loss, epoch_time, args)

            if writer is not None:
                writer.add_scalar("train_loss", train_loss, epoch)
            if run is not None:
                run.log({
                    'train_loss': train_loss,
                    'epoch': epoch
                })
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
                args=args, post_label=post_label, post_pred=post_pred
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
                with open(os.path.join(args.logdir, "train.log"), "w") as f:
                    json.dump(best_log, f)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint and not args.test_mode:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0 and writer is not None:
        writer.close()

    print("Training Finished !, Best Accuracy: ", val_acc_max, "at epoch", best_epoch)

    return val_acc_max
