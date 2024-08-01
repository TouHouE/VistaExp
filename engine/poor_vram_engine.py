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
from utils import model_input as ModelInputer
from utils import terminate as Terminate
from utils.decorator import show_exception_file


@show_exception_file
def iter_slice_patch(slice_ids: np.ndarray, inputs_l: torch.Tensor, labels_l: torch.Tensor, model, optimizer, scaler, image_only, loss_func, args, batch_pack):
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
    tot_slice = len(slice_ids)
    pseudo_bs = args.quasi_batch_size

    for start_idx in range(0, tot_slice, pseudo_bs):
        slice_idx = slice_ids[start_idx: start_idx + pseudo_bs]

        inputs, labels = inputs_l[slice_idx].unsqueeze(dim=0), labels_l[slice_idx].unsqueeze(dim=0)
        # ic(inputs.shape)
        # ic(labels.shape)
        data, target, target_original, skip = ModelInputer.prepare_sam_training_input(
            inputs.cuda(args.rank), labels.cuda(args.rank), args, model
        )
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            outputs = model(data, is_train=True)
        if image_only:
            loss = outputs[0]['vae_loss']
        else:
            # ic(outputs[0]['low_res_logits'].shape)
            # ic(target.shape)
            loss = loss_func(outputs[0]['low_res_logits'].permute(1, 0, 2, 3).contiguous(), target)
            if do_vae:
                loss += .1 * outputs[0]['vae_loss']


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


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    assert args.roi_z_iter % 2 == 1
    n_slice = args.roi_z_iter
    pd = (n_slice // 2, n_slice // 2)

    for idx, batch_data in enumerate(loader):
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
            random_ids, inputs_l, labels_l, model, optimizer, scaler, only_image, loss_func, args, batch_data
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
        Terminate.show_training_info(epoch, idx, len(loader), run_loss.avg, start_time, args)
        start_time = time.time()
    # I suggest this function is like optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def train_epoch_iterative(model, loader, optimizer, scaler, epoch, loss_func, run, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    auger = MF.Compose([
        MF.RandRotate90d(
            keys=["image", "label"],
            prob=0.50,
            max_k=3, allow_missing_keys=True
        ),
        MF.RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ])
    # we need to make sure the number of 2.5D input is an odd number.
    assert args.roi_z_iter % 2 == 1
    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        only_image = 'label' not in batch_data
        if args.batch_type == 'aug':
            inputs_l = list()
            labels_l = list()
            for _ in range(args.quasi_batch):
                _data = auger(batch_data["image"])
                _image = _data['image']
                inputs_l.append(_image)
                labels_l.append(data.get('label', torch.zeros_like(_image)))
            inputs_l = torch.cat(inputs_l, dim=0)
            labels_l = torch.cat(labels_l, dim=0)
        else:
            inputs_l = batch_data['image']
            labels_l = batch_data.get('label', torch.zeros_like(inputs_l))

        # TODO: we only support batch_size = 1 for data loader.
        n_z_before_pad = labels_l.shape[-1]

        n_slice = args.roi_z_iter
        # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
        pd = (n_slice // 2, n_slice // 2)
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        labels_l = F.pad(labels_l, pd, "constant", 0)
        _loss = torch.tensor(0.0).cuda(args.rank)

        for _k in range(min(args.num_patch, n_z_before_pad)):
            # Return random integers from `low` (inclusive) to `high` (exclusive).
            start_idx = int(np.random.randint(low=n_slice // 2, high=(n_slice // 2 + n_z_before_pad)))
            left_ptr = start_idx - n_slice // 2
            right_ptr = start_idx + n_slice // 2 + 1
            # B C H W S -> B C S H W -> B S H W
            inputs = inputs_l[..., left_ptr: right_ptr].permute(0, 1, 4, 2, 3)

            # we only need the label for the center slice
            labels = labels_l[..., left_ptr: right_ptr][..., n_slice // 2]

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

            if skip:
                with autocast(enabled=args.amp):
                    if args.distributed:
                        outputs = model.module.get_mask_prediction(data, image_embeddings)
                    else:
                        outputs = model.get_mask_prediction(data, image_embeddings)
                if not only_image:
                    pred_mask = torch.cat([_out['low_res_logits'] for _out in outputs], dim=1)
                    pred_mask = pred_mask.permute(1, 0, 2, 3).contiguous()
                    loss = loss_func(pred_mask, target) * 0.0
            else:
                # iterative training
                loss = 0
                drop_iter = random.randint(0, args.num_iterative_step - 2)
                for i in range(args.num_iterative_step):
                    with autocast(enabled=args.amp):
                        if args.distributed:
                            outputs = model.module.get_mask_prediction(data, image_embeddings)
                        else:
                            outputs = model.get_mask_prediction(data, image_embeddings)
                    pred_mask = torch.cat([_out['low_res_logits'] for _out in outputs], dim=1)
                    pred_mask = pred_mask.permute(1, 0, 2, 3).contiguous()
                    loss += loss_func(pred_mask, target)
                    vae_loss = .0
                    if 'vae_loss' in outputs[0]:
                        for _out in outputs:
                            vae_loss += _out['vae_loss']

                    loss += .1 * vae_loss

                    if i == args.num_iterative_step - 1:
                        # no need to perform the following operations after the last step
                        continue
                    # we also supply the mask prediction from the previous iteration
                    # as an additional prompt to our model (follow original SAM).
                    previous_point_coords = list()
                    previous_point_labels = list()

                    for i in range(len(outputs)):
                        data[i]["mask_inputs"] = outputs[i]["low_res_logits"].detach()
                        previous_point_labels.append(data[i].get('point_labels', None))
                        previous_point_coords.append(data[i].get('point_coords', None))

                    if i == drop_iter:
                        # for drop iter, no additional points are sampled (follow original SAM).
                        continue

                    # previous_point_coords = data[0].get("point_coords", None)
                    # previous_point_labels = data[0].get("point_labels", None)

                    if all(_member is not None for _member in previous_point_coords) and args.no_more_points_for_cp_only:
                        # if no point prompt at the first prompt generation,
                        # we will not add more additional pointa during iterative training.
                        continue
                    previous_pred = torch.cat([F.sigmoid(_out['high_res_logits'].detach()) > .5 for _out in outputs], dim=1).float()

                    # sample one pos and on neg point based on previous prediction
                    # previous_pred = (F.sigmoid(outputs[0]["high_res_logits"].detach()) > 0.5).float()
                    point_coords, point_labels = ModelInputer.generate_point_prompt(
                        target_original, args=args, points_pos=1, points_neg=1, previous_pred=previous_pred
                    )

                    if previous_point_coords is not None:
                        for i in range(len(data)):
                            data[i]['point_coords'] = torch.cat([previous_point_coords[i], point_coords[i]], dim=1)
                            data[i]['point_labels'] = torch.cat([previous_point_labels[i], point_labels[i]], dim=1)
                        # data[0]["point_coords"] = torch.cat([previous_point_coords, point_coords], dim=1)
                        # data[0]["point_labels"] = torch.cat([previous_point_labels, point_labels], dim=1)
                    else:
                        for i in range(len(data)):
                            data[i]['point_coords'] = point_coords[i]
                            data[i]['point_labels'] = point_labels[i]
                        # data[0]["point_coords"] = point_coords
                        # data[0]["point_labels"] = point_labels

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
        _loss /= min(args.num_patch, n_z_before_pad)
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
        if args.rank == 0:
            dur = time.time() - start_time
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(dur),
            )
            if run is not None:
                run.log({
                    'train iter loss': run_loss.avg,
                    'train iter time': dur,
                })

        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, iterative=False, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():

        for idx, batch_data in enumerate(loader):
            # only take 1 batch
            inputs_l = batch_data["image"]
            labels_l = batch_data["label"]
            B = inputs_l.shape[0]
            n_slice = args.roi_z_iter
            # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
            pd = (n_slice // 2, n_slice // 2)

            if B == 1:
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

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx + 1, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg


def run_training(
    model, train_loader, val_loader, optimizer, loss_func, acc_func, args,
    scheduler=None, start_epoch=0, post_label=None, post_pred=None,
):
    writer = None
    run = None
    scaler = None
    # poor_train_epoch = find_executable_batch_size(train_epoch, args.quasi_batch)
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
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = optimizer.param_groups[0]["lr"]
            print("Current lr:", lr)
            if run is not None:
                run.log({
                    'lr': lr,
                    'epoch': epoch
                })
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
            if args.reuse_img_embedding:
                if args.rank == 0:
                    print("Iterative Training: Reuse image embedding!")
                train_loss = train_epoch_iterative(
                    model, train_loader, optimizer,
                    scaler=scaler, epoch=epoch, loss_func=loss_func,
                    run=run, args=args
                )
            else:
                if args.rank == 0:
                    print("Iterative Training: Don't reuse image embedding!")
                raise NotImplementedError
        else:
            print(f" Rank: {args.rank} Single-step Training")
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
            )
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
                iterative=False,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                post_label=post_label, post_pred=post_pred
            )

            val_avg_acc = np.mean(val_avg_acc)
            val_MA = val_avg_acc if val_MA is None else .9 * val_MA + .1 * val_avg_acc

            if args.rank == 0:
                Terminate.show_valided_info(epoch, val_avg_acc, val_MA, best_epoch, val_acc_max, epoch_time, args)
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if run is not None:
                    run.log({
                        'val_acc': val_avg_acc,
                        'epoch': epoch
                    })

                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    best_log[epoch] = float(val_acc_max)
                    best_epoch = epoch
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args,
                            best_acc=val_acc_max,
                            filename="model_best.pt",
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                with open(os.path.join(args.logdir, "train.log"), "w") as f:
                    json.dump(best_log, f)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0 and writer is not None:
        writer.close()

    print("Training Finished !, Best Accuracy: ", val_acc_max, "at epoch", best_epoch)

    return val_acc_max
