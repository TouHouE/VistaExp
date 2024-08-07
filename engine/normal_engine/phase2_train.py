import time
import random

import torch
from torch import autocast
from torch.nn import functional as F
import numpy as np

from engine.utils import AverageMeter, distributed_all_gather
from utils import model_input as ModelInputer
from utils import terminate as Terminate
from utils import assign_device


def iter_adjust(
        image_embeddings, data, target, target_original,
        model, scaler, epoch, loss_func, run, args
):
    loss = 0
    drop_iter = random.randint(0, args.num_iterative_step - 2)
    for i in range(args.num_iterative_step):
        with autocast(enabled=args.amp):
            if args.distributed:
                outputs = model.module.get_mask_prediction(data, image_embeddings)
            else:
                outputs = model.get_mask_prediction(data, image_embeddings)
        pred_mask = torch.cat([_out['low_res_logits'].permute(1, 0, 2, 3) for _out in outputs], dim=0).contiguous()
        loss += loss_func(pred_mask, target)

        if i == args.num_iterative_step - 1:
            # no need to perform the following operations after the last step
            continue
        # we also supply the mask prediction from the previous iteration
        # as an additional prompt to our model (follow original SAM).
        previous_point_coords = list()
        previous_point_labels = list()

        for i, _out in enumerate(outputs):
            data[i]["mask_inputs"] = _out["low_res_logits"].detach()
            previous_point_labels.append(data[i].get('point_labels', None))
            previous_point_coords.append(data[i].get('point_coords', None))

        if i == drop_iter:
            # for drop iter, no additional points are sampled (follow original SAM).
            continue

        # previous_point_coords = data[0].get("point_coords", None)
        # previous_point_labels = data[0].get("point_labels", None)

        if all(_member is not None for _member in
               previous_point_coords) and args.no_more_points_for_cp_only:
            # if no point prompt at the first prompt generation,
            # we will not add more additional pointa during iterative training.
            continue
        _mask_buf = [F.sigmoid(_out['high_res_logits'].detach()) > .5 for _out in outputs]
        previous_pred = torch.cat(_mask_buf, dim=1).float()
        point_coords = list()
        point_labels = list()

        # sample one pos and on neg point based on previous prediction
        # previous_pred = (F.sigmoid(outputs[0]["high_res_logits"].detach()) > 0.5).float()
        for _target_original, _previous_pred in zip(target_original, previous_pred.permute(1, 0, 2, 3).contiguous()):
            _point_coords, _point_labels = ModelInputer.generate_point_prompt(
                _target_original, args=args, points_pos=1, points_neg=1, previous_pred=_previous_pred
            )
            point_coords.append(_point_coords)
            point_labels.append(_point_labels)

        for bidx in range(len(data)):
            if previous_point_coords[bidx] is None:
                data[bidx]['point_coords'] = point_coords[bidx]
                data[bidx]['point_labels'] = point_labels[bidx]
            else:
                data[bidx]['point_coords'] = torch.cat([previous_point_coords[bidx], point_coords[bidx]], dim=1)
                data[bidx]['point_labels'] = torch.cat([previous_point_labels[bidx], point_labels[bidx]], dim=1)
    return loss


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, run, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    # we need to make sure the number of 2.5D input is an odd number.
    assert args.roi_z_iter % 2 == 1
    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        only_image = 'label' not in batch_data
        inputs_l = batch_data['image']
        labels_l = batch_data.get('label', torch.zeros_like(inputs_l))
        n_z_before_pad = labels_l.shape[-1]
        n_slice = args.roi_z_iter
        # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
        pd = (n_slice // 2, n_slice // 2)
        inputs_l = F.pad(inputs_l, pd, "constant", 0).squeeze()
        inputs_l = inputs_l.unfold(-1, n_slice, 1).permute(2, 3, 0, 1).contiguous()
        _loss = assign_device(torch.tensor(0.0), args.rank)
        labels_l = labels_l.squeeze().permute(2, 0, 1).contiguous()
        ids_size = min(args.num_patch, inputs_l.shape[0])
        random_ids = torch.from_numpy(np.random.choice(inputs_l.shape[0], size=ids_size, replace=False))
        inputs = inputs_l[random_ids]
        labels = labels_l[random_ids]

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
        _loss = iter_adjust(
            image_embeddings, data, target, target_original, model, scaler, epoch, loss_func, run, args
        )
        _loss /= ids_size

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
