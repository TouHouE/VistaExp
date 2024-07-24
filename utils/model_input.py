from argparse import Namespace
from copy import deepcopy
import random
from typing import Type

import torch
import numpy as np
from monai.data import MetaTensor

from models.vista.modeling import Vista2pt5D


def apply_coords_torch(coords, original_size, sam_image_size) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old = original_size
    new = sam_image_size
    coords = deepcopy(coords).float()
    # Here, we can apply a same scale factor to h and w, because we first pad the input to a square image along the
    # longest side then resize it to sam_image_size. In other words, the scale factor is determined by the longest side.
    coords[..., 0] = coords[..., 0] * (new / old)
    coords[..., 1] = coords[..., 1] * (new / old)
    return coords


def sample_points(labelpoints, n_points):
    idx = torch.randperm(len(labelpoints), dtype=torch.long, device=labelpoints.device)[:n_points]
    return [labelpoints[idx]]


def generate_point_prompt(batch_labels_, args, points_pos=None, points_neg=None, previous_pred=None):
    """

    @param batch_labels_:
    @param args:
    @param points_pos:
    @param points_neg:
    @param previous_pred:
    @return:
    """
    max_point = args.max_points
    if points_pos is not None:
        Np = points_pos
    else:
        gauss = random.gauss(mu=0, sigma=max_point // 2)
        gauss_p = int(np.abs(gauss)) + 1
        Np = min(max_point, gauss_p)

    if points_neg is not None:
        Nn = points_neg
    else:
        gauss = random.gauss(mu=0, sigma=max_point // 2)
        gauss_p = int(np.abs(gauss))
        Nn = min(max_point, gauss_p)

    # To follow original SAM, with equal probability either a foreground point
    # is selected randomly for the target mask
    _point = []
    _point_label = []
    b, h, w = batch_labels_.shape
    device = batch_labels_.device
    for i in range(b):
        plabels = batch_labels_[i, ...]
        nlabels = (plabels == 0.0).float()
        if previous_pred is not None:
            ppred = previous_pred[i, 0, ...]
            npred = (previous_pred[i, 0, ...] == 0.0).float()

            # False positive mask (pixels that are predicted as positive but are actually negative)
            fp_mask = torch.logical_and(nlabels, ppred)
            # False negative mask (pixels that are predicted as negative but are actually positive)
            fn_mask = torch.logical_and(plabels, npred)
            # we sample positive points from false negative pred.
            # we sample negative points from false positive pred.
            plabelpoints = torch.nonzero(fn_mask)
            nlabelpoints = torch.nonzero(fp_mask)

        else:
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
        # 1 indicates a foreground point and 0 indicates a background point.
        # -1 indicates a dummy non-point as the placeholder.
        n_placeholder = Np + Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn)

        # Use torch.randperm to generate indices on a GPU tensor
        _point.append(
            torch.cat(
                sample_points(plabelpoints, min(len(plabelpoints), Np))
                + sample_points(nlabelpoints, min(len(nlabelpoints), Nn))
                + [torch.zeros((1, 2), device=device)] * n_placeholder,
                dim=0,
            )
        )
        _point_label.append(
            torch.tensor([1] * min(len(plabelpoints), Np) + [0] * min(len(nlabelpoints), Nn) + [-1] * n_placeholder).to(
                device
            )
        )

    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    point_coords = apply_coords_torch(point, max(h, w), args.sam_image_size)

    return point_coords, point_label


def prepare_sam_training_input(inputs: torch.Tensor, labels: torch.Tensor, args: Namespace, model: Type[Vista2pt5D]):
    """

    @param inputs: (B) roi_z x H x W
    @param labels: (B) H x W
    @param args:
    @param model:
    @return:
    """
    # Shape with Nc
    unique_labels: torch.Tensor | MetaTensor = torch.unique(labels)
    if hasattr(unique_labels, 'as_tensor'):
        unique_labels: torch.LongTensor = unique_labels.as_tensor().long()
    else:
        unique_labels: torch.LongTensor = unique_labels.long()

    nc_in_mask: int = len(unique_labels)
    if args.skip_bk:
        unique_labels: torch.LongTensor = unique_labels[1:]

    if nc_in_mask == 0:
        prepared_input = list()
        for batch_idx, (_inputs, _labels) in enumerate(zip(inputs, labels)):
            prepared_input.append({
                'image': _inputs,
                'original_size': tuple(_labels.shape)
            })
        # prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
        batch_labels = torch.zeros(batch_idx + 1, 1, args.sam_image_size // 4, args.sam_image_size // 4)
        skip = True
        return prepared_input, batch_labels, None, skip

    # random sample args.num_prompt prompts, this will help to manage the GPU memory upper bound.
    if nc_in_mask > args.num_prompt:
        # random some category in nc_in_mask
        idxs: int = random.sample(range(nc_in_mask), args.num_prompt)
        idxs: torch.Tensor = torch.tensor(idxs)
        unique_labels: torch.LongTensor = unique_labels[idxs]

    if len(unique_labels) < args.num_prompt:
        # Cat unique_labels into unique_labels until the size of nc_in_mask(not unique now) >= num_prompt
        while len(unique_labels) < args.num_prompt:
            unique_labels: torch.LongTensor = torch.cat([unique_labels, unique_labels], 0).long()
        # make sure size of unique_labels == num_prompt
        unique_labels = unique_labels[: args.num_prompt]

    # add 4 background labels to every batch
    # The background labels is meaning
    background_labels = list(set(range(1, args.nc)) - set(unique_labels.cpu().numpy()))
    random.shuffle(background_labels)
    unique_labels: torch.LongTensor = torch.cat([unique_labels, torch.tensor(background_labels[:4]).cuda(args.rank)]).long()

    # preprocess make the size of label same as low_res_logit
    # The shape is (B, Nc, H, W)
    batch_labels_ = torch.cat([labels == unique_labels[i] for i in range(len(unique_labels))], dim=1).float()
    # The shape will become (B, NC, sam_H / 4, sam_W / 4)
    if args.distributed:
        batch_labels = model.module.preprocess(batch_labels_, is_input=False)
    else:
        batch_labels = model.preprocess(batch_labels_, is_input=False)

    # TODO: we currently only use class-label and points prompt.

    prepared_input = list()
    for batch_idx, (_inputs, _labels, _batch_labels_) in enumerate(zip(inputs, labels, batch_labels_)):
        prepared_input.append({
            'image': _inputs,
            'original_size': tuple(_labels.shape)
        })
        if args.label_prompt:
            labels_prompt = unique_labels.unsqueeze(-1)
            prepared_input[batch_idx].update({'labels': labels_prompt})
        if args.point_prompt:
            point_coords, point_labels = generate_point_prompt(_batch_labels_, args)
            prepared_input[batch_idx].update({
                'point_coords': point_coords,
                'point_labels': point_labels
            })
        if args.label_prompt and args.point_prompt:
            if random.uniform(0, 1) < args.drop_label_prob:
                prepared_input[batch_idx].pop('labels')
                continue
            if random.uniform(0, 1) < args.drop_point_prob:
                prepared_input[batch_idx].pop('point_coords')
                prepared_input[batch_idx].pop('point_labels')
    return prepared_input, batch_labels.cuda(args.rank), batch_labels_, False


def prepare_sam_test_input(inputs, labels, args, previous_pred=None):
    unique_labels = torch.tensor([i for i in range(1, args.nc)]).cuda(args.rank)

    # preprocess make the size of lable same as high_res_logit
    batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
    if args.label_prompt:
        labels_prompt = unique_labels.unsqueeze(-1)
        prepared_input[0].update({"labels": labels_prompt})

    if args.point_prompt:
        point_coords, point_labels = generate_point_prompt(
            batch_labels,
            args,
            points_pos=args.points_val_pos,
            points_neg=args.points_val_neg,
            previous_pred=previous_pred,
        )
        prepared_input[0].update({"point_coords": point_coords, "point_labels": point_labels})

    return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), unique_labels


def prepare_sam_val_input_cp_only(inputs, labels, args):
    """

    @param inputs: A 3d tensor with shape roi_z_iter x H x W
    @param labels: A 2d tensor with shape H x W
    @param args:
    @return:
    """
    # Don't exclude background in val but will ignore it in metric calculation
    unique_labels = torch.tensor([i for i in range(1, args.nc)]).cuda(args.rank)

    """
        Some annotation for `batch_labels`
        - preprocess make the size of label same as high_res_logit.
        - As the result, just become the one-hot encoding.
        - The shape is (nc - 1, H, W). nc - 1 is for skip background
    """
    batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]

    labels_prompt = unique_labels.unsqueeze(-1)
    prepared_input[0].update({"labels": labels_prompt})

    return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), unique_labels