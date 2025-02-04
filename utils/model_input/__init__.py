from utils.model_input._model_input import *
# from argparse import Namespace
# from copy import deepcopy
# import random
# from typing import Type, Optional, Tuple, List, Dict, Any
#
# import torch
# import numpy as np
# from monai.data import MetaTensor
# from icecream import ic
# from numpy import ndarray
# from torch import Tensor
#
# from models.vista.modeling import Vista2pt5D
# from utils import assign_device
# from utils.model_input import model_inputs_utils as InputUtils
#
#
# def find_possible(args, get_down=False):
#     n_patch = args.num_patch
#     qbs = args.quasi_batch_size
#     dif = n_patch % qbs
#     if dif == 0:
#         return qbs
#     if get_down:
#         return n_patch // (n_patch // qbs)
#
#     return qbs + dif
#
#
# def apply_coords_torch(coords, original_size, sam_image_size) -> np.ndarray:
#     """
#     Expects a numpy array of length 2 in the final dimension. Requires the
#     original image size in (H, W) format.
#     """
#     old = original_size
#     new = sam_image_size
#     coords = deepcopy(coords).float()
#     # Here, we can apply a same scale factor to h and w, because we first pad the input to a square image along the
#     # longest side then resize it to sam_image_size. In other words, the scale factor is determined by the longest side.
#     coords[..., 0] = coords[..., 0] * (new / old)
#     coords[..., 1] = coords[..., 1] * (new / old)
#     return coords
#
#
# def sample_points(labelpoints, n_points):
#     idx = torch.randperm(len(labelpoints), dtype=torch.long, device=labelpoints.device)[:n_points]
#     return [labelpoints[idx]]
#
#
# def generate_point_prompt(
#         batch_labels_, args,
#         points_pos: Optional[int] = None, points_neg: Optional[int] = None,
#         previous_pred: Optional[Type[torch.Tensor] | Type[MetaTensor]] = None
# ):
#     """
#     This method can only process one sample each time.
#     @param batch_labels_:
#     @param args:
#     @param points_pos:
#     @param points_neg:
#     @param previous_pred:
#     @return:
#     """
#     max_point = args.max_points
#     if points_pos is not None:
#         Np = points_pos
#     else:
#         gauss = random.gauss(mu=0, sigma=max_point // 2)
#         gauss_p = int(np.abs(gauss)) + 1
#         Np = min(max_point, gauss_p)
#
#     if points_neg is not None:
#         Nn = points_neg
#     else:
#         gauss = random.gauss(mu=0, sigma=max_point // 2)
#         gauss_p = int(np.abs(gauss))
#         Nn = min(max_point, gauss_p)
#
#     # To follow original SAM, with equal probability either a foreground point
#     # is selected randomly for the target mask
#     _point = []
#     _point_label = []
#     ic(batch_labels_.shape)
#     b, h, w = batch_labels_.shape
#     device = batch_labels_.device
#     for prompt_idx in range(b):
#         plabels = batch_labels_[prompt_idx, ...]
#         nlabels = (plabels == 0.0).float()
#         if previous_pred is not None:
#             ppred = previous_pred[prompt_idx, 0, ...]
#             npred = (previous_pred[prompt_idx, 0, ...] == 0.0).float()
#
#             # False positive mask (pixels that are predicted as positive but are actually negative)
#             fp_mask = torch.logical_and(nlabels, ppred)
#             # False negative mask (pixels that are predicted as negative but are actually positive)
#             fn_mask = torch.logical_and(plabels, npred)
#             # we sample positive points from false negative pred.
#             # we sample negative points from false positive pred.
#             plabelpoints = torch.nonzero(fn_mask)
#             nlabelpoints = torch.nonzero(fp_mask)
#
#         else:
#             plabelpoints = torch.nonzero(plabels)
#             nlabelpoints = torch.nonzero(nlabels)
#         # 1 indicates a foreground point and 0 indicates a background point.
#         # -1 indicates a dummy non-point as the placeholder.
#         n_placeholder = Np + Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn)
#
#         # Use torch.randperm to generate indices on a GPU tensor
#         _point.append(
#             torch.cat(
#                 sample_points(plabelpoints, min(len(plabelpoints), Np))
#                 + sample_points(nlabelpoints, min(len(nlabelpoints), Nn))
#                 + [torch.zeros((1, 2), device=device)] * n_placeholder,
#                 dim=0,
#             )
#         )
#         _point_label.append(
#             torch.tensor([1] * min(len(plabelpoints), Np) + [0] * min(len(nlabelpoints), Nn) + [-1] * n_placeholder).to(
#                 device
#             )
#         )
#     ic(_point[0].shape)
#     point = torch.stack(_point)
#     ic(point.shape)
#     point_label = torch.stack(_point_label)
#     ic(point_label.shape)
#     point_coords = apply_coords_torch(point, max(h, w), args.sam_image_size)
#
#     return point_coords, point_label
#
#
# def prepare_sam_training_input(
#         inputs: torch.Tensor | MetaTensor, labels: torch.Tensor | MetaTensor, args: Namespace,
#         model: Type[Vista2pt5D], only_challenge_categories: bool = False):
#     """
#
#     :param inputs: B x roi_z x H x W
#     :param labels: B x H x W
#     :param args:
#     :param model:
#     :param only_challenge_categories:
#     :return:
#     """
#     ic(inputs.shape)
#     ic(labels.shape)
#     # breakpoint()
#     # Shape with Nc
#     unique_labels: list[torch.Tensor | MetaTensor] = [torch.unique(_l) for _l in labels]
#     unique_labels: list[torch.LongTensor] = InputUtils.get_unique_labels(
#         unique_labels, getattr(args, 'poor_categories') if only_challenge_categories else None
#     )
#
#     nc_in_mask: int = len(unique_labels)
#     if args.skip_bk:
#         unique_labels: list[torch.Tensor] = [ul[ul != 0].long() for ul in unique_labels]
#     # Only possible when background was skipped.
#     if nc_in_mask == 0:
#         prepared_input = list()
#         for batch_idx, (_inputs, _labels) in enumerate(zip(inputs, labels)):
#             prepared_input.append({
#                 'image': _inputs,
#                 'original_size': tuple(_labels.shape)
#             })
#         # prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
#         batch_labels = torch.zeros(batch_idx + 1, 1, args.sam_image_size // 4, args.sam_image_size // 4)
#         skip = True
#         return prepared_input, batch_labels, None, skip
#
#     # random sample args.num_prompt prompts, this will help to manage the GPU memory upper bound.
#     # if nc_in_mask > args.num_prompt:
#     #     # random some category in nc_in_mask
#     #     idxs = torch.from_numpy(np.random.choice(nc_in_mask, args.num_prompt).astype(np.int32))
#     #     unique_labels: torch.LongTensor = unique_labels[idxs]
#     #
#     # ic(unique_labels.shape)
#     # if len(unique_labels) < args.num_prompt:
#     #
#     #     # Cat unique_labels into unique_labels until the size of nc_in_mask(not unique now) >= num_prompt
#     #     while len(unique_labels) < args.num_prompt:
#     #         double_labels = [unique_labels] * 2
#     #         # Let the shape at least greater than num_prompt
#     #         unique_labels = torch.cat(double_labels, 0).long()
#     #     # make sure size of unique_labels == num_prompt
#     #     # The shape become [num_prompt]
#     #     unique_labels = unique_labels[: args.num_prompt]
#     #
#     # # add 4 background labels to every batch
#     # # The background labels is meaning: missing label in current labels.
#     # background_labels = list(set(range(1, args.nc)) - set(unique_labels.cpu().numpy()))
#     # random.shuffle(background_labels)
#     # candidate_tensor = [unique_labels, assign_device(torch.tensor(background_labels[:4]), args.rank)]
#     # unique_labels: torch.LongTensor = torch.cat(candidate_tensor).long()
#     # unique_labels = keep_unique
#
#     # preprocess make the size of label same as low_res_logit
#     # The shape is (B, num_prompt, H, W)
#     unique_labels = InputUtils.keep_label_prompt_size(unique_labels, args.num_prompt)
#     unique_labels = InputUtils.append_not_exist_label_prompt(unique_labels, args)
#     print(*[ul.shape for ul in unique_labels], sep='\t')
#     batch_labels = InputUtils.generate_prompt_labels(unique_labels, labels)
#     # buf_labels = [labels == unique_labels[i] for i in range(len(unique_labels))]  # BoolTensor
#     # batch_labels = torch.stack(buf_labels, dim=1).float()
#
#     # The shape will become (B, num_prompt, sam_H / 4, sam_W / 4)
#     if args.distributed:
#         batch_labels = model.module.preprocess(batch_labels, is_input=False)
#     else:
#         batch_labels = model.preprocess(batch_labels, is_input=False)
#
#     prepared_input = list()
#     for batch_idx, (_inputs, _labels, _batch_labels, _labels_prompt) in enumerate(zip(inputs, labels, batch_labels, unique_labels)):
#         prepared_input.append({
#             'image': _inputs,
#             'original_size': tuple(_labels.shape)
#         })
#         if args.label_prompt:
#             # labels_prompt = unique_labels.unsqueeze(-1)
#             prepared_input[batch_idx].update({'labels': _labels_prompt.unsqueeze(-1)})
#         if args.point_prompt:
#             point_coords, point_labels = generate_point_prompt(_batch_labels, args)
#             prepared_input[batch_idx].update({
#                 'point_coords': point_coords,
#                 'point_labels': point_labels
#             })
#         if args.label_prompt and args.point_prompt:
#             if random.uniform(0, 1) < args.drop_label_prob:
#                 prepared_input[batch_idx].pop('labels')
#                 continue
#             if random.uniform(0, 1) < args.drop_point_prob:
#                 prepared_input[batch_idx].pop('point_coords')
#                 prepared_input[batch_idx].pop('point_labels')
#     return prepared_input, assign_device(batch_labels, args.rank), batch_labels, False
#
#
# def prepare_sam_test_input(inputs, labels, args, previous_pred=None) -> tuple[list[dict], Tensor, Tensor]:
#     """
#
#     :param inputs:
#     :param labels:
#     :param args:
#     :param previous_pred:
#     :return: a list of input data, batch labels and labels
#     """
#     unique_labels = torch.tensor([i for i in range(1, args.nc)])
#     unique_labels = assign_device(unique_labels, args.rank)
#     # preprocess make the size of lable same as high_res_logit
#     batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=1).float()
#     ic(labels.shape)
#     ic(batch_labels.shape)
#     ic(inputs.shape)
#     prepared_input = list()
#
#     for _inputs, _labels, _batch_labels in zip(inputs, labels, batch_labels):
#         pack = {
#             'image': _inputs,
#             'original_size': tuple(_labels.shape)
#         }
#         if args.label_prompt:
#             pack['labels'] = unique_labels.unsqueeze(-1)
#         if args.point_prompt:
#             point_coords, point_labels = generate_point_prompt(
#                 _batch_labels, args,
#                 points_pos=args.points_val_pos, points_neg=args.points_val_neg,
#                 previous_pred=previous_pred
#             )
#             pack['point_coords'] = point_coords
#             pack['point_labels'] = point_labels
#         prepared_input.append(pack)
#
#     # prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
#     # if args.label_prompt:
#     #     labels_prompt = unique_labels.unsqueeze(-1)
#     #     prepared_input[0].update({"labels": labels_prompt})
#     #
#     # if args.point_prompt:
#     #     point_coords, point_labels = generate_point_prompt(
#     #         batch_labels,
#     #         args,
#     #         points_pos=args.points_val_pos,
#     #         points_neg=args.points_val_neg,
#     #         previous_pred=previous_pred,
#     #     )
#     #     prepared_input[0].update({"point_coords": point_coords, "point_labels": point_labels})
#
#     return prepared_input, assign_device(batch_labels.unsqueeze(1), args.rank), unique_labels
#
#
# def prepare_sam_val_input_cp_only(inputs, labels, args):
#     """
#
#     @param inputs: A 3d tensor with shape B x roi_z_iter x H x W
#     @param labels: A 2d tensor with shape B x H x W
#     @param args:
#     @return:
#     """
#     # Don't exclude background in val but will ignore it in metric calculation
#     unique_labels = assign_device(torch.arange(1, args.nc).unsqueeze(0).repeat_interleave(inputs.shape[0], dim=0), args.rank)
#     ic(unique_labels.shape)
#     """
#         Some annotation for `batch_labels`
#         - preprocess make the size of label same as high_res_logit.
#         - As the result, just become the one-hot encoding.
#         - The shape is (nc - 1, H, W). nc - 1 is for skip background
#     """
#     # buffer = [labels == unique_labels[i] for i in range(len(unique_labels))]
#     # batch_labels = torch.stack(buffer, dim=1).float()
#     batch_labels = InputUtils.generate_prompt_labels(unique_labels, labels)
#     prepared_input = list()
#     for _inputs, _labels, _unique_labels in zip(inputs, labels, unique_labels):
#         pack = {
#             'image': _inputs,
#             'original_size': tuple(_labels.shape),
#             'labels': _unique_labels.unsqueeze(-1)
#         }
#         prepared_input.append(pack)
#     # prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
#     #
#     # labels_prompt = unique_labels.unsqueeze(-1)
#     # prepared_input[0].update({"labels": labels_prompt})
#
#     return prepared_input, assign_device(batch_labels, args.rank), unique_labels
