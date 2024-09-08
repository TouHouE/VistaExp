import argparse
import os

from numba.cuda.libdevicedecl import args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from typing import Optional, Sequence
import logging
import random

from monai.data import MetaTensor
from monai.utils.type_conversion import convert_to_tensor
import numpy as np
import torch


def intersections_element(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """

    :param tensor_a: The shape mu
    :param tensor_b:
    :return:
    """
    return tensor_a[(tensor_a.view(-1, 1) == tensor_b.view(1, -1)).any(dim=1)]


# def different_element(tensor_a:)


def get_unique_labels(
        unique_labels: list[torch.Tensor | MetaTensor], poor_categories: Optional[list[int]] = None
) -> list[torch.LongTensor]:
    if poor_categories is not None:
        poor_categories = torch.as_tensor(poor_categories)
        unique_labels: list[torch.Tensor] = [intersections_element(ul, poor_categories) for ul in unique_labels]
        logging.info('Now only provides category hints for challenging training samples.')
        logging.debug(f"Poor categories: {poor_categories}")

    unique_labels: list[torch.LongTensor] = [convert_to_tensor(ul, dtype=torch.long) for ul in unique_labels]
    return unique_labels


def keep_label_prompt_size(raw_labels: list[torch.Tensor], dst_size: int) -> list[torch.Tensor]:
    label_prompt: list[torch.Tensor] = list()
    for raw_label in raw_labels:

        if (org_size := len(raw_label)) < dst_size:
            rate = org_size // dst_size + int(org_size % dst_size != 0)
            prefix_label_prompt = torch.cat([raw_label] * rate, dim=0).long()[:dst_size]
        elif org_size > dst_size:
            prefix_label_prompt = torch.from_numpy(np.random.choice(raw_label, dst_size, replace=True).astype(np.int64))
        else:
            prefix_label_prompt = raw_label
        label_prompt.append(prefix_label_prompt)

    return label_prompt


def append_not_exist_label_prompt(org_label_prompts, args: argparse.Namespace, num_append_element: int = 4) -> list[torch.Tensor] | torch.Tensor:
    """
        The return will be N of torch.Tensor, the N represent batch size
    :param org_label_prompts:
    :param args :type: Namespace: Contains nc: num of categories
    :param num_append_element: How many label prompt used to append not
    :return:
    """
    whole_labels = set(range(1, args.nc))
    batch_size = len(org_label_prompts)

    for bs in range(batch_size):
        org_label_prompt = org_label_prompts[bs]
        not_exist_labels = list(whole_labels - set(org_label_prompt.cpu().numpy()))
        random.shuffle(not_exist_labels)
        org_label_prompts[bs] = torch.cat([org_label_prompt, torch.as_tensor(not_exist_labels[:num_append_element])]).long()

    return org_label_prompts


def generate_prompt_labels(label_prompts, label_digits, args: argparse.Namespace=None) -> torch.Tensor:
    labels = list()

    for label_prompt, label_digit in zip(label_prompts, label_digits):
        labels.append(torch.stack([(label_digit == lp).float() for lp in label_prompt], dim=0))
        # breakpoint()
    return torch.stack(labels, dim=0)



