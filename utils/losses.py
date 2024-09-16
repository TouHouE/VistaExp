from copy import deepcopy
from typing import Sequence, Callable, Tuple, Union, Optional
from functools import partial
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from skimage.morphology import skeletonize
import torch
from torch import nn
from torch.nn import functional as F
import monai
from monai import losses as ML


def _generate_center_kernel(dim, size, background_value=1, center_value=0):
    _kernel = background_value * torch.ones(*[size] * dim, dtype=torch.float32)
    _kernel[tuple(size // 2 for _ in range(dim))] = center_value
    return _kernel


class GapLoss(nn.Module):
    def __init__(
            self,
            loss_map_func: Callable,
            binarize_threshold=0.5, endpoint_threshold=8, K=60,
            data_dim=2
    ):
        super(GapLoss, self).__init__()
        self.K = K
        self.loss_map_func = loss_map_func
        self.binarize_threshold = binarize_threshold
        self.neighbor_threshold = endpoint_threshold

        self.conv_method: Callable = getattr(F, f'conv{data_dim}d')
        self.data_dim = data_dim
        if data_dim == 2:
            _kernel = torch.ones((1, 1, 3, 3))
            _kernel[..., 1, 1] = 0
        else:
            _kernel = torch.ones((1, 1, 3, 3, 3))
            _kernel[..., 1, 1, 1] = 0
        self.endpoint_kernel = nn.Parameter(
            _generate_center_kernel(data_dim, 3, 1, 0).view(1, 1, *[3] * data_dim),
            requires_grad=False
        )
        self.neighbor_kernel = nn.Parameter(
            _generate_center_kernel(data_dim, 9, 1, 0).view(1, 1, *[9] * data_dim),
            requires_grad=False
        )

    def binarize(self, x):
        x = torch.softmax(x, dim=0)
        x[x < self.binarize_threshold] = 0
        x[x >= self.binarize_threshold] = 1
        return x

    def calculate_endpoint(self, x_skeleton):
        print(x_skeleton.dtype)
        bs = x_skeleton.shape[0]
        image_shape = x_skeleton.shape[-self.data_dim:]

        c_map = self.conv_method(
            x_skeleton.view(-1, 1, *image_shape), self.endpoint_kernel, padding=1, stride=1
        ).view(bs, -1, *image_shape)
        # c[c_map >= self.neighbor_threshold] = 1
        return (c_map >= self.neighbor_threshold).float()

    def get_weight_map(self, x):
        bs = x.shape[0]
        image_shape = x.shape[-self.data_dim:]
        weight_map = self.conv_method(
            x.view(-1, 1, *image_shape), self.neighbor_kernel, padding=4, stride=1
        ).view(bs, -1, *image_shape) * self.K
        return weight_map

    def forward(self, x: torch.Tensor, y: torch.Tensor, label_prompts: Optional[torch.Tensor] = None):
        """

        :param x: [B, Np, H, W]
        :param y: [B, Np, H, W]
        :param label_prompts :type Optional: [B, Np, 1]
        :return: a value
        """
        loss_map = self.loss_map_func(x, y)
        A = self.binarize(x).detach().cpu().numpy()
        B = torch.stack([torch.from_numpy(skeletonize(a)) for a in A], dim=0).float()
        C = self.calculate_endpoint(B)
        W = self.get_weight_map(C)
        return torch.mean(loss_map * W)


if __name__ == '__main__':
    loss_func = GapLoss(road_like_label_digits=[8, 9, 10], loss_map_func=monai.losses.DiceFocalLoss(softmax=True, batch=True, reduction='none'))
    x = torch.randn((2, 3, 45, 45))
    y = torch.randint(0, 2, (2, 3, 45, 45))
    label_prompts = torch.randint(1, 11, (2, 3, 1))
    loss_map = loss_func(x, y)
    print(loss_map)