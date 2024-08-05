from typing import Type, Callable

import torch
from torch import nn
from torch.nn import functional as F
import monai
from monai.data import MetaTensor

from models.common import LayerNorm2d
PATCH_SIZE = 16


class NormalDecoder(nn.Module):
    def __init__(self, channels, embed_dim, neck_out_dim, sim_func: Type[nn.Module] | Callable = nn.MSELoss, **kwargs):
        super().__init__()
        self.decode = nn.Linear(neck_out_dim, embed_dim)
        self.poster = nn.Conv2d(channels, channels, 3, padding=1)
        if isinstance(sim_func, Callable):
            self.sim_func = sim_func
        else:
            self.sim_func = sim_func(**kwargs.get('sim_func_param', dict()))

    def forward(self, image_embedding, input_image):
        """

        :param image_embedding: [B, neck_out=256, H / p, W / p]
        :param input_image: [z_roi_iter, H, W]
        :return:
        """
        B, neck_out, Hp, Wp = image_embedding.shape
        embryo_image = self.decode(image_embedding)
        embryo_image = embryo_image.view(B, neck_out // 256, Hp * 16, Wp * 16)
        pseudo_image = self.poster(embryo_image)
        sim_loss = self.sim_func(pseudo_image[0], input_image)
        return sim_loss

