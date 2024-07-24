from unittest import TestCase
import torch
import numpy as np


class ImageSplitTest(TestCase):
    def test_function_raises_exception(self):
        n_slice = 27
        a = torch.randn((1, 1, 512, 512, 320))
        b = torch.randn_like(a)
        a = torch.nn.functional.pad(a, (n_slice // 2, n_slice // 2), mode='constant', value=0)
        # b = torch.nn.functional.pad(b, (n_slice // 2, n_slice // 2), mode='constant', value=0)
        a = a.squeeze(0).squeeze(0).unfold(-1, 27, 1).permute(2, 3, 0, 1)
        random_ids = torch.from_numpy(np.random.choice(a.shape[0], size=64, replace=False))
        a = a[random_ids]
        b = b[..., random_ids]


