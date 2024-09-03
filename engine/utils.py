# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import functools
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import logging

LOG_FORMAT = '[%(asctime)s %(levelname)s %(filename)s:%(lineno)d (%(funcName)s)] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
from typing import Type, Optional, Callable, Literal
import inspect
import gc

import monai
from monai import transforms as MT
import numpy as np
import scipy.ndimage as ndimage
import torch
import wandb.wandb_run
from wandb.wandb_run import Run
from icecream import ic

from utils import io as UIO


def select_random_ids(selectable_range: range, args: argparse.Namespace) -> torch.Tensor:
    if (n_sample := args.num_patch) >= (selectable_size := len(selectable_range)):
        return torch.as_tensor(selectable_range)

    return torch.from_numpy(
        np.random.choice(selectable_size, size=n_sample, replace=False)
    )


def change_drop_prob(args, epoch) -> bool:
    if not args.label_prompt or not args.point_prompt:
        return False
    args.drop_point_prob = .5
    if epoch < args.label_prompt_warm_up_epoch:
        args.drop_label_prob = .2
    else:
        args.drop_label_prob = .5
    return True


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    val: int | float
    avg: int | float
    sum: int | float
    count: int | float
    worst_sample_name: list
    worst_val: list
    worst_pred: list[list]
    worst_label: list[list]
    worst_image: list[list]

    def __init__(self, args: Optional[Type[argparse.Namespace]] = None):
        self.reset()
        self.args = args
        if args is not None:
            self.LABELS = UIO.load_labels(getattr(args, 'label_map_path'))
        else:
            self.LABELS = None

    pass

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.worst_sample_name = ['UNK']
        self.worst_val = [1000]
        self.worst_pred = [[]]
        self.worst_label = [[]]
        self.worst_image = [[]]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

    def add(
            self, val_acc: Optional[float] = None,
            sample_name: Optional[list[str, str]] = None,
            buf_image: Optional[Type[np.ndarray]] = None,
            buf_pred: Optional[Type[np.ndarray]] = None,
            buf_label: Optional[Type[np.ndarray]] = None
    ):
        """

        :param val_acc(Optional[float]): The metrics value for current sample.
        :param sample_name
        :param buf_image:
        :param buf_pred:
        :param buf_label:
        :return:
        """
        if self.args is None:
            return
        # Not so bad.
        if val_acc >= max(self.worst_val):
            return
        ic(self.worst_val)
        self.worst_val.append(val_acc)
        self.worst_sample_name.append(sample_name)
        self.worst_image.append(buf_image)
        self.worst_pred.append(buf_pred)
        self.worst_label.append(buf_label)

        (
            self.worst_val, self.worst_sample_name,
            self.worst_image, self.worst_pred, self.worst_label
        ) = map(list, (zip(*sorted(list(zip(
            self.worst_val, self.worst_sample_name,
            self.worst_image, self.worst_pred, self.worst_label
        )), reverse=True))))

        while len(self.worst_val) > 5:
            _ = self.worst_sample_name.pop(0)
            _ = self.worst_val.pop(0)
            _ = self.worst_image.pop(0)
            _ = self.worst_pred.pop(0)
            _ = self.worst_label.pop(0)

    def log_worst(self, run: Type[wandb.wandb_run.Run], epoch: int):
        if len(self.worst_val) <= 0:
            print(f'No data in list, can\'t upload any thing')
            return
        if run is None:
            print(f'Wandb Logger cannot be None.')
            return
        worst_zip: zip = zip(
            self.worst_val, self.worst_sample_name,
            self.worst_image, self.worst_pred, self.worst_label
        )
        for tot_dice, fname, image_list, pred_list, label_list in worst_zip:
            for slice_idx, (image, pred, label) in enumerate(zip(image_list, pred_list, label_list)):
                mask_pack = {
                    'predictions': {
                        'mask_data': label,
                        'class_labels': self.LABELS
                    },
                    'ground_truth': {
                        'mask_data': label,
                        'class_labels': self.LABELS
                    }

                }
                image_obj = wandb.Image(
                    image,
                    masks=mask_pack,
                    caption=f'Slice:{slice_idx}, file:{fname}'
                )
                run.log({
                    f'Worst Valid-{tot_dice:.3f}': image_obj,
                    'epoch': epoch
                })


class RandomPermute:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        img_size = args.sam_image_size
        self.do_permute = self.args.random_permute
        logging.info('Permuter not enable')
        self.permute_prob = self.args.permute_prob
        self.image_keeper = MT.Compose(
            [MT.ResizeWithPadOrCrop(spatial_size=(img_size, img_size, -1), method='end', mode='minimum')])
        self.label_keeper = MT.Compose(
            [MT.ResizeWithPadOrCrop(spatial_size=(img_size, img_size, -1), method='end', mode='constant', value=0)])

    def __call__(self, image: torch.Tensor | monai.data.MetaTensor, label: Optional = None):
        if not self.do_permute:
            return image, label
        if (rand_var := torch.rand(1)) > self.permute_prob:
            return image, label
        print(f'Do random axes from HWS -> ', end='')
        n_axes = len(image.shape)
        new_indices = torch.arange(n_axes)
        new_HWS = torch.randperm(3)
        print(''.join("HWS"[i] for i in new_HWS))
        new_indices[-3:] = new_HWS + n_axes - 3
        ic(image.shape)
        image = image.permute(*new_indices)
        image = torch.stack([self.image_keeper(_image) for _image in image], dim=0)
        if label is not None:
            ic(label.shape)
            label = label.permute(*new_indices)
            label = torch.stack([self.label_keeper(_label) for _label in label], dim=0)
        return image, label


class TrainingAlgoManager:
    def __init__(
            self, algo_map: dict[int, Callable],
            mode: Literal['min', 'max'] = 'max', patience: int = 10,
            enable_eta: float = 1e-4, eta_mode: Literal['rel', 'abs'] = 'rel',
            cooldown: int = 0, run: Optional[Run] = None
    ):
        self.algo_map = algo_map
        self.mode = mode
        self.patience = patience
        self.threshold = enable_eta
        self.threshold_mode = eta_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.current_stage = 0
        self.final_stage = max(algo_map.keys())
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.run = run

    def is_better(self, metrics, best):
        metrics = float(metrics)

        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_eps = 1 - self.threshold
            return metrics < best * rel_eps
        if self.mode == 'min' and self.threshold_mode == 'abs':
            return metrics < best - self.threshold
        if self.mode == 'max' and self.threshold_mode == 'rel':
            rel_eps = 1 + self.threshold
            return metrics > best * rel_eps

        # when mode == 'max' and threshold_mode == 'abs'
        return metrics > best + self.threshold

    def next_algo(self, metrics: float, last_epoch: Optional[int] = None) -> Callable:
        if self.current_stage == self.final_stage:
            logging.info(f'Next stage index is: {self.current_stage}')
            return self.algo_map[self.current_stage]

        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self.current_stage += 1
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        self.log({
            'epoch': last_epoch,
            'stage': self.current_stage
        })
        logging.info(f'Next stage index is: {self.current_stage}')
        return self.algo_map[self.current_stage]

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def log(self, pack: dict) -> None:
        if self.run is None:
            return
        self.run.log(pack)


def distributed_all_gather(
        tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


"""
 Those method are reference by accelerate module, used to automatic decision executable batch size.
"""


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from accelerate.utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        logging.info(f'Initial batch_size as: {starting_batch_size}')
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    logging.warning(f'Scale batch size to {batch_size}')
                else:
                    raise

    return decorator
