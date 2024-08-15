from typing import Optional
import traceback
from operator import lt, gt
from functools import partial
import os

import numpy as np
import torch
import wandb
from icecream import ic

from utils import io as UIO


def is_worst(cur_metrics: float | torch.Tensor, history: list, mode: str) -> bool:
    if mode == 'min':
        last_metrics = max(history)
        compare_method = lt
    else:
        last_metrics = min(history)
        compare_method = gt

    return compare_method(cur_metrics, last_metrics)


class WorstDataRecord(object):
    maxlen: int
    metrics: list
    image_name: list
    label_name: list
    image_list: list[list[np.ndarray]]
    label_list: list[list[np.ndarray]]
    pred_list: list[list[np.ndarray]]


    def __init__(self, args, just_name: bool):
        self.args = args
        self.rest()
        self.maxlen = args.bad_image_maxlen
        self.mode = getattr(args, 'worst_mode', 'max')
        self.sorted = partial(sorted, reverse=self.mode == 'max')
        self.just_name = just_name
        self.LABELS = UIO.load_labels(getattr(args, 'label_map_path'))

    def rest(self):
        self.metrics = [-1]
        self.image_name = ["UNK"]
        self.label_name = ["UNK"]

        if not self.just_name:
            self.image_list = [None]
            self.label_list = [None]
            self.pred_list = [None]

    @torch.no_grad()
    def add(
            self, metrics: torch.Tensor | float, image_name, label_name,
            image_list: Optional[list[np.ndarray]] = None,
            label_list: Optional[list[np.ndarray]] = None,
            pred_list: Optional[list[np.ndarray]] = None
    ):
        """

        :param metrics:
        :param image_name :type str:
        :param label_name :type str:
        :param image_list :type Optional[list[np.ndarray]]:
        :param label_list:
        :param pred_list:
        :return:
        """
        if self.maxlen <= 0:
            return
        if torch.is_tensor(metrics):
            metrics = metrics.cpu().tolist()
        ic(self.metrics)
        if not isinstance(metrics, float):
            self._iter_add(metrics, [image_name, label_name])
            return
        if not is_worst(metrics, self.metrics, self.mode):
            return
        self.metrics.append(metrics)
        self.image_name.append(image_name)
        self.label_name.append(label_name)

        if not self.just_name:
            self.image_list.append(image_list)
            self.label_list.append(label_list)
            self.pred_list.append(pred_list)

        self._keep_maxlen()

    def _iter_add(self, metrics, file_name):
        for loss, (_image_name, _label_name) in zip(metrics.cpu(), file_name):
            if not is_worst(loss, self.metrics, self.mode):
                continue
            self.metrics.append(loss)
            self.image_name.append(_image_name)
            self.label_name.append(_label_name)

        self._keep_maxlen()

    def _keep_maxlen(self):
        # ic(len(self.metrics), len(self.image_name), len(self.label_name))
        if self.just_name:
            (
                self.metrics, self.image_name, self.label_name
            ) = map(list, zip(*self.sorted(list(zip(self.metrics, self.image_name, self.label_name)))))
            while len(self.metrics) > self.maxlen:
                _ = self.metrics.pop(0)
                _ = self.image_name.pop(0)
                _ = self.label_name.pop(0)
            return
        (
            self.metrics, self.image_name, self.label_name,
            self.image_list, self.label_list, self.pred_list
        ) = map(list, zip(*self.sorted(list(zip(
            self.metrics, self.image_name, self.label_name,
            self.image_list, self.label_list, self.pred_list
        )))))

        while len(self.metrics) > self.maxlen:
            _ = self.metrics.pop(0)
            _ = self.image_name.pop(0)
            _ = self.label_name.pop(0)
            _ = self.image_list.pop(0)
            _ = self.label_list.pop(0)
            _ = self.pred_list.pop(0)

    def store(self, epoch: int, run: Optional[wandb.run] = None):
        if self.maxlen <= 0:
            return
        store_folder = getattr(self.args, 'logdir', './')
        path = os.path.join(store_folder, f'the_worse_sample.json')

        pack = [{'loss': loss, 'image': image_path, 'label': label_path} for loss, image_path, label_path in
                zip(self.metrics, self.image_name, self.label_name)]
        new_content: list[dict] = [{
            'epoch': int(epoch),
            'pack': pack
        }]
        try:
            UIO.save_continue_json(path, new_content)
        except Exception as e:
            traceback.format_exc()
            print(f'Record Error')

        if not self.just_name:
            self.upload(epoch, run)


    def upload(self, epoch: int, run: Optional[wandb.run] = None):
        worst_zip: zip = zip(
            self.metrics, [self.image_name, self.label_name],
            self.image_list, self.pred_list, self.label_list
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