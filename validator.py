import argparse
from collections import OrderedDict
import gc
import json
import logging
from typing import Callable
import re
import shutil
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import hydra
from icecream import ic
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import monai
from monai import transforms as MT
from monai.metrics import DiceHelper, MeanIoU, compute_dice, compute_iou, get_confusion_matrix
from monai.data import decollate_batch
# compute_dice()
import numpy as np
from omegaconf import DictConfig, OmegaConf

from models.factory.vista_factory import vista_model_registry
from utils import assign_device
from utils import model_input as ModelInputer
from utils.io import load_json, save_json
from engine.utils import find_executable_batch_size
ic.disable()

def clean_cuda(obj_in_cuda):
    del obj_in_cuda
    gc.collect()
    torch.cuda.empty_cache()
    return None


def load_model(cfg: DictConfig) -> Callable:
    print(cfg['model'])
    if cfg['debug']['enable']:
        if (no_model := getattr(cfg['debug'], 'no_model', False)):
            return lambda x: [{
                'high_res_logits': torch.randn((1, cfg['nc'], cfg['image_size'], cfg['image_size']))
            } for _ in range(len(x))]
    old_state_dict = torch.load(cfg['model']['checkpoint'], map_location='cpu')
    weight_dict = OrderedDict()
    print(old_state_dict.keys())
    if (_mapper := old_state_dict.get('state_dict')) is not None:
        for k, v in _mapper.items():
            weight_dict[k] = v
    else:
        weight_dict = old_state_dict.copy()
    model = vista_model_registry[getattr(cfg['model'], 'model_size', 'vit_b')](
        **cfg['model']['init_kwargs']
    )
    model.load_state_dict(weight_dict)

    return model.to(getattr(cfg, 'device', 'cpu'))


def load_dataset(cfg: DictConfig) -> list[dict]:
    root_dir = cfg['input']['root_dir']
    data_json_path = cfg['input']['data_json']
    data_map = load_json(data_json_path)
    if getattr(cfg['debug'], 'dataset', False) is True:
        return data_map[:2]
    if (phase := getattr(cfg['input'], 'phase', 'test')) == 'test':
        data_list = data_map['testing']
    elif phase == 'all':
        data_list = data_map['training'] + data_map['testing']
    elif phase == 'val':
        fold = getattr(cfg['input'], 'fold', 0)
        data_list = list(filter(lambda _pack: _pack['fold'] == fold, data_map['training']))
    elif phase == 'train':
        fold = getattr(cfg['input'], 'fold', 0)
        data_list = list(filter(lambda _pack: _pack['fold'] != fold, data_map['training']))
    else:
        raise NotImplementedError(f'Unknown phase: {phase}')
    for pack in data_list:
        if pack['image'][0] in ['/', r'\\']:
            pack['image'] = pack['image'][1:]
        if pack['label'][0] in ['/', r'\\']:
            pack['label'] = pack['label'][1:]
        pack['image'] = os.path.join(root_dir, pack['image'])
        pack['label'] = os.path.join(root_dir, pack['label'])
    if cfg['debug']['enable']:
        if cfg['debug']['small_dataset']:
            return data_list[:1]
    return data_list


@torch.no_grad()
def iter_slice(batch_size, patch_image, patch_label, model, poster: Callable, cfg, **kwargs):
    # ic(patch_image.shape)
    if (pm := cfg.get('prepare_method', 'val')) == 'val':
        indices_pack: tuple[torch.Tensor, ...] = torch.split(torch.arange(0, patch_image.shape[0]), batch_size)
    else:
        indices_pack: torch.Tensor = torch.arange(0, patch_image.shape[0])

    predict_collections: list = list()
    args = argparse.Namespace(
        nc=cfg['nc'],
        rank=cfg['device'], **cfg.get('args', dict())
        # label_prompt=cfg.get('label_prompt', True), point_prompt=cfg.get('point_prompt', True),
        # points_val_pos=cfg['points_val_pos'], points_val_neg=cfg['points_val_neg']
    )
    history = None

    for indices in tqdm(indices_pack, total=len(indices_pack)):
        sub_image = patch_image[indices]
        sub_label = patch_label[indices]
        if pm == 'val':
            data, *useless = ModelInputer.prepare_sam_val_input_cp_only(
                assign_device(sub_image, args.rank), assign_device(sub_label, args.rank), args
            )
        elif pm == 'test' and history is None:
            sub_image = sub_image.unsqueeze(0)
            sub_label = sub_label.unsqueeze(0)
            data, *useless = ModelInputer.prepare_sam_test_input(
                assign_device(sub_image, args.rank), assign_device(sub_label, args.rank), args
            )
        else:
            sub_image = sub_image.unsqueeze(0)
            sub_label = sub_label.unsqueeze(0)
            data, *useless = ModelInputer.prepare_sam_test_input(
                assign_device(sub_image, args.rank), assign_device(sub_label, args.rank), args, previous_pred=history
            )

        # print(data[0]['original_size'])
        clean_cuda(useless)
        outputs = model(data)
        multi_slice_digits_mask: torch.Tensor = torch.cat([output['high_res_logits'] for output in outputs], dim=1)
        # clean_cuda(outputs)
        ic(multi_slice_digits_mask.shape)
        if (hae := getattr(multi_slice_digits_mask, 'as_tensor', None)) is not None:
            multi_slice_digits_mask: monai.data.MetaTensor
            multi_slice_digits_mask: torch.Tensor = multi_slice_digits_mask.as_tensor()
        # print(hae)
        multi_slice_one_hot_mask = torch.stack(poster(decollate_batch(multi_slice_digits_mask)), dim=0)
        history = multi_slice_one_hot_mask
        # ic(one_hot_batch_mask.shape)
        # clean_cuda(batch_digits_mask)
        predict_collections.append(multi_slice_one_hot_mask.cpu())
        # breakpoint()
    # print(*[x.shape for x in predict_collections], sep=',')
    final_predict = torch.cat(predict_collections, dim=1).permute(0, 2, 3, 1)
    if len(final_predict.shape) < 5:
        return final_predict.unsqueeze(0)
    return final_predict


def compute_all_metrics(y_pred, y_gt, cfg: DictConfig):
    """

    :param y_pred: B x N x H x W x S, y_pred is one-hot encoded
    :param y_gt: B x H x W x S, y_gt store digits label
    :param cfg:
    :return:
    """
    print(y_pred.shape)
    print(y_gt.shape)
    # make sure batch axis is exist
    if len(y_pred.shape) < 5:
        y_pred = y_pred.unsqueeze(0)
    if len(y_gt.shape) < 4:
        y_gt = y_gt.unsqueeze(0)

    nc = cfg['nc']
    onehot_gt = torch.stack([(y_gt == category).long() for category in range(1, nc)], dim=1)
    batch_dice = compute_dice(y_pred, onehot_gt)    # B x Nc
    batch_miou = compute_iou(y_pred, onehot_gt)  # B x Nc
    batch_cm = get_confusion_matrix(y_pred, onehot_gt)   # Bx Nc x 4
    return batch_dice, batch_miou, batch_cm


def validate(model: nn.Module, data_list: list[dict], cfg: DictConfig):
    keys = ['image', 'label']
    image_size = cfg['image_size']
    z_roi = cfg['z_roi_iter']
    threshold = cfg['threshold']
    padding_size = (z_roi // 2, z_roi // 2)
    loader: Callable = MT.Compose([
        MT.LoadImaged(keys, allow_missing_keys=True),
        MT.EnsureChannelFirstd(keys, allow_missing_keys=True),
        MT.Orientationd(keys, axcodes='RAS', allow_missing_keys=True),
        MT.ResizeWithPadOrCropd(keys, spatial_size=(image_size, image_size, -1), method='end', mode='minimum', allow_missing_keys=True),
        MT.ScaleIntensityRanged(keys=['image'], **cfg['preprocessor']['scale'])
    ])
    poster: Callable = MT.Compose([
        MT.Activations(sigmoid=True),
        MT.AsDiscrete(threshold=threshold)
    ])
    output_dir = getattr(cfg, 'output_dir', './output')
    saver: Callable = MT.SaveImage(
        output_dir=os.path.join(output_dir, 'predict'), output_postfix=getattr(cfg, 'output_postfix', 'predict')
    )
    slicePader: Callable = lambda x: F.pad(x, padding_size, 'constant', 0)
    auto_size_iter_slice: Callable = find_executable_batch_size(iter_slice, getattr(cfg, 'batch_size', 32))
    summary = {
        'metrics_by_cases': list()
    }

    for predict_id, pack in enumerate(data_list):
        data: dict = loader(pack)
        plan_image: monai.data.MetaTensor = data['image']
        image_name = plan_image.meta["filename_or_obj"]
        image_name_no_ext = re.split(r'[/\\]', image_name)[-1].replace('.nii.gz', '')
        plan_label: monai.data.MetaTensor | torch.Tensor = data.get('label', torch.zeros_like(plan_image))
        if (meta := getattr(plan_label, 'meta')) is not None:
            label_name = meta['filename_or_obj']
            label_name_no_ext = re.split(r'[/\\]', label_name)[-1].replace('.nii.gz', '')
        else:
            label_name = 'none'
            label_name_no_ext = 'none'

        image = slicePader(plan_image).squeeze().unfold(-1, z_roi, 1).permute(2, 3, 0, 1).contiguous()
        label = plan_label.squeeze()
        with autocast(enabled=cfg['amp']):
            predict_mask: torch.Tensor = auto_size_iter_slice(image, label.permute(2, 0, 1).contiguous(), model, poster, cfg)
            # breakpoint()
        if getattr(cfg, 'save_mask', False):
            pred_shape = predict_mask.shape[-3:]
            bg = torch.zeros((1, *pred_shape))
            save_mask = monai.data.MetaTensor(
                # B x nc - 1 x H x W x S -> nc - 1 x H x W x S -> nc x H x W x S
                torch.argmax(torch.cat([bg, predict_mask.squeeze(0)], dim=0), dim=0),
                affine=plan_image.meta['affine'], meta=plan_image.meta
            )            
            saver(save_mask, meta_data=plan_image.meta)
        dice, iou, cm = compute_all_metrics(predict_mask, label, cfg)

        for bdice, biou, bcm in zip(dice, iou, cm):
            m_by_case = {
                'image name': image_name,
                'label name': label_name,
                'predit id': predict_id,
                'metrics': {
                    str(c + 1): {
                        'Dice': 0 if torch.isnan(cdice) else cdice.item(),
                        f'IoU@{threshold}': 0 if torch.isnan(ciou) else ciou.item(),
                        'TP': 0 if torch.isnan(ccm[0]) else ccm[0].item(),
                        'FP': 0 if torch.isnan(ccm[1]) else ccm[1].item(),
                        'FN': 0 if torch.isnan(ccm[2]) else ccm[2].item(),
                        'TN': 0 if torch.isnan(ccm[3]) else ccm[3].item()
                    } for c, (cdice, ciou, ccm) in enumerate(zip(bdice, biou, bcm))
                }
            }
            summary['metrics_by_cases'].append(m_by_case)
        # print(json.dumps(summary, indent=2))
        mean_dict = {str(i): {key: list() for key in ['Dice', f'IoU@{threshold}', 'TP', 'FP', 'FN', 'TN']} for i in range(1, cfg['nc'])}
        for case in summary['metrics_by_cases']:
            for digit, m_pack in case['metrics'].items():
                for indicator_name, indicator_value in m_pack.items():
                    mean_dict[digit][indicator_name].append(indicator_value)
        for digit in range(1, cfg['nc']):
            digit = str(digit)
            mean_dict[digit] = {
                indicator_name: sum(v / len(value_collections) for v in value_collections)
                for indicator_name, value_collections in mean_dict[digit].items()
            }
            # for indicator_name, value_collections in mean_dict[digit].items():
            #     denominator = len(value_collections)
            #     mean_dict[digit] = sum(value / denominator for value in value_collections)
            # print(m_pack)

        summary['mean'] = mean_dict
        save_json(os.path.join(
            cfg['output_dir'], 'summary.json'
        ), summary)


@hydra.main(config_path='./conf', config_name='val.yaml')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg['output_dir'] = getattr(cfg, 'output_dir', './output_dir')
    model = load_model(cfg)
    data_list = load_dataset(cfg)
    try:
        validate(model, data_list, cfg=cfg)
    except KeyboardInterrupt or Exception:
        if cfg['debug']['enable']:
            shutil.rmtree('./outputs')



if __name__ == '__main__':
    main()
