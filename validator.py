from typing import Callable
import argparse
import os


from omegaconf import DictConfig, OmegaConf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import hydra
import torch
from torch import nn
import monai
from monai import transforms as MT

from models.factory.vista_factory import vista_model_registry
from utils import model_input as ModelInputer
from utils.io import load_json

def load_model(cfg: DictConfig) -> nn.Module:
    model = vista_model_registry[getattr(cfg['model'], 'model_size', 'vit_b')](
        **cfg['model']['init_kwargs']
    ).to(getattr(cfg, 'device', 'cpu'))
    return model


def load_dataset(cfg: DictConfig) -> list[dict]:
    root_dir = cfg['input']['root_dir']
    data_json_path = cfg['input']['data_json']
    data_map = load_json(data_json_path)
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
        pack['image'] = os.path.join(root_dir, pack['image'])
        pack['label'] = os.path.join(root_dir, pack['label'])
    return data_list


def validate(model: nn.Module, data_list: list[dict], cfg: DictConfig) ->
    keys = ['image', 'label']
    preprocessor: Callable = MT.Compose([
        MT.LoadImaged(keys),
        MT.EnsureShape(keys),
        MT.Orientationd(keys, axcodes='RAS'),
        MT.ResizeWithPadOrCropd(keys, spatial_size=(cfg['image_size'], cfg['image_size'], -1), )
    ])

    for pack in data_list:



@hydra.main(config_path='./conf', config_name='val.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg)
    print(OmegaConf.to_yaml(cfg))




if __name__ == '__main__':
    main()
