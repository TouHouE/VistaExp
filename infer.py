import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import matplotlib.pyplot as plt

from models.factory.vista_factory import vista_model_registry
from utils import model_input as ModelInputer


def load_model(ckpt_path, model_size, init_param: dict):
    model = vista_model_registry[model_size](checkpoint=ckpt_path, **init_param)
    return model


def main():
    args = argparse.Namespace(
        image_size=128, encoder_in_chans=81, patch_embed_3d=True,
        point_prompt=False, label_prompt=False, nc=11, rank=0
    )
    bs = 2
    image = torch.randn(bs, args.encoder_in_chans // 3, args.image_size, args.image_size).cuda(args.rank)
    label = torch.randint(0, 10, (bs, args.image_size, args.image_size)).cuda(args.rank)
    model = load_model(ckpt_path=None, model_size='vit_b', init_param=args.__dict__).cuda(args.rank)
    model.eval()
    with torch.no_grad():
        data, target, _ = ModelInputer.prepare_sam_val_input_cp_only(image, label, args)
        # breakpoint()
        pred = model(data)







if __name__ == '__main__':
    main()