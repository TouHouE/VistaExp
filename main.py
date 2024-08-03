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
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import sys
import warnings
from subprocess import Popen

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from monai.utils.enums import MetricReduction
from icecream import ic

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from engine import builtin_engine as NormEngine
from engine import poor_vram_engine as PoorEngine
from utils.data_utils import get_loader
from utils import get_args
from models import vista_model_registry
from utils import terminate as Terminator
from utils import asker as Asker


def start_tb(log_dir):
    cmd = ["tensorboard", "--logdir", log_dir]
    Popen(cmd, stderr=sys.stderr, stdout=sys.stdout, shell=False)


def main():
    args = get_args()
    if not args.test_mode:
        ic.disable()
        args.logdir = Asker.ask_logdir_root(args)
        args.cache = Asker.ask_cache_root(args)
    args.amp = not args.noamp

    if args.seed > -1:
        set_determinism(seed=args.seed)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    model = vista_model_registry[args.sam_base_model](
        args.sam_pretrain_ckpt,
        image_size=args.sam_image_size,
        encoder_in_chans=args.roi_z_iter * 3,
        patch_embed_3d=args.patch_embed_3d,
        vae=args.vae
    )
    if (lf := args.loss_func) == 'dice_ce':
        dice_loss = DiceCELoss(sigmoid=True)
    elif lf == 'dice_focal':
        dice_loss = DiceFocalLoss(sigmoid=True)
    else:
        raise NotImplementedError(f'Loss: {lf} not implement now.')

    post_label = AsDiscrete(to_onehot=args.nc)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_acc = DiceMetric(include_background=args.eval_bg, reduction=MetricReduction.MEAN, get_not_nans=True)

    Terminator.show_model_info(model, args)
    Terminator.show_some_hyper(args)

    best_acc = 0
    start_epoch = 0
    optimizer_state = None
    # Doing checkpoint recover
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k] = v
        if args.pop_pos_embed:
            print("pop_pos_embed")
            new_state_dict.pop("image_encoder.patch_embed.proj.weight")
            new_state_dict.pop("image_encoder.patch_embed.proj.bias")
            model.load_state_dict(new_state_dict, strict=False)
        elif args.pop_point_embed:
            print("pop_point_embed")
            new_state_dict.pop("prompt_encoder.point_embeddings.0.weight")
            new_state_dict.pop("prompt_encoder.point_embeddings.1.weight")
            new_state_dict.pop("prompt_encoder.point_embeddings.2.weight")
            new_state_dict.pop("prompt_encoder.point_embeddings.3.weight")
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=True)
        if args.resume_ckpt:
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            if "best_acc" in checkpoint:
                best_acc = checkpoint["best_acc"]
            if "optimizer" in checkpoint:
                optimizer_state = checkpoint["optimizer"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        # override lr by the given value
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.optim_lr

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    if args.poor_mode:
        run_training = PoorEngine.run_training
    else:
        run_training = NormEngine.run_training
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
