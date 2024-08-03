# Copyright 2020 - 2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os

import numpy as np
import torch
from monai import data, transforms
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai import transforms as MF


def get_transforms(args):
    if args.data_aug:
        random_transforms = ([
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3, allow_missing_keys=True),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.10)
        ])
        print("using data augmentation")
    else:
        random_transforms = ([])
        print("No data augmentation")

    return random_transforms


def get_ds(args, phase, datalist, transform) -> data.Dataset | data.CacheDataset | data.PersistentDataset:
    dataset_type = args.dataset_type
    if args.dataset_type == 'normal':
        if phase == 'train':
            ds = data.Dataset(
                datalist,
                transform=transform
            )
            return ds
        else:
            dataset_type = 'persis' if args.poor_mode else 'cache'

    if args.distributed:
        datalist = data.partition_dataset(
            data=datalist,
            shuffle=phase == 'train',
            num_partitions=args.world_size,
            even_divisible=phase == 'train',
        )[args.rank]

    if dataset_type == 'cache':
        ds = data.CacheDataset(
            datalist,
            transform=transform,
            cache_rate=1,
            num_workers=args.workers,
        )
    elif dataset_type == 'persis':
        ds = data.PersistentDataset(
            datalist,
            transform=transform,
            cache_dir=os.path.join(args.cache, phase)
        )
    else:
        raise NotImplementedError(f'Dataset type: {dataset_type} not implement.')
    return ds


def get_loader(args):
    train_files, val_files, test_files = split_data(args)
    random_transforms = get_transforms(args)
    resizer = MF.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=(args.sam_image_size, args.sam_image_size, 320), allow_missing_keys=True)
    if args.poor_mode:
        print(f'Resizer change from {resizer.get_transform_info()} with {resizer.padder.padder.spatial_size}', end=' => ')
        resizer = MF.SpatialPadd(
            keys=['image', 'label'], allow_missing_keys=True,
            spatial_size=(args.sam_image_size, args.sam_image_size, -1), mode='minimum'
        )
        print(f'{resizer.spatial_size}')
        # resizer = MF.Identityd(keys=['image', 'label'], allow_missing_keys=True)

    train_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True, allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
            Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True),
            Spacingd(keys=["image", "label"], pixdim=args.pixdim, mode=("bilinear", "nearest"), allow_missing_keys=True),
            resizer,
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
        ]
        + random_transforms
    )

    val_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True, allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
            Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True),
            Spacingd(keys=["image", "label"], pixdim=args.pixdim, mode=("bilinear", "nearest"), allow_missing_keys=True),
            resizer,
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
        ]
    )

    datalist = train_files
    train_ds = get_ds(args, 'train', datalist, train_transform)
    train_sampler = None

    train_loader = data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, pin_memory=True
    )
    val_files = val_files
    val_ds = get_ds(args, 'val', val_files, val_transform)
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=None, pin_memory=True
    )
    loader = [train_loader, val_loader]
    return loader


def split_data(args):
    data_dir = args.data_dir
    import json

    with open(args.json_list, "r") as f:
        json_data = json.load(f)

    list_train = []
    list_valid = []
    if "validation" in json_data.keys():
        list_train = json_data["training"]
        list_valid = json_data["validation"]
        list_test = json_data["testing"]
    else:
        for item in json_data["training"]:
            if item["fold"] == args.fold:
                item.pop("fold", None)
                list_valid.append(item)
            else:
                item.pop("fold", None)
                list_train.append(item)
        if "testing" in json_data.keys() and "label" in json_data["testing"][0]:
            list_test = json_data["testing"]
        else:
            list_test = copy.deepcopy(list_valid)
        if args.splitval > 0:
            list_train = sorted(list_train, key=lambda x: x["image"])
            l = int((len(list_train) + len(list_valid)) * args.splitval)
            list_valid = list_train[-l:]
            list_train = list_train[:-l]

    if args.test_mode:
        list_train = list_train[:2]
        list_valid = list_valid[:2]
        list_test = list_test[:2]

    if hasattr(args, "rank") and args.rank == 0:
        # print("train files", len(list_train), [os.path.basename(_["image"]).split(".")[0] for _ in list_train])
        # print("val files", len(list_valid), [os.path.basename(_["image"]).split(".")[0] for _ in list_valid])
        # print("test files", len(list_test), [os.path.basename(_["image"]).split(".")[0] for _ in list_test])
        print(f'# of train: {len(list_train)}')
        print(f'# of val  : {len(list_valid)}')
        print(f'# of test : {len(list_test)}')

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(data_dir, list_train[_i]["image"])
        str_seg = os.path.join(data_dir, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg, 'image_name': str_img, 'label_name': str_seg})

    train_files = copy.deepcopy(files)

    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg, 'image_name': str_img, 'label_name': str_seg})
    val_files = copy.deepcopy(files)

    files = []
    for _i in range(len(list_test)):
        str_img = os.path.join(data_dir, list_test[_i]["image"])
        str_seg = os.path.join(data_dir, list_test[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg, 'image_name': str_img, 'label_name': str_seg})
    test_files = copy.deepcopy(files)
    return train_files, val_files, test_files
