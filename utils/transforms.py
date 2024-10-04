from copy import deepcopy
import logging
LOG_FORMAT = '[%(asctime)s %(levelname)s %(filename)s:%(lineno)d (%(funcName)s)] %(message)s'
logging.basicConfig(format=LOG_FORMAT)
import torch
import numpy as np
from monai import transforms as MT
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor, DtypeLike, KeysCollection
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype
from monai.data.meta_obj import get_track_meta


class PaddingBackgroundMask(MT.Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dtype: DtypeLike = np.float32):
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img_: NdarrayTensor = convert_to_tensor(img, track_meta=get_track_meta(), dtype=self.dtype)

        img_[img != img.min()] = 1
        img_[img == img.min()] = 0
        return img_

class PaddingBackgroundMaskd(MT.MapTransform):
    backend = PaddingBackgroundMask.backend

    def __init__(
            self,
            keys: KeysCollection, dtype: DtypeLike = np.float32, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.masker = PaddingBackgroundMask(
            dtype=dtype
        )
    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.masker(d[key])
        return d


class AdditionalInfoExpanderd(MT.MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, dtype: DtypeLike = np.float32, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.masker = PaddingBackgroundMask(dtype=dtype)

    def trainable_range(self, data: dict) -> range:
        label = convert_to_tensor(data['label'])
        n_slice = label.shape[-1]
        has_label = torch.sum(label.view(-1, label.shape[-1]), dim=0)
        indicate = torch.argwhere(has_label != 0)
        if len(indicate) >= 200:
            return range(min(indicate), max(indicate) + 1 - 20)
        return range(min(indicate), max(indicate) + 1)

    def __call__(self, data: dict) -> dict:

        d = dict(data)
        d['range'] = self.trainable_range(d)
        # d['padding_mask'] = self.masker(deepcopy(d['image']))
        d['label'] *= self.masker(deepcopy(d['image']))
        logging.warning('Adding new keys [range]')
        return d


class Debugd(MT.MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        # self.masker = PaddingBackgroundMask(dtype=dtype)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key, value in d.items():
            logging.info(f'{key}: {value}')
        return d


class RetryLoadImaged(MT.LoadImaged):
    def __init__(self, keys, retry_time: int = 5, allow_missing_keys: bool = False, **kwargs):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.loader = MT.LoadImage(**kwargs)
        self.retry_time = retry_time

    def _call_loader(self, key_name, path):
        loaded_data: NDArrayOrTensor
        flag = False
        for time_out in range(self.retry_time):
            try:
                loaded_data = self.loader(path)
                flag = True
                break
            except Exception as e:
                logging.error(f'Retry[{time_out}/{self.retry_time}]Loading [{key_name}]: {path} failed')
                logging.error(f'{e.args[0]}')
        if flag:
            return loaded_data
        breakpoint()
        return loaded_data

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._call_loader(key, d[key])
        return d





if __name__ == '__main__':
    dimage = r"C:\Users\hsuwi\Downloads\cand_0_Calcium_Score_Axial_Axial_FC12_Cardiac_3.0_20010907093219_3.nii.gz"
    dimage = MT.ResizeWithPadOrCrop(spatial_size=(512, 512, -1), method='end', mode='minimum')(MT.Orientation(axcodes="RAS")(MT.EnsureChannelFirst()(MT.LoadImage()(dimage))))
    dlabel = torch.randint_like(dimage, 0, 11)
    dlabel[..., -dimage.shape[-1] // 3:] = 0
    ds = {
        'image': dimage,
        'label': dlabel
    }
    test = AdditionalInfoExpanderd(keys=['image', 'label'])
    tmp = test(ds)
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import matplotlib.pyplot as plt
