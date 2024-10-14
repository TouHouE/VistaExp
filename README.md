## Model Declaration
- Declaration
   ```python
   import argparse
   from models.factory.vista_factory import vista_model_registry
   args = argparse.Namespace(
      sam_image_size=512, z_roi_iter=27
   )
   
   vista = vista_model_registry['vit_b'](
      checkpoint=None, image_size=args.sam_iamge_size, encoder_in_chans=args.z_roi_iter * 3, patch_embed_3d=True
   )
   ```
- loading pretrained weight 
   ```python
   import torch
   state_dict = torch.load(<...>)
   model.load_dict(state_dict)
   ```
   Please write by your self.
## Prepare model input
### Training
...
### Validation (No point prompt)
```python
import argparse
import torch
from utils import model_input as ModelInputer
args = argparse.Namespace(rank='cuda:0', nc=11)
# The dummy x, y
dx = torch.randn((2, 27, 512, 512)) # (B, z_roi_iter, sam_image_size, sam_image_size)
dy = torch.randint(0, args.nc, (2, 512, 512)) # (B, sam_image_size, sam_image_size)

data, *_ = ModelInputer.prepare_sam_val_input_cp_only(dx, dy, args)

```
### Testing (With point prompt)
```python
import argparse
import torch
from utils import model_input as ModelInputer
args = argparse.Namespace(
    rank='cuda:0', nc=11, label_prompt=True, sam_image_size=512, 
    point_prompt=True, points_val_pos=1, points_val_neg=0, max_points=2
)
dx = torch.randn((2, 27, args.sam_image_size, args.sam_image_size))
dy = torch.randint(0, args.nc, (2, args.sam_image_size, args.sam_image_size))

data, *_ = ModelInputer.prepare_sam_test_input(dx, dy, args)
```
- Description of `args`
  - rank: device full name
  - nc: how many categories(include background)
  - label_prompt(bool): using label_prompt or not
  - sam_image_size(int): image size
  - point_prompt(bool): using point prompt or not, if set to `False`, the `points_val_pos`, `points_val_neg`, `max_points` will be ignored.
  - points_val_pos(int, optional): how many positive point(represent by 1) use in point prompt
  - points_val_neg(int, optional): how many negative point(represent by 0) use in point prompt
  - max_points(int, optional): if use point prompt but the above 2 value not indicate, will set as max_points // 2
- Detail of data
  - The `data` datatype is `list[dict]`, the length of data is same as input batch
  - for `data[i]` we have:
    - image: just the input image and drop the batch axis
    - original_size: a tuple to store the original image size
    - labels: is a torch.Tensor with shape (nc - 1, 1) use to store class prompt, ex: let nc=3, labels=\[\[1], \[2]] 
    - point_coords: is a tensor with shape (nc - 1, pos+neg or pos+max/2 or neg+max/2 or max, 2), store a 2D coordinate
    - point_labels: is a tensor with shape (nc - 1, pos+neg or pos+max/2 or neg+max/2 or max, 1), 0 for neg, 1 for pos