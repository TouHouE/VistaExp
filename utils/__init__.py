import warnings, argparse
from typing import Optional
import logging
LOG_FORMAT = '[%(asctime)s %(levelname)s %(filename)s:%(lineno)d(%(funcName)s)] %(message)s'
logging.basicConfig(format=LOG_FORMAT)
import torch


def sorted_by(sort_array, by, return_both=False, reverse=False):
    sorted_array = sorted(zip(by, sort_array), reverse=reverse)
    xlist = list(x for _, x in sorted_array)

    if not return_both:
        return xlist
    ylist = list(y for y, _ in sorted_array)
    return xlist, ylist


def get_unique_labels(unique_labels, poor_categories: Optional[list[int]] = None) -> torch.LongTensor:
    if poor_categories is not None:
        unique_labels = unique_labels[(unique_labels.view(1, -1) == torch.as_tensor(poor_categories).view(-1, 1)).any(dim=0)]
        logging.info('Now only provides category hints for challenging training samples.')
        logging.debug(f"Poor categories: {poor_categories}")

    if hasattr(unique_labels, 'as_tensor'):
        return unique_labels.as_tensor().long()
    return unique_labels.long()


def assign_device(objected, device) -> torch.Tensor:
    if isinstance(device, torch.device):
        return objected.to(device)
    if isinstance(device, str):
        if device == 'cpu':
            return objected.cpu()
    return objected.cuda(device)


def get_args():
    warnings.filterwarnings("ignore", category=UserWarning, module="monai")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="nibabel")
    parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument('--project', type=str)
    parser.add_argument('--nc', default=11, type=int)
    parser.add_argument('--name')
    parser.add_argument('--dataset_type', default='cache', choices=['cache', 'persis', 'normal'], type=str,
                        help='belong to [cache, persis, normal], cache for CacheDataset persis for ')
    parser.add_argument('--no_spacing', default=False, action='store_true',
                        help='Do not using monai.transforms.Spacing during CT loading.')
    parser.add_argument('--pixdim', default=[1.5, 1.5, 1.5], nargs=3, type=float,
                        help='specified the Spacing transforms')
    parser.add_argument('--vae', action='store_true', default=False, help='using vae branch do reconstruct image loss.')
    parser.add_argument('--loss_func', choices=['dice_ce', 'dice_focal'], default='dice_ce')
    parser.add_argument('--batch_type', choices=['patch', 'aug', 'not'], default='not',
                        help='If in [patch, aug] --quasi_batch_size should been setting.')
    parser.add_argument('--quasi_batch_size', type=int, default=1)
    parser.add_argument('--num_aug', type=int, default=0)
    parser.add_argument('--poor_mode', action='store_true', default=False)
    parser.add_argument('--eval_bg', action='store_true', default=False,
                        help='setting this into monai.metrics.DiceMetrics\'s include_background argument')
    parser.add_argument('--label_map_path', type=str, default='./res/labels_id.json')
    parser.add_argument('--bad_image_maxlen', type=int, required=False, default=-1)
    parser.add_argument('--random_permute', action='store_true', default=False)
    parser.add_argument('--permute_prob', type=float, default=.5)
    parser.add_argument('--worst_mode', choices=['min', 'max'], default=None)
    parser.add_argument('--poor_classes', nargs='+', type=int, help='The digits of hard learning classes')
    parser.add_argument('--id', type=str, default=None, required=False, help='This argument used to log reuse wandb runs.')

    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default="vista2pt5d", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
    parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
    parser.add_argument("--max_epochs", default=1600, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", choices=['adam', 'adamw', 'sgd'], type=str,
                        help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")

    parser.add_argument("--a_min", default=-1024, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1024, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--fold", default=0, type=int, help="fold")
    parser.add_argument(
        "--splitval", default=0, type=float,
        help="if not zero, split the last portion to validation and validation to test"
    )
    parser.add_argument("--roi_z_iter", default=9, type=int, help="roi size in z direction")
    parser.add_argument("--roi_z_iter_dilation", default=0, type=int, help="dilation size in z direction")
    parser.add_argument("--lrschedule", default="No", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
    parser.add_argument("--num_patch", default=4, type=int, help="number of patches in each volume")
    parser.add_argument("--num_patch_val", default=30, type=int,
                        help="number of patches in each volume during validation")
    parser.add_argument("--num_prompt", default=8, type=int, help="number of prompts for each training instance")
    parser.add_argument("--clip", default=None, type=float, help="gradient clip")
    parser.add_argument("--seed", default=-1, type=int, help="seed")
    parser.add_argument("--sam_pretrain_ckpt", type=str, default=None, help="sam_pretrain_ckpt")
    parser.add_argument("--sam_base_model", type=str, default="vit_b", help="sam_pretrain_ckpt")
    parser.add_argument("--sam_image_size", type=int, default=1024, help="sam input res")
    parser.add_argument("--label_prompt", action="store_true", help="using class label prompt in training")
    parser.add_argument("--drop_label_prob", default=0.5, type=float, help="prob for dropping label prompt in training")
    parser.add_argument(
        "--label_prompt_warm_up_epoch",
        default=20,
        type=int,
        help="before this number of epoch, we will drop label prompt with low prob.",
    )
    parser.add_argument("--point_prompt", action="store_true", help="using point prompt in training")
    parser.add_argument("--drop_point_prob", default=0.5, type=float, help="prob for dropping point prompt in training")
    parser.add_argument(
        "--max_points",
        default=8,
        type=int,
        help="max number of point prompts in training for the first ponit prompt generation",
    )
    parser.add_argument("--points_val_pos", default=1, type=int, help="number of positive point prompts in evaluation")
    parser.add_argument("--points_val_neg", default=0, type=int, help="number of negative point prompts in evaluation")
    parser.add_argument("--num_iterative_step", default=5, type=int, help="number of iterative step in training")
    parser.add_argument("--reuse_img_embedding", action="store_true",
                        help="reuse image embedding in iterative training")
    parser.add_argument(
        "--no_more_points_for_cp_only", action="store_true",
        help="if no point prompt at the first prompt generation we will not add "
             "more additional pointa during iterative training.",
    )
    parser.add_argument(
        "--iterative_training_warm_up_epoch", default=100, type=int,
        help="before this number of epoch, we will not start iterative_training_.",
    )
    parser.add_argument("--data_aug", action="store_true", help="using data augmentation in training")
    parser.add_argument("--pop_pos_embed", action="store_true", help="remove pos embedding when load checkpoint")
    parser.add_argument("--pop_point_embed", action="store_true", help="remove point embedding when load checkpoint")
    parser.add_argument("--skip_bk", action="store_true", help="skip background (0) during training")
    parser.add_argument("--patch_embed_3d", action="store_true", help="using 3d patch embedding layer")
    parser.add_argument('--test_mode', action='store_true', default=False)
    return parser.parse_args()
