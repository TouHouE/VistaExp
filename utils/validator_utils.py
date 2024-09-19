import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import torch
from monai.metrics import compute_dice, compute_iou, get_confusion_matrix
from utils.io import save_json

def compute_all_metrics(y_pred, y_gt, nc):
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

    onehot_gt = torch.stack([(y_gt == category).long() for category in range(1, nc)], dim=1)
    batch_dice = compute_dice(y_pred, onehot_gt)    # B x Nc
    batch_miou = compute_iou(y_pred, onehot_gt)  # B x Nc
    batch_cm = get_confusion_matrix(y_pred, onehot_gt)   # Bx Nc x 4
    return batch_dice, batch_miou, batch_cm


def make_summary(pred, gt, nc, threshold=.5, **kwargs):
    summary = dict(metrics_by_cases=list())
    dice, iou, cm  = compute_all_metrics(pred, gt, nc)
    image_name = kwargs.get('image_name', 'NaN')
    label_name = kwargs.get('label_name', 'NaN')
    predict_id = kwargs.get('predict_id', 0)
    output_dir = kwargs.get('output_dir', './')

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
    mean_dict = {str(i): {key: list() for key in ['Dice', f'IoU@{threshold}', 'TP', 'FP', 'FN', 'TN']} for i in
                 range(1, nc)}
    for case in summary['metrics_by_cases']:
        for digit, m_pack in case['metrics'].items():
            for indicator_name, indicator_value in m_pack.items():
                mean_dict[digit][indicator_name].append(indicator_value)
    for digit in range(1, nc):
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
        output_dir, 'summary.json'
    ), summary)