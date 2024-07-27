import time

import torch.nn


def show_prob(args):
    print(
        "rank:", args.rank,
        "label_prompt (train):", args.label_prompt,
        ", label_drop_prob:", args.drop_label_prob,
        "| point_prompt (train):", args.point_prompt,
        ", point_drop_prob:", args.drop_point_prob
    )
    return


def show_trained_info(epoch, train_loss, epoch_time, args):
    print(
        "Final training  {}/{}".format(epoch, args.max_epochs - 1),
        "loss: {:.4f}".format(train_loss),
        "time {:.2f}s".format(time.time() - epoch_time),
    )


def show_before_valid_info(args):
    if args.rank != 0:
        return
    print("Start validation")
    print("label_prompt (val):", args.label_prompt, "point_prompt (val):", args.point_prompt)


def show_valided_info(epoch, val_avg_acc, val_MA, best_epoch, val_acc_max, epoch_time, args):
    print(
        "Final validation  {}/{},".format(epoch, args.max_epochs - 1),
        f"Acc {val_avg_acc:.4f},",
        f"mv Acc {val_MA:.4f},",
        "Previous Best validation at epoch {} is {:.4f},".format(best_epoch, val_acc_max),
        "time {:.2f}s".format(time.time() - epoch_time),
    )


@torch.no_grad()
def show_model_info(model, args):
    if args.rank != 0:
        return
    sub_module = dict()
    # breakpoint()

    for module_name, module_param in model.named_parameters():
        if not module_param.requires_grad:
            continue
        key = module_name.split('.')[0]
        if sub_module.get(key, None) is None:
            sub_module[key] = list()
        sub_module[key].append(module_param.numel())
    print(f'=' * 43)
    print(f'|{"Module Name":^20}|{"Size(MB)":^20}|')
    print(f'|{"-" * 20}|{"-" * 20}|')
    tot_size = .0
    for mname, param_list in sub_module.items():
        size = f'{sum(param_list) * 1e-6:^.4f}'
        tot_size += sum(param_list)
        print(f'|{mname:^20}|{size:^20}|')
    print(f'|{"-" * 41}|')
    info = f'Total size: {tot_size * 1e-6:.4f}'
    print(f'|{info:41}|')
    print(f'=' * 43)
    return


def show_training_info(epoch, idx, len_loader, avg_run_loss, start_time, args):
    if args.rank != 0:
        return None
    print(
        "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len_loader),
        "loss: {:.4f}".format(avg_run_loss),
        "time {:.2f}s".format(time.time() - start_time),
    )


def show_some_hyper(args):
    if args.rank != 0:
        return
    info = dict()
    info['nc'] = args.nc
    info['dataset_type'] = args.dataset_type
    info['eval_bg'] = args.eval_bg
    info['roi_z_iter'] = args.roi_z_iter
    info['fold'] = args.fold
    info['num_patch'] = args.num_patch
    info['num_patch_val'] = args.num_patch_val
    string_length = min(max(len(key)for key in info.keys()), 30)
    print(f'Hyper Hint\n')
    for key, value in info.items():
        print(f'{key:{string_length}}: {value}')
    return

