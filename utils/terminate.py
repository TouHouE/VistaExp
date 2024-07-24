import time


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


def show_model_info(model, args):
    if args.rank != 0:
        return
    sub_module = dict()
    for module_name, module_param in model.named_parameters(recurse=False):
        if not module_param.requires_grad:
            continue
        if sub_module.get(module_name, None) is None:
            module_param[module_param] = list()
        for name, param in module_param.named_parameters():
            module_param[module_name].append(param.numel())

    print(f'|{"Module Name":20}|{"Size(MB)":20}|')
    for mname, param_list in sub_module.items():
        size = f'{sum(param_list) * 1e-6:.4f}'
        print(f'|{mname:20}|{size:20}|')
    return


def show_training_info(epoch, idx, len_loader, avg_run_loss, start_time, args):
    if args.rank != 0:
        return None
    print(
        "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len_loader),
        "loss: {:.4f}".format(avg_run_loss.avg),
        "time {:.2f}s".format(time.time() - start_time),
    )