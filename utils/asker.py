import os


def ask_logdir_root(args):
    response = input(f'Where to allocate logdir: {args.logdir}[default:./runs]')
    final_path = os.path.join(response, args.logdir)
    os.makedirs(final_path, exist_ok=True)
    return final_path


def ask_cache_root(args):
    if args.dataset_type == 'persis' or (args.dataset_type == 'normal' and args.poor_mode):
        response = input(f'Where to allocate cache path [default:./cache]')
        os.makedirs(response, exist_ok=True)
        return response
    return None
