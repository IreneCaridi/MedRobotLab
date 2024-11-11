import os
import numpy as np
import random
import warnings
import torch
from pathlib import Path
import json


# fixes random states to same seed and silenced warnings
def random_state(seed=36):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def integer_division(a: int, b: int, floor=False):
    """

    args:
        -a: first term
        -b: second term
        -floor: if true returns division "per difetto", False returns "per eccesso"
    :return:
    """

    if floor:
        return a // b
    else:
        return a // b + 1


def increment_path(folder="runs", name="exp", exist_ok=False, sep=''):
    """

    Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    :param
        folder: addressed folder
        name: name of the new directory inside folder
        exist_ok: if True keeps the existing dir, else actually increments it
        sep: how to connect the increment with name (see above)
    """

    path = Path(folder) / name  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if not os.path.isdir(path):
        os.mkdir(path)

    print(f"saving folder is {path}")
    return path


def json_from_parser(parser_args, save_path, name="arguments.json"):
    args_dict = vars(parser_args)

    save_path = save_path / name
    with open(save_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=2)
