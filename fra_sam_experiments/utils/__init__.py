import os
import numpy as np
import random
import warnings
import torch


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


