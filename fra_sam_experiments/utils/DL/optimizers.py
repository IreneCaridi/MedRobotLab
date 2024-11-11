import torch
import math


def get_optimizer(model, name, lr0, momentum=None, weight_decay=None):

    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay)
    elif name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr0, betas=(momentum, 0.999), weight_decay=weight_decay,
                                eps=1e-08)
    elif name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr0, betas=(momentum, 0.999), weight_decay=weight_decay,
                                 eps=1e-08)
    else:
        raise TypeError("optimizer name not recognised")


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def linear(y1, y2, steps):
    # lamda linear decay function
    return lambda x: (y1 - x / steps) * (y1 - y2) + y2


def scheduler(opt, name, lrf, epochs):
    if name == "cos_lr":
        lf = one_cycle(1, lrf, epochs)  # cosine 1->hyp['lrf']
    elif name == "linear":
        lf = linear(1.0, lrf, epochs)
    elif not name:
        # no scheduler
        return None
    else:
        raise TypeError("scheduler name not recognised")

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)


