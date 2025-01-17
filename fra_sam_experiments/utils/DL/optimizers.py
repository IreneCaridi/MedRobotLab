import torch
import math


def get_optimizer(model, name, lr0, momentum=None, weight_decay=None):

    # not doing weight decay on batch norm
    params_no_wd = []
    params_with_wd = []
    for n, param in model.named_parameters():
        if 'bn' in n or 'norm' in n:  # Assuming BatchNorm layers' names contain 'bn'
            params_no_wd.append(param)
        else:
            params_with_wd.append(param)

    params = [{'params': params_with_wd, 'weight_decay': weight_decay}, {'params': params_no_wd, 'weight_decay': 0}]
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr0, momentum=momentum)
    elif name == "Adam":
        return torch.optim.Adam(params, lr=lr0, betas=(momentum, 0.999), eps=1e-08)
    elif name == "AdamW":
        return torch.optim.AdamW(params, lr=lr0, betas=(momentum, 0.999), eps=1e-08)
    else:
        raise TypeError("optimizer name not recognised")


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def linear(y1, y2, steps):
    # lamda linear decay function
    return lambda x: (y1 - x / steps) * (y1 - y2) + y2


def linear_warmup(warmup_epochs, y1=1e-4, y2=1.0):
    return lambda epoch: (y2 - y1) * (epoch / warmup_epochs) + y1


def scheduler(opt, name, lrf, epochs, warmup: int = None):
    if not warmup:
        warmup_epochs = epochs // 15
    else:
        warmup_epochs = warmup
    warmup_scheduler = linear_warmup(warmup_epochs)
    if name == "cos_lr":
        lf = one_cycle(1, lrf, epochs)  # cosine 1->hyp['lrf']
    elif name == "linear":
        lf = linear(1.0, lrf, epochs)
    elif not name:
        # no scheduler
        return None
    else:
        raise TypeError("scheduler name not recognised")

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_scheduler(epoch)
        else:
            return lf(epoch - warmup_epochs)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


