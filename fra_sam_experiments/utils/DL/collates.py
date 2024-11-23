import torch


def keep_unchanged(batch):

    return batch


def imgs_masks_polys(batch):
    x, y, p = zip(*batch)

    x = torch.stack(x)
    y = torch.stack(y)
    return x, y, p

