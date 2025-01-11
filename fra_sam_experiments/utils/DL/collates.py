import torch


def keep_unchanged(batch):

    return batch


def edgesam_collate(batch):
    x = [i for i, _ in batch]
    y = [j for _, j in batch]
    return x, y

def imgs_masks_polys(batch):

    x, y, p = zip(*batch)

    x = torch.stack(x)
    y = torch.stack(y)
    return x, y, p


def from_grid_crop(batch):

    xx = []
    yy = []
    for b in batch:
        x, y, p = zip(*b)
        xx += x
        yy += y

    x = torch.stack(xx, dim=0)
    y = torch.stack(yy, dim=0)

    return x, y, p
