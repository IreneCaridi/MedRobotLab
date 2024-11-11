import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(nn.Module):
    def __init__(self, layer_scale_init_value, dim):
        super().__init__()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        return x


class SelfAttentionModule(nn.Module):
    def __init__(self, c_in, k=9, s=1, p=4):
        super().__init__()
        self.conv = nn.Conv1d(c_in, 1, 1, 1)
        self.spatial = nn.Sequential(nn.Conv1d(1, 32, k, s, p),
                                     nn.ReLU(),
                                     nn.Conv1d(32, 1, k, s, p)
                                     )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # (bs, C, L)
        x_ = self.conv(x)
        # (bs, 1, L)
        x_ = self.spatial(x_)
        # (bs, 1, L)
        scale = self.act(x_)

        return x * scale


#  computes attention over features along all timesteps
class SelfAttentionModuleFC(nn.Module):
    def __init__(self, c_in, d_lin, return_map=False):
        super().__init__()
        self.return_map = return_map
        self.conv = nn.Conv1d(c_in, 1, 1, 1)
        self.spatial = nn.Sequential(nn.Linear(d_lin, d_lin//4),
                                     nn.ReLU(),
                                     nn.Linear(d_lin//4, d_lin)
                                     )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # (bs, L, C)
        x_ = self.conv(x)
        # (bs, 1, C)
        x_ = self.spatial(x_)
        # (bs, 1, C)
        scale = self.act(x_)

        if not self.return_map:
            return x * scale
        else:
            return x * scale, scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        enc = self.encoding[:, :x.size(1)].detach().to(x.device)
        #print(x.shape, enc.shape)
        return x + enc


# from swin transformers (adapted)
class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        assert x.shape[1] % 2 == 0, f"x size ({x.shape[1]}) is not even."

        x0 = x[:, 0::2, :]  # BS H/2 C
        x1 = x[:, 1::2, :]  # BS H/2 C

        x = torch.cat([x0, x1], -1)  # BS H/2 2*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
        """ Patch Expanding Layer.

        Args:
            dim (int): Number of input channels.
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

        def __init__(self, dim, norm_layer=nn.LayerNorm):
            super().__init__()
            self.dim = dim
            self.reduction = nn.Linear(dim//2, dim//2, bias=False)
            self.norm = norm_layer(dim//2)

        def forward(self, x):
            """
            x: B, L, C
            """

            x = x.view(-1, x.shape[1]*2, x.shape[-1]//2)  # BS H*2 C/2

            x = self.norm(x)
            x = self.reduction(x)

            return x


class SelfAttentionModuleLin(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(c_in, c_in//2),
                                     nn.ReLU(),
                                     nn.Linear(c_in//2, c_in)
                                    )
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_ = self.spatial(x)
        scale = self.act(x_)

        return x * scale
