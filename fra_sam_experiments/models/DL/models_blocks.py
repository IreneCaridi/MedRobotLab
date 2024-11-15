import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from .utility_blocks import LayerNorm, DropPath, SelfAttentionModule, LayerScale, PositionalEncoding


class CNeXtStem(nn.Module):
    def __init__(self, c1, c2, k=4, s=4, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p)
        self.norm = LayerNorm(c2)

    def forward(self, x):
        return self.norm(self.conv(x))


class CNeXtBlock(nn.Module):  #come paper
    def __init__(self, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        c_ = 4 * dim
        self.act = nn.GELU()
        self.add = shortcut
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        # layer scale
        self.gamma = LayerScale(layer_scale_init_value, dim) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma:
            x = self.gamma(x) * x

        if self.add:
            x = input + self.drop_path(x)

        return x


class CNeXtDownSample(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.norm = LayerNorm(c1)
        self.layer = nn.Conv2d(c1, c2, k, s, p)

    def forward(self, x):
        x = self.layer(self.norm(x))
        return x


class ConvNormAct(nn.Module):
    def __init__(self, c1, c2, k, s, p, act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.act = act
        self.norm = norm(c2)
        self.conv = nn.Conv2d(c1, c2, k, s, p)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c1, k=5, s=1, p=2, act=nn.ReLU(), norm=nn.BatchNorm2d):
        super().__init__()

        self.m = nn.Sequential(ConvNormAct(c1, c1//2, 1, 1, 0, act, norm),
                               ConvNormAct(c1//2, c1//2, k, s, p, act, norm),
                               ConvNormAct(c1//2, c1, 1, 1, 0, act, norm)
                               )

    def forward(self, x):
        return self.m(x) + x


class ResBlockDP(nn.Module):
    def __init__(self, c1, k=5, s=1, p=2, dp=0.1, act=nn.ReLU(), norm=nn.BatchNorm2d):
        super().__init__()

        self.dp = nn.Dropout2d(dp)
        self.m = nn.Sequential(self.dp,
                               ConvNormAct(c1, c1//2, 1, 1, 0, act, norm),
                               ConvNormAct(c1//2, c1//2, k, s, p, act, norm),
                               ConvNormAct(c1//2, c1, 1, 1, 0, act, norm)
                               )

    def forward(self, x):
        return self.m(x) + x


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.q = nn.Linear(input_dim, d_model, bias=False)
        self.k = nn.Linear(input_dim, d_model, bias=False)
        self.v = nn.Linear(input_dim, d_model, bias=False)

        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.feedforward = nn.Sequential(nn.Linear(d_model, d_model//2),
                                         nn.GELU(),
                                         nn.Linear(d_model//2, d_model)
                                         )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):

        x = self.norm1(x)

        q = self.positional_encoding(self.q(x))
        k = self.positional_encoding(self.k(x))
        v = self.positional_encoding(self.v(x))

        attn_output, _ = self.mha(q, k, v, need_weights=False)

        x = q + self.dropout(attn_output)  # da capire se ha senso questo shortcut

        x = self.norm2(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)

        return x


#  Stolen from YOLO repo
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, act=nn.ReLU(), norm=nn.BatchNorm2d):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvNormAct(c1, c_, 1, 1, 0, act, norm)
        self.cv2 = ConvNormAct(c_ * 4, c2, 1, 1, 0, act, norm)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):

        x = self.cv1(x)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n, shortcut=True, e=0.5, act=nn.SiLU(), norm=nn.BatchNorm2d):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, c_, 1, 1, 0, act=act, norm=norm)
        self.cv2 = ConvNormAct(c1, c_, 1, 1, 0, act=act, norm=norm)
        self.cv3 = ConvNormAct(2 * c_, c2, 1, 1, 0, act=act, norm=norm)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0, act=act, norm=norm) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, e=0.5, act=nn.SiLU(), norm=nn.BatchNorm2d):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, c_, 1, 1, 0, act=act, norm=norm)
        self.cv2 = ConvNormAct(c_, c2, 3, 1, 1, act=act, norm=norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ----------------------------------------------------------------------------------------------------------------------
#        NUOVIIIII
# ----------------------------------------------------------------------------------------------------------------------
#  params = {'in_chns': in_chns,
#                   'feature_chns': [16, 32, 64, 128, 256],
#                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                   'class_num': class_num,
#                   'bilinear': False,
#                   'acti_func': 'relu'}

# adapted from "mamba U-net"
class UnetBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, c1, c2, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            ConvNormAct(c1, c2, 3, 1, 1, nn.LeakyReLU(), nn.BatchNorm2d),
            nn.Dropout(dropout_p),
            ConvNormAct(c2, c2, 3, 1, 1, nn.LeakyReLU(), nn.BatchNorm2d),
        )

    def forward(self, x):
        return self.conv_conv(x)


# adapted from "mamba U-net"
class UnetDown(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, c1, c2, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), UnetBlock(c1, c2, dropout_p))

    def forward(self, x):
        return self.maxpool_conv(x)


# adapted from "mamba U-net"
class UnetUpBlock(nn.Module):
    """ 2x up-sampling"""
    def __init__(self, c1, c2, dropout_p):
        super().__init__()
        self.up = nn.ConvTranspose2d(c1, c2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(c2 * 2, c2, 1),
                                  UnetBlock(c2, c2, dropout_p))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



