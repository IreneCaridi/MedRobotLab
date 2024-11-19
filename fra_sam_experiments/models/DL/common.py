import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from .models_blocks import CNeXtBlock, CNeXtStem, CNeXtDownSample, ResBlock, ConvNormAct, ResBlockDP, TransformerBlock, \
                           SPPF, C3, UnetBlock, UnetDown, UnetUpBlock
from .utility_blocks import SelfAttentionModule, PatchMerging, PatchExpanding, LayerNorm, SelfAttentionModuleFC, SelfAttentionModuleLin


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvNormAct(3, 64, 6, 2, 2)
        self.a = ConvNormAct(64, 128, 6, 2, 2)
        self.b = ConvNormAct(128, 256, 6, 2, 2)
        self.start_h = (80 - 64) // 2  # 8
        self.start_w = (80 - 64) // 2  # 8

    def forward(self,x):
        x = self.b(self.a(self.stem(x)))

        return x[:, :, self.start_h:self.start_h + 64, self.start_w:self.start_w + 64]


class ConvNeXt(nn.Module):
    # ConvNeXt-T: C = (96; 192; 384; 768), B = (3; 3; 9; 3)
    # ConvNeXt-S: C = (96; 192; 384; 768), B = (3; 3; 27; 3)
    # ConvNeXt-B: C = (128; 256; 512; 1024), B = (3; 3; 27; 3)
    # ConvNeXt-L: C = (192; 384; 768; 1536), B = (3; 3; 27; 3)
    # ConvNeXt-XL: C = (256; 512; 1024; 2048), B = (3; 3; 27; 3)
    def __init__(self, num_classes, model_type='T', drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        if model_type == 'T':
            self.B = [3, 3, 9, 3]
            self.C = [96, 192, 384, 768]
        else:
            self.B = [3, 3, 27, 3]

        if model_type == 'S':
            self.C = [96, 192, 384, 768]
        elif model_type == 'B':
            self.C = [128, 385, 768, 1024]
        elif model_type == 'L':
            self.C = [192, 384, 768, 1536]
        elif model_type == 'XL':
            self.C = [256, 512, 1024, 2048]

        self.stem = CNeXtStem(1, self.C[0], k=2, s=2)

        self.S = nn.ModuleList([nn.Sequential(*(CNeXtBlock(self.C[i], drop_path=drop_path, layer_scale_init_value=layer_scale_init_value)
                                                for _ in range(self.B[i]))) for i in range(4)])
        self.DownSample = nn.ModuleList([CNeXtDownSample(self.C[i], self.C[i+1], 2, 2, 0) for i in range(3)])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1]//4),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//4, self.C[-1]//16),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//16, self.C[-1]//64),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//64, num_classes)
                                        )

    def forward(self, x):

        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)


class ConvNeXtSAM(nn.Module):
    # ConvNeXt-T: C = (96; 192; 384; 768), B = (3; 3; 9; 3)
    # ConvNeXt-S: C = (96; 192; 384; 768), B = (3; 3; 27; 3)
    # ConvNeXt-B: C = (128; 256; 512; 1024), B = (3; 3; 27; 3)
    # ConvNeXt-L: C = (192; 384; 768; 1536), B = (3; 3; 27; 3)
    # ConvNeXt-XL: C = (256; 512; 1024; 2048), B = (3; 3; 27; 3)
    def __init__(self, num_classes, in_c, model_type='T', drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        if model_type == 'T':
            self.B = [3, 3, 9, 3]
            self.C = [96, 192, 384, 768]
        else:
            self.B = [3, 3, 27, 3]

        if model_type == 'S':
            self.C = [96, 192, 384, 768]
        elif model_type == 'B':
            self.C = [128, 256, 512, 1024]
        elif model_type == 'L':
            self.C = [192, 384, 768, 1536]
        elif model_type == 'XL':
            self.C = [256, 512, 1024, 2048]

        self.stem = CNeXtStem(in_c, self.C[0], k=2, s=2)

        self.SAM = nn.ModuleList([SelfAttentionModule(self.C[i]) for i in range(4)])

        self.S = nn.ModuleList([nn.Sequential(*(CNeXtBlock(self.C[i], drop_path=drop_path, layer_scale_init_value=layer_scale_init_value)
                                                for _ in range(self.B[i]))) for i in range(4)])
        self.DownSample = nn.ModuleList([CNeXtDownSample(self.C[i], self.C[i+1], 2, 2, 0) for i in range(3)])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1]//4),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//4, self.C[-1]//16),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//16, self.C[-1]//64),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//64, num_classes)
                                        )

    def forward(self, x):

        x = self.stem(x)
        x = self.SAM[0](x)

        x = self.S[0](x)
        x = self.DownSample[0](x)
        x = self.SAM[1](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)
        x = self.SAM[2](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)
        x = self.SAM[3](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)


class ResNet1(nn.Module):
    def __init__(self, num_classes, in_dim=1):
        super().__init__()

        self.C = [128, 386, 768, 1024]
        self.B = [3, 4, 6, 3]

        self.stem = ConvNormAct(in_dim, self.C[0], 9, 2, 4)

        # ho sbagliato dovevo usare ConvNormAct...
        self.DownSample = nn.ModuleList([CNeXtDownSample(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([nn.Sequential(*(ResBlock(self.C[i]) for _ in range(self.B[i]))) for i in range(4)])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1] // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 4, self.C[-1] // 16),
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 16, num_classes)
                                        )

    def forward(self, x):

        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)


class ResNet2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # old with errors
        # self.C = [128, 386, 768, 1024]
        self.C = [128, 256, 512, 1024]
        self.B = [3, 4, 6, 3]

        self.stem = ConvNormAct(1, self.C[0], 9, 2, 4)

        self.dropout = nn.Dropout(0.1)

        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([nn.Sequential(*(ResBlockDP(self.C[i]) for _ in range(self.B[i]))) for i in range(4)])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1] // 4),
                                        self.dropout,
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 4, self.C[-1] // 16),
                                        self.dropout,
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 16, num_classes)
                                        )

    def forward(self, x):

        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)


class ResNetTransform(nn.Module):
    def __init__(self, num_classes, T_dim=128):
        super().__init__()

        self.C = [128, 256, 512, 1024]
        self.B = [3, 4, 6, 3]

        self.stem = ConvNormAct(1, self.C[0], 9, 2, 4)

        self.dropout = nn.Dropout(0.1)

        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([nn.Sequential(*(ResBlockDP(self.C[i]) for _ in range(self.B[i]))) for i in range(4)])

        self.T1 = TransformerBlock(input_dim=self.C[-1], d_model=T_dim, n_heads=1, dropout=0.)

        self.head = nn.Linear(T_dim, num_classes)

    def forward(self, x):
        BS = x.shape[0]
        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)
        # shape = (BS, features)

        x = self.T1(x)
        return self.head(x.view(BS, -1))


class ResNetTransform2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.C = [128, 256, 512, 1024]
        self.B = [3, 4, 6, 3]
        self.BT = [2, 4, 2]

        self.stem = ConvNormAct(1, self.C[0], 9, 2, 4)

        self.dropout = nn.Dropout(0.1)

        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([nn.Sequential(*(ResBlockDP(self.C[i]) for _ in range(self.B[i]))) for i in range(4)])

        self.adapt = nn.Linear(self.C[-1], self.C[0])

        self.Te = nn.ModuleList([nn.Sequential(*(TransformerBlock(input_dim=self.C[i], d_model=self.C[i], n_heads=8, dropout=0.) for _ in range(self.BT[i]))) for i in range(3)])
        self.Td = nn.ModuleList([nn.Sequential(*(TransformerBlock(input_dim=self.C[i], d_model=self.C[i], n_heads=8, dropout=0.) for _ in range(self.BT[i]))) for i in range(2, -1, -1)])

        self.patch_dw = nn.ModuleList([PatchMerging(self.C[i]) for i in range(2)])
        self.patch_up = nn.ModuleList([PatchExpanding(self.C[i]) for i in range(2, -1, -1)])

        self.bottleneck = nn.Sequential(nn.Linear(self.C[2], self.C[3]),
                                        nn.GELU(),
                                        nn.Linear(self.C[3], self.C[2]))

        self.head = nn.Linear(self.C[0], num_classes)

    def forward(self, x):
        BS = x.shape[0]
        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1).unsqueeze(0)
        # shape = (1, BS, features)

        x = self.adapt(x)  # 1, BS, 128

        x = self.Te[0](x)
        x = self.patch_dw[0](x)  # 1, BS/2, 256
        x = self.Te[1](x)
        x = self.patch_dw[1](x)   # 1, BS/4, 512
        x = self.Te[2](x)

        self.bottleneck(x)

        x = self.Td[0](x)
        x = self.patch_up[0](x)  # 1, BS/2, 256
        x = self.Td[1](x)
        x = self.patch_up[1](x)  # 1, BS, 128
        x = self.Td[2](x)

        return self.head(x.view(BS, -1))


class ResNetTransformerAtt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # old with errors
        # self.C = [128, 386, 768, 1024]
        self.C = [128, 256, 512, 1024]
        self.B = [3, 4, 6, 3]

        self.stem = ConvNormAct(1, self.C[0], 9, 2, 4)

        self.dropout = nn.Dropout(0.1)

        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([nn.Sequential(*(ResBlockDP(self.C[i]) for _ in range(self.B[i]))) for i in range(4)])

        self.T = nn.ModuleList([TransformerBlock(input_dim=self.C[i], d_model=self.C[i], n_heads=8, dropout=0.) for i in range(4)])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1] // 4),
                                        self.dropout,
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 4, self.C[-1] // 16),
                                        self.dropout,
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 16, num_classes)
                                        )

    def forward(self, x):

        x = self.stem(x)
        x = x.permute(0, 2, 1)
        x = self.T[0](x)
        x = x.permute(0, 2, 1)

        x = self.S[0](x)
        x = self.DownSample[0](x)
        x = x.permute(0, 2, 1)
        x = self.T[1](x)
        x = x.permute(0, 2, 1)

        x = self.S[1](x)
        x = self.DownSample[1](x)
        x = x.permute(0, 2, 1)
        x = self.T[2](x)
        x = x.permute(0, 2, 1)

        x = self.S[2](x)
        x = self.DownSample[2](x)
        x = x.permute(0, 2, 1)
        x = self.T[3](x)
        x = x.permute(0, 2, 1)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)


class TransformerEncDec(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.C = [128, 256, 512, 1024]
        self.BT = [2, 2, 6, 2]

        self.values_to_find = torch.tensor([0, 1, 2])

        self.adapt = nn.Sequential(ConvNormAct(1, 32, 9, 1, 4, act=nn.GELU(), norm=LayerNorm),
                                   ConvNormAct(32, 128, 9, 1, 4, act=nn.GELU(), norm=LayerNorm)
                                   )

        self.Te = nn.ModuleList([nn.Sequential(*(TransformerBlock(input_dim=self.C[i], d_model=self.C[i], n_heads=8*(2**i), dropout=0.) for _ in range(self.BT[i]))) for i in range(4)])
        self.Td = nn.ModuleList([nn.Sequential(*(TransformerBlock(input_dim=self.C[i], d_model=self.C[i], n_heads=8*(2**i), dropout=0.) for _ in range(self.BT[i]))) for i in range(3, -1, -1)])

        self.patch_dw = nn.ModuleList([PatchMerging(self.C[i]) for i in range(3)])
        self.patch_up = nn.ModuleList([PatchExpanding(self.C[i]) for i in range(3, -1, -1)])

        self.bottleneck = nn.Sequential(nn.Linear(self.C[3], self.C[1]),
                                        nn.GELU(),
                                        nn.Linear(self.C[1], self.C[3]))

        self.head = nn.Linear(self.C[0], num_classes)

    def forward(self, x):
        """
            param:
                --x  (x, y == which output to classify)
        """

        x, y = x

        x = self.adapt(x)  # creating embeddings (I'm not really working with patches now)

        x00 = x.permute(0, 2, 1)  # to BS, L, C (BS, 1000, 128)

        x = self.Te[0](x00)
        x0 = self.patch_dw[0](x)  # BS, 500, 256

        x = self.Te[1](x0)
        x1 = self.patch_dw[1](x)  # BS, 250, 512

        x = self.Te[2](x1)
        x2 = self.patch_dw[2](x)  # BS, 125, 1024

        x = self.Te[3](x2)

        x = self.bottleneck(x)      #PROVARE SOMMARE TIPO SHORTCUT UNET

        x = self.Td[0](x + x2)
        x = self.patch_up[0](x)  # BS, 250, 512

        x = self.Td[1](x + x1)
        x = self.patch_up[1](x)  # BS, 500, 256

        x = self.Td[2](x + x0)
        x = self.patch_up[2](x)  # BS, 1000, 128

        x = self.Td[3](x + x00)

        idx = torch.where(y == self.values_to_find.to(y.device))

        x = x[idx[0], idx[1]]   # (n_vect, 128)

        return self.head(x)


class ResUnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.C = [128, 256, 512, 1024]
        self.B = [3, 4, 6, 3]

        self.values_to_find = torch.tensor([0, 1, 2, 5])  # 5 == surroundings

        self.stem = nn.Sequential(ConvNormAct(1, 32, 9, 1, 4, act=nn.GELU(), norm=LayerNorm),
                                  ConvNormAct(32, 128, 9, 2, 4, act=nn.GELU(), norm=LayerNorm)
                                  )

        self.Se = nn.ModuleList([nn.Sequential(*(ResBlock(self.C[i], act=nn.GELU(), norm=LayerNorm) for _ in range(self.B[i]))) for i in range(4)])
        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2, act=nn.GELU(), norm=LayerNorm) for i in range(3)])
        self.Su = nn.ModuleList([nn.Sequential(*(ResBlock(self.C[i], act=nn.GELU(), norm=LayerNorm) for _ in range(3))) for i in range(3, -1, -1)])

        self.pw = nn.ModuleList([ConvNormAct(self.C[-1], self.C[-2], 1, 1, 0, act=nn.GELU(), norm=LayerNorm),
                                 ConvNormAct(self.C[-1], self.C[1], 1, 1, 0, act=nn.GELU(), norm=LayerNorm),
                                 ConvNormAct(self.C[2], self.C[0], 1, 1, 0, act=nn.GELU(), norm=LayerNorm),
                                 ConvNormAct(self.C[1], self.C[0], 1, 1, 0, act=nn.GELU(), norm=LayerNorm)
                                 ])
        self.UpSample = nn.Upsample(scale_factor=2, mode='nearest')

        self.sppf = SPPF(self.C[-1], self.C[-1], act=nn.GELU(), norm=LayerNorm)

        self.head = nn.Linear(self.C[0], num_classes)

    def forward(self, x):
        """
             param:
                 --x  (x, y == which output to classify)
         """

        x, y = x
        idx = torch.where(y == self.values_to_find.to(y.device))

        x = self.stem(x)

        x0 = self.Se[0](x)  # BS, 128, L/2
        x = self.DownSample[0](x0)

        x1 = self.Se[1](x)  # BS, 256, L/4
        x = self.DownSample[1](x1)

        x2 = self.Se[2](x)  # BS, 512, L/8
        x = self.DownSample[2](x2)

        x = self.Se[3](x)  # BS, 1024, L/16

        x = self.sppf(x)

        x = self.UpSample(self.pw[0](x))  # BS, 512, L/8


        x = torch.concat((x, x2), dim=1)  # BS, 1024, L/8
        x = self.UpSample(self.pw[1](self.Su[0](x)))  # BS, 256, L/4

        x = torch.concat((x, x1), dim=1)  # BS, 512, L/4
        x = self.UpSample(self.pw[2](self.Su[1](x)))  # BS, 128, L/2

        x = torch.concat((x, x0), dim=1)  # BS, 256, L/2
        x = self.UpSample(self.pw[3](self.Su[2](x)))  # BS, 128, L

        x = self.Su[3](x)  # BS, 128, L

        x = x.permute(0, 2, 1)  # to BS, L, C
        x = x[idx[0], idx[1]]  # (n_vect + 10*n_vect, 128)

        x = x.view(-1, 11, 128)  # (n_vect, surroundings, 128)

        return self.head(x)


class ResUnetAtt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.C = [128, 256, 512, 1024]
        self.B = [3, 4, 6, 3]

        self.values_to_find = torch.tensor([0, 1, 2, 5])  # 5 == surroundings

        self.stem = nn.Sequential(ConvNormAct(2, 32, 9, 1, 4, act=nn.GELU(), norm=LayerNorm),
                                  ConvNormAct(32, 128, 9, 2, 4, act=nn.GELU(), norm=LayerNorm)
                                  )

        self.Se = nn.ModuleList([nn.Sequential(*(ResBlock(self.C[i], act=nn.GELU(), norm=LayerNorm) for _ in range(self.B[i]))) for i in range(4)])
        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2, act=nn.GELU(), norm=LayerNorm) for i in range(3)])
        self.Su = nn.ModuleList([nn.Sequential(*(ResBlock(self.C[i], act=nn.GELU(), norm=LayerNorm) for _ in range(3))) for i in range(3, -1, -1)])

        self.pw = nn.ModuleList([ConvNormAct(self.C[-1], self.C[-2], 1, 1, 0, act=nn.GELU(), norm=LayerNorm),
                                 ConvNormAct(self.C[-1], self.C[1], 1, 1, 0, act=nn.GELU(), norm=LayerNorm),
                                 ConvNormAct(self.C[2], self.C[0], 1, 1, 0, act=nn.GELU(), norm=LayerNorm),
                                 ConvNormAct(self.C[1], self.C[0], 1, 1, 0, act=nn.GELU(), norm=LayerNorm)
                                 ])
        self.UpSample = nn.Upsample(scale_factor=2, mode='nearest')

        self.sppf = SPPF(self.C[-1], self.C[-1], act=nn.GELU(), norm=LayerNorm)

        self.att = SelfAttentionModuleFC(128, 11, return_map=True)

        self.fc = nn.Linear(11, 1)

        self.head = nn.Linear(self.C[0], num_classes)

    def forward(self, x):
        """
             param:
                 --x  (x, y == which output to classify)
         """

        x, y, _ = x

        idx = torch.where(y == self.values_to_find.to(y.device))

        look_at_map = torch.zeros(x.shape)
        look_at_map[idx[0], 0, idx[1]] = 1

        x = torch.concat((x, look_at_map.to(x.device)), dim=1)  # adding as info where the peaks are

        # elements = torch.eq(idx[-1], 3)  # idk but 5 becomes 3
        # n5 = torch.sum(elements)
        # print(idx[-1].shape)

        x = self.stem(x)

        x0 = self.Se[0](x)  # BS, 128, L/2
        x = self.DownSample[0](x0)

        x1 = self.Se[1](x)  # BS, 256, L/4
        x = self.DownSample[1](x1)

        x2 = self.Se[2](x)  # BS, 512, L/8
        x = self.DownSample[2](x2)

        x = self.Se[3](x)  # BS, 1024, L/16

        x = self.sppf(x)

        x = self.UpSample(self.pw[0](x))  # BS, 512, L/8


        x = torch.concat((x, x2), dim=1)  # BS, 1024, L/8
        x = self.UpSample(self.pw[1](self.Su[0](x)))  # BS, 256, L/4

        x = torch.concat((x, x1), dim=1)  # BS, 512, L/4
        x = self.UpSample(self.pw[2](self.Su[1](x)))  # BS, 128, L/2

        x = torch.concat((x, x0), dim=1)  # BS, 256, L/2
        x = self.UpSample(self.pw[3](self.Su[2](x)))  # BS, 128, L

        x = self.Su[3](x)  # BS, 128, L

        x = x.permute(0, 2, 1)  # to BS, L, C
        # print(x.shape)
        x = x[idx[0], idx[1]]  # (n_vect + 10*n_vect, 128)
        # print(idx[0].shape)
        # print(x.shape)

        x = x.view(-1, 11, 128)  # (n_vect, surroundings, 128)

        x, att = self.att(x.permute(0, 2, 1))

        x = self.fc(x)

        return self.head(x.squeeze())   # (n_vect, 128)


class ResUnetAtt2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.C = [128, 256, 512, 1024]
        self.B = [3, 4, 6, 3]

        self.values_to_find = torch.tensor([0, 1, 2, 5])  # 5 == surroundings

        self.stem = nn.Sequential(ConvNormAct(2, 32, 9, 1, 4, act=nn.GELU(), norm=LayerNorm),
                                  ConvNormAct(32, 128, 9, 2, 4, act=nn.GELU(), norm=LayerNorm)
                                  )

        self.Se = nn.ModuleList([nn.Sequential(*(ResBlock(self.C[i], act=nn.GELU(), norm=LayerNorm) for _ in range(self.B[i]))) for i in range(4)])
        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2, act=nn.GELU(), norm=LayerNorm) for i in range(3)])

        self.sppf = SPPF(self.C[-1], self.C[-1], act=nn.GELU(), norm=LayerNorm)

        self.att = SelfAttentionModuleFC(128, 11, return_map=True)

        self.fc = nn.Linear(11, 1)

        self.head = nn.Linear(self.C[-1], num_classes)

    def forward(self, x):
        """
             param:
                 --x  (x, y == which output to classify)
         """

        x, y, on = x

        idx = torch.where(y == self.values_to_find.to(y.device))

        look_at_map = torch.zeros(x.shape)
        look_at_map[idx[0], 0, idx[1]] = 1

        x = torch.concat((x, look_at_map.to(x.device)), dim=1)  # adding as info where the peaks are

        # elements = torch.eq(idx[-1], 3)  # idk but 5 becomes 3
        # n5 = torch.sum(elements)
        # print(idx[-1].shape)

        x = self.stem(x)

        x0 = self.Se[0](x)  # BS, 128, L/2
        x = self.DownSample[0](x0)

        x1 = self.Se[1](x)  # BS, 256, L/4
        x = self.DownSample[1](x1)

        x2 = self.Se[2](x)  # BS, 512, L/8
        x = self.DownSample[2](x2)

        x = self.Se[3](x)  # BS, 1024, L/16

        x = self.sppf(x)  # BS, 1024, L/16

        x = x.permute(0, 2, 1)

        new = 1
        # for i in range(x.shape[0]):
        #     for j in range(len(v_on)-1):
        #         if b_on[j] == i:
        #             if new:
        #                 new_t = x[i, v_on[j]:v_on[j+1], :].unsqueeze(0).mean(dim=1)
        #                 new = 0
        #             else:
        #                 new_t = torch.concat((new_t, x[i, v_on[j]:v_on[j+1], :].unsqueeze(0).mean(dim=1)), dim=0)

        for i, o in enumerate(on):
            for j in range(len(o)-1):
                if new:
                    new_t = x[i, o[j]:o[j+1], :].unsqueeze(0).mean(dim=1)
                    new = 0
                else:
                    new_t = torch.concat((new_t, x[i, o[j]:o[j+1], :].unsqueeze(0).mean(dim=1)), dim=0)

        return self.head(new_t.squeeze())  # (n_vect, 128)


class DarkNetCSP(nn.Module):
    def __init__(self, num_classes, in_dim):
        super().__init__()

        self.C = [128, 256, 512, 1024]
        self.B = [3, 6, 9, 3]

        self.stem = ConvNormAct(in_dim, self.C[0], 9, 2, 4)

        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([C3(self.C[i], self.C[i], self.B[i]) for i in range(4)])

        self.sppf = SPPF(self.C[-1], self.C[-1])

        self.classifier = nn.Sequential(
                                        # nn.Linear(self.C[-1], self.C[-1] // 4),
                                        # nn.SiLU(),
                                        # nn.Linear(self.C[-1] // 4, self.C[-1] // 16),
                                        # nn.SiLU(),
                                        nn.Linear(self.C[-1], num_classes)
                                        )

    def forward(self, x):

        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)

        x = self.sppf(x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)


class LearnableInitBiLSTM(nn.Module):
    def __init__(self, num_classes, in_dim):
        super(LearnableInitBiLSTM, self).__init__()

        c2 = 128
        num_layers = 8

        self.h0 = nn.Parameter(torch.zeros(num_layers * 2, 1, c2))
        self.c0 = nn.Parameter(torch.zeros(num_layers * 2, 1, c2))

        self.lstm = nn.LSTM(in_dim, c2, num_layers, batch_first=True, bidirectional=True)

        self.pw = nn.Conv1d(c2*2, c2, 1)
        self.act = nn.ReLU()

        # Output layer
        self.fc = nn.Linear(c2, num_classes)  # Multiply by 2 for bidirectional

    #     self.init_parameters()
    #
    # def init_parameters(self):
    #     # Initialize learnable parameters using Glorot initialization
    #     for param in self.parameters():
    #         if len(param.shape) >= 2:
    #             nn.init.xavier_uniform_(param.data)
    #         else:
    #             nn.init.zeros_(param.data)

    def forward(self, x):
        # Use learnable initial hidden states
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
        c0 = self.c0.expand(-1, x.size(0), -1).contiguous()

        x = x.permute(0, 2, 1)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        out = self.act(self.pw(out.permute(0, 2, 1)).mean(dim=2).view(x.shape[0], -1))

        # Fully connected layer
        out = self.fc(out)

        return out



class DarkNetCSPBoth(nn.Module):
    def __init__(self, in_dim, bi=False, tri=False, both=False):
        super().__init__()

        self.C = [128, 256, 512, 1024]
        self.B = [3, 6, 9, 3]

        self.stem = ConvNormAct(in_dim, self.C[0], 9, 2, 4)

        self.DownSample = nn.ModuleList([ConvNormAct(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([C3(self.C[i], self.C[i], self.B[i]) for i in range(4)])

        self.sppf = SPPF(self.C[-1], self.C[-1])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1] // 4),
                                         nn.SiLU(),
                                         nn.Linear(self.C[-1] // 4, self.C[-1] // 16),
                                         nn.SiLU(),
                                         nn.Linear(self.C[-1] // 16, 2)
                                         )

        self.head2 = nn.Sequential(nn.Linear(self.C[-1], self.C[-1] // 4),
                                         nn.SiLU(),
                                         nn.Linear(self.C[-1] // 4, self.C[-1] // 16),
                                         nn.SiLU(),
                                         nn.Linear(self.C[-1] // 16, 3)
                                         )

        self.bi = bi
        self.tri = tri
        self.both = both

        self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)

        x = self.sppf(x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        if self.bi:
            return self.classifier(x)
        if self.tri:
            return self.head2(x)
        if self.both:
            x1 = self.classifier(x)
            x_sf = self.softmax(x1)

            x2 = None
            idx = None
            for i, T in enumerate(x_sf):
                if torch.argmax(T).item() == 1:
                    if x2 is None:
                        x2 = self.head2(x[i]).unsqueeze(0)
                        idx = torch.tensor([i], device=x.device)
                    else:
                        x2 = torch.concat((x2, self.head2(x[i]).unsqueeze(0)), dim=0)
                        idx = torch.concat((idx, torch.tensor([i], device=x.device)), dim=0)

            return x1, (x2, idx)


class LearnableInitBiLSTM2(nn.Module):
    def __init__(self, num_classes, in_dim):
        super().__init__()

        c2 = 128
        num_layers = 4

        self.stem = ConvNormAct(1, 64, 5, 2, 2)
        self.a = ResBlock(64)
        self.b = ConvNormAct(64, 128, 5, 2, 2)
        self.c = ResBlock(128)

        self.h0 = nn.Parameter(torch.zeros(num_layers * 2, 1, c2))
        self.c0 = nn.Parameter(torch.zeros(num_layers * 2, 1, c2))

        self.lstm = nn.LSTM(in_dim, c2, num_layers, batch_first=True, bidirectional=True)

        self.pw = nn.Conv1d(c2*2, c2, 1)
        self.act = nn.ReLU()

        # Output layer
        self.fc = nn.Linear(c2*2, num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Use learnable initial hidden states
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
        c0 = self.c0.expand(-1, x.size(0), -1).contiguous()

        x2 = self.c(self.b(self.a(self.stem(x)))).mean(dim=2).view(x.shape[0], -1)

        x = x.permute(0, 2, 1)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        out = self.act(self.pw(out.permute(0, 2, 1)).mean(dim=2).view(x.shape[0], -1))

        # Fully connected layer
        out = self.fc(torch.concat((out, x2), dim=1))

        return out


class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.fc0 = nn.Linear(11, 32)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.act(self.fc0(x))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(x)
        return x


class MLPdo(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.do = nn.Dropout(0.1)
        self.fc0 = nn.Linear(11, 32)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.act(self.fc0(x))
        x = self.do(x)
        x = self.act(self.fc1(x))
        x = self.do(x)
        x = self.act(self.fc2(x))
        x = self.do(x)
        x = self.out(x)
        return x


class MLPatt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(11, 32),
                                 nn.LayerNorm(32, eps=1e-6),
                                 nn.ReLU(),
                                 SelfAttentionModuleLin(32),
                                 nn.Linear(32, 64),
                                 torch.nn.LayerNorm(64, eps=1e-6),
                                 nn.ReLU(),
                                 SelfAttentionModuleLin(64),
                                 nn.Linear(64, 32),
                                 torch.nn.LayerNorm(32, eps=1e-6),
                                 nn.ReLU(),
                                 SelfAttentionModuleLin(32),
                                 nn.Linear(32, num_classes)
                                 )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLPattDo(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(11, 32),
                                 nn.LayerNorm(32, eps=1e-6),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 SelfAttentionModuleLin(32),
                                 nn.Linear(32, 64),
                                 torch.nn.LayerNorm(64, eps=1e-6),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 SelfAttentionModuleLin(64),
                                 nn.Linear(64, 32),
                                 torch.nn.LayerNorm(32, eps=1e-6),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 SelfAttentionModuleLin(32),
                                 nn.Linear(32, num_classes)
                                 )

    def forward(self, x):
        x = self.mlp(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
#                   NUOVIIIIIIIIIII
# ----------------------------------------------------------------------------------------------------------------------


# adapted from "mamba U-net"
class UnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ch = 3  # RGB
        self.ch_dims = [32, 64, 128, 256, 512]
        self.dropout = [0.05, 0.1, 0.2, 0.3, 0.5]
        assert (len(self.ch_dims) == 5)
        self.stem = UnetBlock(
            self.in_ch, self.ch_dims[0], self.dropout[0])
        self.down1 = UnetDown(
            self.ch_dims[0], self.ch_dims[1], self.dropout[1])
        self.down2 = UnetDown(
            self.ch_dims[1], self.ch_dims[2], self.dropout[2])
        self.down3 = UnetDown(
            self.ch_dims[2], self.ch_dims[3], self.dropout[3])
        self.down4 = UnetDown(
            self.ch_dims[3], self.ch_dims[4], self.dropout[4])

    def forward(self, x):     # ->  512x512x3
        x0 = self.stem(x)     # ->  512x512x32
        x1 = self.down1(x0)   # ->  256x256x64
        x2 = self.down2(x1)   # ->  128x128x128
        x3 = self.down3(x2)   # ->  64x64x256
        x4 = self.down4(x3)   # ->  32x32x512
        return x0, x1, x2, x3, x4


# adapted from "mamba U-net"
class UnetDecoder(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.ch_dims = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        assert (len(self.ch_dims) == 5)

        self.up1 = UnetUpBlock(
            self.ch_dims[4], self.ch_dims[3], dropout_p=0.0)
        self.up2 = UnetUpBlock(
            self.ch_dims[3], self.ch_dims[2], dropout_p=0.0)
        self.up3 = UnetUpBlock(
            self.ch_dims[2], self.ch_dims[1], dropout_p=0.0)
        self.up4 = UnetUpBlock(
            self.ch_dims[1], self.ch_dims[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ch_dims[0], self.n_classes, 1, 1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features

        x = self.up1(x4, x3)    # -> 64x64x256
        x = self.up2(x, x2)     # -> 128x128x128
        x = self.up3(x, x1)     # -> 256x256x64
        x = self.up4(x, x0)     # -> 512x512x32
        out = self.out_conv(x)  # -> 512x512xN
        return out


class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.encoder = UnetEncoder()
        self.decoder = UnetDecoder(n_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)  # raw logit output
        return out


# includes a smal fpn to allow matching with sam embedding dims
class UNetEncoderTrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UnetEncoder()
        self.up1 = UnetUpBlock(512, 256, dropout_p=0.0)

    def forward(self, x):
        x = self.encoder(x)

        return self.up1(x[-1], x[-2])  # 64x64x256
