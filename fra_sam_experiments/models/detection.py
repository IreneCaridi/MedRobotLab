
import torch.nn as nn
from torch.cuda.amp import autocast

from mmdet.models.dense_heads import YOLOXHead, CenterNetUpdateHead
from mmdet.models.necks import FPN
from mmengine.config import ConfigDict
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from mmengine.structures import InstanceData

from .Rep_ViT import RepViT


class RpnYolox(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # upgrade to handle archs

        self.cfg = dict(rpn=dict(assigner=dict(type=MaxIoUAssigner, pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.2,
                                          match_low_quality=True, ignore_iof_thr=-1)),
                   nms=ConfigDict(nms_pre=2000, min_bbox_size=0, score_thr=0.35, nms=dict(type='nms', iou_threshold=0.7),
                                  max_per_img=1000),
                   metas = dict(img_shape=(input_shape, input_shape))
        )
        self.fpn = FPN(in_channels=[96, 192, 384], out_channels=96, num_outs=5)
        # strides [8, 16, 32, 64, 128] used by edgeSAM
        self.head = YOLOXHead(num_classes=num_classes, in_channels=96, strides=[32, 64, 128], train_cfg=self.cfg['rpn'],
                              loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1.0),
                              loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1.0),
                              loss_bbox=dict(type='CIoULoss', eps=1e-16, reduction='mean', loss_weight=1.0))

    def forward(self, x):
        x_, _ ,_ = x

        B, C, H, W = x_.shape

        x = self.fpn([*x])
        x_raw = self.head(x)

        batched_img_metas = [self.cfg['metas']] * B

        x = self.head.predict_by_feat(*x_raw, batch_img_metas=batched_img_metas, cfg=self.cfg['nms'], with_nms=True)

        return x_raw, x

    def get_loss(self, x_raw, y: [InstanceData]):
        batched_img_metas = [self.cfg['metas']] * len(y)

        loss_dict = self.head.loss_by_feat(*x_raw, batch_img_metas=batched_img_metas, batch_gt_instances=y)

        return loss_dict


class RpnCenterNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # upgrade to handle archs

        self.cfg = dict(rpn=dict(assigner=dict(type=MaxIoUAssigner, pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.2,
                                          match_low_quality=True, ignore_iof_thr=-1)),
                   nms=ConfigDict(nms_pre=2000, min_bbox_size=0, score_thr=0.35, nms=dict(type='nms', iou_threshold=0.7),
                                  max_per_img=1000),
                   metas = dict(img_shape=(input_shape, input_shape))
        )
        self.fpn = FPN(in_channels=[96, 192, 384], out_channels=96, num_outs=5)
        # strides [8, 16, 32, 64, 128] used by edgeSAM
        self.head = CenterNetUpdateHead(num_classes=num_classes, in_channels=96, stacked_convs=4, feat_channels=96,
                                        train_cfg=self.cfg['rpn'], strides=[8, 16, 32, 64, 128],
                                        loss_cls = dict(type='GaussianFocalLoss', reduction='mean', pos_weight=0.25, neg_weight=0.75, loss_weight=1.0),
                                        loss_bbox=dict(type='CIoULoss', eps=1e-16, reduction='mean', loss_weight=1.0))

    def forward(self, x):
        x_, _ ,_ = x

        B, C, H, W = x_.shape

        x = self.fpn([*x])
        x_raw = self.head(x)

        batched_img_metas = [self.cfg['metas']] * B

        x = self.head.predict_by_feat(*x_raw, batch_img_metas=batched_img_metas, cfg=self.cfg['nms'], with_nms=True)

        return x_raw, x

    def get_loss(self, x_raw, y: [InstanceData]):
        batched_img_metas = [self.cfg['metas']] * len(y)

        loss_dict = self.head.loss_by_feat(*x_raw, batch_img_metas=batched_img_metas, batch_gt_instances=y)

        return loss_dict


class RepViTDetYolox(nn.Module):
    def __init__(self, arch='m1', n_classes=1, img_size=1024, fuse=True, freeze=False):
        super().__init__()

        self.name = 'RepViTDet'
        # upgrade to handle archs

        self.backbone = RepViT(arch, img_size, fuse, freeze, True,
                               out_indices=['stage1', 'stage2', 'stage3', 'final'])
        self.rpn = RpnYolox(input_shape=img_size, num_classes=n_classes)

    def forward(self, x):

        x_enc = self.backbone(x)

        x1, x2, x3, xf = x_enc

        x_raw, x_pred = self.rpn([x1, x2, x3])

        return xf, x_raw, x_pred

    def get_loss(self, x_raw, y):
        return self.rpn.get_loss(x_raw, y)


class RepViTDetCenterNet(nn.Module):
    def __init__(self, arch='m1', n_classes=1, img_size=1024, fuse=True, freeze=False):
        super().__init__()

        self.name = 'RepViTDet'
        # upgrade to handle archs

        self.backbone = RepViT(arch, img_size, fuse, freeze, True,
                               out_indices=['stage1', 'stage2', 'stage3', 'final'])
        self.rpn = RpnCenterNet(input_shape=img_size, num_classes=n_classes)

    def forward(self, x):
        x_enc = self.backbone(x)

        x1, x2, x3, xf = x_enc

        x_raw, x_pred = self.rpn([x1, x2, x3])

        return xf, x_raw, x_pred

    def get_loss(self, x_raw, y):
        return self.rpn.get_loss(x_raw, y)

