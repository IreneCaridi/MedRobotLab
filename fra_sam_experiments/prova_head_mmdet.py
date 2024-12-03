import torch
import torch.nn as nn
from torch import dtype
import numpy as np

from models.common import UNet
from models.Rep_ViT import RepViT, RepViTBlock, RepViTUnet

from mmdet.models.dense_heads import YOLOXHead
from mmdet.models.necks import FPN
from mmdet.utils import (ConfigType, OptConfigType)
from mmengine.config import ConfigDict
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from mmengine.structures import InstanceData


train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type=MaxIoUAssigner,
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            # sampler=dict(
            #     type=RandomSampler,
            #     num=256,
            #     pos_fraction=0.5,
            #     neg_pos_ub=-1,
            #     add_gt_as_proposals=False),
        )
)


fpn = FPN(in_channels=[96, 192, 384], out_channels=96, num_outs=5)
head = YOLOXHead(num_classes=1, in_channels=96, strides=[8, 16, 32, 64, 128], train_cfg=train_cfg['rpn'])

t1 = torch.rand((1, 96, 128, 128), dtype=torch.float32)
t2 = torch.rand((1, 192, 64, 64), dtype=torch.float32)
t3 = torch.rand((1, 384, 32, 32), dtype=torch.float32)

o = fpn([t1, t2, t3])

p = head(o)

cfg = ConfigDict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1000)
batch_size = 1
batched_img_metas = [dict(img_shape=(1024, 1024))] * batch_size

out = head.predict_by_feat(*p, batch_img_metas=batched_img_metas, cfg=cfg, with_nms=False)

gt_bboxes = [[30, 30, 50, 50], [45, 30, 30, 45]]
gt_labels = [[1], [0]]
data_ann_img1 = dict(bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                 labels=np.array(gt_labels, dtype=np.long))

i = InstanceData(metainfo=batched_img_metas[0])

i.bboxes = torch.tensor(data_ann_img1['bboxes'])
i.labels = torch.tensor(data_ann_img1['labels'])

loss = head.loss_by_feat(*p, batch_img_metas=batched_img_metas, batch_gt_instances=[i])

# print(len(p))
# print([len(x) for x in list(p)])
# print(f"boxes: {out[0]['bboxes'].detach().numpy()}\n",
#       f"labels: {out[0]['labels'].detach().numpy()}\n",
#       f"scores: {out[0]['scores'].detach().numpy()}")
# print(loss)