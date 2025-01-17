import torch
import torch.nn as nn
from torch import dtype
import numpy as np

from models.common import UNet
from models.Rep_ViT import RepViT, RepViTBlock, RepViTEncDec

from mmdet.models.dense_heads import YOLOXHead
from mmdet.models.necks import FPN
from mmdet.utils import (ConfigType, OptConfigType)
from mmengine.config import ConfigDict
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from mmengine.structures import InstanceData

from models.Rep_ViT import RepViTDet


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
head = YOLOXHead(num_classes=2, in_channels=96, strides=[8, 16, 32, 64, 128], train_cfg=train_cfg['rpn'])

t1 = torch.rand((2, 96, 128, 128), dtype=torch.float32)
t2 = torch.rand((2, 192, 64, 64), dtype=torch.float32)
t3 = torch.rand((2, 384, 32, 32), dtype=torch.float32)

o = fpn([t1, t2, t3])

p = head(o)

cfg = ConfigDict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.3,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1000)
batch_size = 2
batched_img_metas = [dict(img_shape=(1024, 1024))] * batch_size

out = head.predict_by_feat(*p, batch_img_metas=batched_img_metas, cfg=cfg, with_nms=False)

gt_bboxes = [[30, 30, 50, 50], [45, 30, 30, 45]]  # xyxy
gt_labels = [[1], [0]]
data_ann_img1 = dict(bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                 labels=np.array(gt_labels, dtype=np.long))

i = InstanceData(metainfo=dict(img_shape=(1024, 1024)))

i.bboxes = torch.tensor(data_ann_img1['bboxes'])
i.labels = torch.tensor(data_ann_img1['labels'])

loss = head.loss_by_feat(*p, batch_img_metas=batched_img_metas, batch_gt_instances=[i] * 2)

# print(len(p))
# print([len(x) for x in list(p)])
# print(f"boxes: {out[0]['bboxes'].detach().numpy()}\n",
#       f"labels: {out[0]['labels'].detach().numpy()}\n",
#       f"scores: {out[0]['scores'].detach().numpy()}")
# print(loss)
from PIL import Image

m = RepViTDet()
path_to_weights = r'C:\Users\franc\Documents\EdgeSAM\weights\edge_sam_3x.pth'
m.backbone.load_from(path_to_weights)

img = np.array(Image.open(r'C:\Users\franc\OneDrive - Politecnico di Milano\dataset_mmi\images\train\image_1.png').convert('RGB'))
x = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)


# x = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)

x_enc, x_raw, x_pred = m(x)

loss = m.get_loss(x_raw,  [i])

print(loss)

print(len(x_raw) )
#
# bbox = x_pred[0]['bboxes'].detach().numpy().tolist()
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# f, ax = plt.subplots(1,1)
#
# ax.imshow(img)
#
# for bbox in bbox:
#     x_min, y_min, x_max, y_max = bbox
#     width = x_max - x_min  # Calculate width
#     height = y_max - y_min  # Calculate height
#
#     # Add a rectangle for each bounding box
#     rect = patches.Rectangle(
#         (x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none'
#     )
#     ax.add_patch(rect)
#
#
# plt.show()
