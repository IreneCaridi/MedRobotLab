
import torch
from pathlib import Path
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.DL.loaders import EdgeSAMLoader
from utils.DL.collates import edgesam_collate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(str(Path(parent_dir) / 'EdgeSAM'))

from edge_sam import sam_model_registry, SamPredictor
from edge_sam.utils.transforms import ResizeLongestSide

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'

def compute_iou(x, y):
    """
    Compute the Intersection over Union (IoU) between two binary mask tensors.

    Args:
        x (torch.Tensor): A binary mask tensor of shape (N, H, W), values should be 0 or 1.
        y (torch.Tensor): A binary mask tensor of the same shape as mask1, values should be 0 or 1.

    Returns:
        torch.Tensor: The IoU score(s). If inputs are 2D (H, W), returns a single IoU value.
                     If inputs are 3D (N, H, W), returns a tensor of shape (N,) with IoU scores per mask.
    """

    assert len(x.shape) == 3 and len(y.shape) == 3

    x = x.flatten(start_dim=1).to(torch.bool)  # Shape becomes (N, H*W)
    y = y.flatten(start_dim=1).to(torch.bool)

    # Compute the intersection and union
    intersection = torch.sum(x & y, dim=1)  # Element-wise AND, then sum over H and W
    union = torch.sum(x | y, dim=1)  # Element-wise OR, then sum over H and W

    # Avoid division by zero
    union = torch.clamp(union, min=1e-6)

    # Compute IoU
    iou = intersection / union

    return iou  # Nx1

src = Path(r'C:\Users\franc\Documents\MedRobotLab\dataset\Cholect_dataset\images\test')

path_to_weights = r'C:\Users\franc\Documents\EdgeSAM\weights\edge_sam_3x.pth'

sam_checkpoint = r"C:\Users\franc\Documents\MedRobotLab\EdgeSAM\weights\edge_sam_3x.pth"
model_type = "edge_sam"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

lab_type = 'bbox'

dataset = EdgeSAMLoader(src, lab_type, device=device)

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=edgesam_collate)

IoU_dict = {i: [] for i in range(7)}

with torch.no_grad():
    for x, y in tqdm(loader, desc="Batches", bar_format=TQDM_BAR_FORMAT):
        batched_output = sam(x, num_multimask_outputs=1)


# for ms, n in zip(batched_output, ['seg8k_video12_015810', 't50_VID10_000660', 't50_VID74_001380']):
#     m_np = np.zeros((480, 854)).astype(np.uint8)
#     for m in ms['masks']:
#         m_np += m.squeeze().numpy().astype(np.uint8)
#     m_np = np.clip(m_np, a_min=0, a_max=1)
#     m_np = m_np * 255
#     image = Image.fromarray(m_np)
#     image.save(f'{n}_mask.png')


#     for i in range(len(batched_output)):
#         IoU = compute_iou(batched_output[i]['masks'].squeeze(1), y[i][0])
#         for iou, l in zip(IoU.numpy().tolist(), y[i][1]):
#             IoU_dict[l].append(iou)
#
# map_dict = {'grasper': 0,
#             'snare': 1,
#             'irrigator': 2,
#             'clipper': 3,
#             'scissors': 4,
#             'bipolar': 5,
#             'hook': 6}
#
# inverse_map = {map_dict[k]: k for k in map_dict.keys()}
#
# dst = Path(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data') / f'{lab_type}_02'
# os.makedirs(dst, exist_ok=True)
# for k in IoU_dict.keys():
#     np.save(dst / f'{inverse_map[k]}_iou.npy', IoU_dict[k])



