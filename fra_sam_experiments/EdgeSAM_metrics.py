
import torch
from pathlib import Path
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm

from utils.DL.loaders import EdgeSAMLoader
from utils.DL.collates import edgesam_collate
from utils.plot import color_map

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

# src = Path(r'C:\Users\franc\Documents\MedRobotLab\dataset\Cholect_dataset\images\test')

src = Path(r'C:\Users\franc\Documents\MedRobotLab\dataset\Cholect_dataset\to_use\images')

path_to_weights = r'C:\Users\franc\Documents\EdgeSAM\weights\edge_sam_3x.pth'

sam_checkpoint = r"C:\Users\franc\Documents\MedRobotLab\EdgeSAM\weights\edge_sam_3x.pth"
model_type = "edge_sam"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

lab_type = 'bbox'

dst = src.parent / 'masks' / f'{lab_type}'
os.makedirs(dst, exist_ok=True)

dataset = EdgeSAMLoader(src, lab_type, device=device)

loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=edgesam_collate)

IoU_dict = {i: [] for i in range(1,8,1)}

with torch.no_grad():
    for x, y in tqdm(loader, desc="Batches", bar_format=TQDM_BAR_FORMAT):
        batched_output = sam(x, num_multimask_outputs=1)

labs = [i for _, i in y]
prompts = [i['prompt_init'] for i in x]
images = [i['original_image'] for i in x]

dst = Path(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\images') / f'palle'
os.makedirs(dst, exist_ok=True)

for ms, n, p, im, ls in zip(batched_output, ['seg8k_video12_015810', 't50_VID10_000660', 't50_VID74_001380'],
                        prompts, images, labs):
    # p = np.reshape(p, (-1, 2))
    # m_np = np.zeros((480, 854)).astype(np.uint8)
    # for m in ms['masks']:
    #     m_np += m.squeeze().numpy().astype(np.uint8)
    # m_np = np.clip(m_np, a_min=0, a_max=1)
    # m_np = m_np * 255
    # Plot the original slice with overlay

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.imshow(im)
    for m, l in zip(ms['masks'], ls):
        # rgb_color = color_map[l]
        # cmap = ListedColormap([rgb_color])

        m = m.squeeze().numpy()
        overlay_pred = np.zeros((*m.shape, 3))
        overlay_pred[m == 1, :] = color_map[l]
        # overlay_pred[overlay_pred >= 1] = 1  # same color
        ax.imshow(overlay_pred, alpha=0.3)  # Red overlay for the mask

    # ax.scatter(p[:,0], p[:,1], c=[[0, 1, 0]], s=100)
    ax.axis('off')  # Remove axes

    for xyxy in p:
        top_left_x = xyxy[0]
        top_left_y = xyxy[1]
        bottom_right_x = xyxy[2]
        bottom_right_y = xyxy[3]

        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                 linewidth=2, edgecolor=[0,1,0], facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()

    plt.savefig(dst / f'{n}.png')



    # image = Image.fromarray(m_np)
    # image.save(dst / f'{n}_mask.png')

#
#     for i in range(len(batched_output)):
#         IoU = compute_iou(batched_output[i]['masks'].squeeze(1), y[i][0])
#         for iou, l in zip(IoU.numpy().tolist(), y[i][1]):
#             IoU_dict[l].append(iou)
#
# map_dict = {'grasper': 1,
#             'snare': 2,
#             'irrigator': 3,
#             'clipper': 4,
#             'scissors': 5,
#             'bipolar': 6,
#             'hook': 7}
#
# inverse_map = {map_dict[k]: k for k in map_dict.keys()}
#
# dst = Path(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data') / f'{lab_type}'
# os.makedirs(dst, exist_ok=True)
# for k in IoU_dict.keys():
#     np.save(dst / f'{inverse_map[k]}_iou.npy', IoU_dict[k])



