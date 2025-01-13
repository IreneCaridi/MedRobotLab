import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import os
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.patches as patches
import random
from tqdm import tqdm


from fra_sam_experiments.utils.CholectinstanceSeg_utils import get_mask_from_json
from fra_sam_experiments.utils.image_handling import mask_list_to_array, bbox_from_poly

colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, 6))
class_colormap = ListedColormap(colors)


class ClosenessChecker:
    def __init__(self, th=50):   # th in pixels
        self.old_bboxs = []
        self.th = th

    def update(self, box):
        self.old_bboxs = [b for b, _ in box]

    def is_new_instance(self, new):
        """ a new instance is when one corner is significally different from any of the old one"""
        for b in self.old_bboxs:
            diff = np.abs(np.array(b) - np.array(new))
            if np.all(diff <= self.th):
                # if it similar to at least one, then it is not new
                return False
        return True


    def check(self, current_bboxs):
        current_bboxs = [b for b, _ in current_bboxs]

        for new in current_bboxs:
            if self.is_new_instance(new):
                return True
        return False

    # def check(self, current_bboxs):
    #     current_bboxs = [b for b, _ in current_bbox]
    #     if len(self.old_bboxs) != len(current_bboxs):
    #         return True
    #
    #     diff = np.abs(np.array(self.old_bboxs[:, :, None]) - np.array(current_bboxs))
    #     if np.any(diff >= self.th):
    #         return True
    #     else:
    #         return False
    #     # for old, now in zip(self.old_bbox, current_bbox):
    #     #     if np.any(np.abs((np.array(old)) - np.array(now))) <= self.th:
    #     #
    #     #     else:
    #     #         self.old_bbox = current_bbox
    #     #         return False

    def __call__(self, current_bbox):
        return self.check(current_bbox)


checker = ClosenessChecker()

# src = Path(r"D:\poli_onedrive_backup\CholectInstanceSeg\images\VID01_t50_full")
src = Path(r'D:\poli_onedrive_backup\CholectInstanceSeg\images\VID12_seg8k')


data = []

print('loading data...')
for i in tqdm(os.listdir(src)):
    img = np.array(Image.open(src / i).convert('RGB'))
    poly = get_mask_from_json(Path(str(src).replace('images', 'labels')) / Path(i).with_suffix('.json'))
    lab = mask_list_to_array(poly, img.shape)
    bbox = bbox_from_poly([poly])
    data.append((img, lab, bbox))


frame_height, frame_width = img.shape[:2]  # Dimensions of each slice
fps = 25  # Frames per second
# video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frames = []

print('creating video...')
# Create frames for each slice ( VS )
i = 0
for x, y, bboxs in tqdm(data):  # Iterate through all slices

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)

    # Overlay preparation
    overlay_pred = np.zeros((*y.shape, 3))
    overlay_pred[:, :, 0] = y  # Red channel for the mask
    overlay_pred[overlay_pred >= 1] = 1  # same color
    # Plot the original slice with overlay
    ax.imshow(x)  # Grayscale background
    ax.imshow(overlay_pred, cmap='Reds', alpha=0.4)  # Red overlay for the mask
    ax.axis('off')  # Remove axes
    plt.tight_layout()

    if i == 0 or (i > 0 and checker(bboxs)):
        for box, _ in bboxs:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
            frame = frame[:, :, :3]  # Drop the alpha channel to convert RGBA to RGB
        for _ in range(50):  # 25 frame uguali (2 sec)
            frames.append(frame)
    else:
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
        frame = frame[:, :, :3]  # Drop the alpha channel to convert RGBA to RGB
        frames.append(frame)

    checker.update(bboxs)
    plt.close(fig)  # Close the figure to save memory
    i += 1

# Finalize and release the video writer
# video_writer.release()
imageio.mimwrite(f'fra_sam_experiments/data/videos/{"prova_con_box"}.mp4', frames, fps=fps, codec='libx264')

print("Video saved")

# src = Path(r"D:\poli_onedrive_backup\dataset_self\Cholect_dataset\images\test\seg8k_video17_001823.png")
#
# data = {}
#
# map_dict0 = {'grasper': 1,
#             'snare': 2,
#             'irrigator': 3,
#             'clipper': 4,
#             'scissors': 5,
#             'bipolar': 6,
#             'hook': 7}
#
# map_dict = {map_dict0[k]: k for k in map_dict0.keys()}
#
# # for i in random.shuffle(os.listdir(src)):
# #     p = src / i
# #     x = np.array(Image.open(p).convert('RGB'))
# #     poly = get_mask_from_json(Path(str(p).replace('images', 'labels')).with_suffix('.json'))
# #     y = mask_list_to_array(poly, x.shape)
# #
# #     for _, class_id in poly:
# #         if map_dict[class_id] in data.keys():
# #             continue
# #         else:
# #             data[class_id] = (x, y)
# #
# # for k in data.keys():
# #     x, y = data[k]

# x = np.array(Image.open(src).convert('RGB'))
# poly = get_mask_from_json(Path(str(src).replace('images', 'labels')).with_suffix('.json'))
# y = mask_list_to_array(poly, x.shape)
#
# fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
# overlay_pred = np.zeros((*y.shape, 3))
# overlay_pred[:, :, 0] = y  # Red channel for the mask
#
# # Plot the original slice with overlay
# ax.imshow(x)  # Grayscale background
# ax.imshow(overlay_pred, cmap='Reds', alpha=0.3)  # Red overlay for the mask
# ax.axis('off')  # Remove axes
# plt.tight_layout()
#
# plt.savefig(f'fra_sam_experiments/data/images/grasper.png')
