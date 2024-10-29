import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
import os
from PIL import Image
from pathlib import Path
from utils.plot import show_mask, show_points, show_box, show_masks, show_batch, predict_and_plot
from utils.general import check_device
from utils.active_graphic import annotate_image, obj2lab, select_images
from utils import random_state

random_state()

# placing myself in sam2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(Path(parent_dir) / 'sam2')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Main function to run the tool
def main():
    """
        asks the user to select some image to run sam2 on. The user can then select points inside every image
        belonging to different target classes, as well as points belonging to background.
        WATCH-OUT: once a point is selected, you cannot remove it!!!

    """


    # setting up sam2
    device = check_device()

    sam2_checkpoint = Path(parent_dir) / r'sam2\checkpoints\sam2.1_hiera_large.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print("Please select images to annotate:")
    image_paths = select_images()

    imgs = []
    img_name = []
    pts_batch = []
    lbs_batch = []
    for image_path in image_paths:

        image = Image.open(image_path)

        # batching images
        imgs.append(np.array(image.convert("RGB")))

        img_name.append(Path(image_path).name)

        print(f"\nAnnotating image: {img_name[-1]}")
        labeled_points = annotate_image(image_path)

        prompts_dict = {}
        pts = []
        lbs = []
        for points, label in labeled_points:
            if label in prompts_dict.keys():
                prompts_dict[label].append(points)
            else:
                prompts_dict[label] = [points]

            # converting labels into a list of shape (n° prompt, 1)
            lbs = [np.array([1 for _ in prompts_dict[k]] + [0 for _ in prompts_dict['bkg']]) if 'bkg' in prompts_dict.keys()
                 else np.array([1 for _ in prompts_dict[k]]) for k in prompts_dict.keys() if k != 'bkg']
            # converting points into a list of shape (n° prompt, 1, 2)
            pts = [np.array([point for point in prompts_dict[k]] + prompts_dict['bkg']) if 'bkg' in prompts_dict.keys()
                 else np.array([point for point in prompts_dict[k]]) for k in prompts_dict.keys() if k != 'bkg']

        # batching points and labels
        pts_batch.append(pts)
        lbs_batch.append(lbs)
    try:
        predict_and_plot(predictor, imgs, pts_batch, lbs_batch, img_name=img_name)
    except ValueError:
        raise('Probably you selected different n° of points per class. '
              '\n You can select only equal n° of points per class, excluding badckground, per image. '
              '\n (That means that different images can have different n° of points per class)')

if __name__ == "__main__":
    main()

