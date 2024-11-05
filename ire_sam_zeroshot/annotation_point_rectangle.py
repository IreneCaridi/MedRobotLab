import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt
import sys
import os
from PIL import Image
from pathlib import Path
from utils.plot import show_mask, show_points, show_box, show_masks, show_batch, predict_and_plot
from utils.general import check_device
from utils.active_graphic import annotate_image, obj2lab, select_images
from utils import random_state
from utils.rectangle_capture import rectangle_capture


random_state()

# placing myself in sam2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(str(Path(parent_dir) / 'sam2'))

# import pkgutil
#
# for module in pkgutil.iter_modules():
#     if 'sam' in module.name:
#         print(module.name)


# Main function to run the tool
def main():
    """
        asks the user to select some image to run sam2 on. The user can then select points inside every image
        belonging to different target classes, as well as points belonging to background (bkg).
        If 'r' is pressed, the last point in removed.
        WATCH-OUT: per every image, one must select the same n° of points per target (except bkg)!!!

    """


    # setting up sam2
    device = check_device()


    # Define the output directory
    output_dir = "../image/mmi"
    os.makedirs(output_dir, exist_ok=True)

    print("Please select images to annotate:")
    image_paths = select_images()

    imgs = []
    img_name = []
    pts_batch = []
    lbs_batch = []
    rect_batch = []
    for image_path in image_paths:

        image = Image.open(image_path)

        # batching images
        imgs.append(np.array(image.convert("RGB")))

        img_name.append(Path(image_path).name)

        print(f"\nAnnotating image: {img_name[-1]}")
        print('Select bbox')
        rectangle_coords = rectangle_capture(image_path)
        rect_batch.append(rectangle_coords)

        # print('Select points')
        # labeled_points = annotate_image(image_path)
        #
        # prompts_dict = {}
        # pts = []
        # lbs = []
        # for points, label in labeled_points:
        #     if label in prompts_dict.keys():
        #         prompts_dict[label].append(points)
        #     else:
        #         prompts_dict[label] = [points]
        #
        #     # converting labels into a list of shape (n° prompt, 1)
        #     lbs = [np.array([1 for _ in prompts_dict[k]] + [0 for _ in prompts_dict['bkg']]) if 'bkg' in prompts_dict.keys()
        #          else np.array([1 for _ in prompts_dict[k]]) for k in prompts_dict.keys() if k != 'bkg']
        #     # converting points into a list of shape (n° prompt, 1, 2)
        #     pts = [np.array([point for point in prompts_dict[k]] + prompts_dict['bkg']) if 'bkg' in prompts_dict.keys()
        #          else np.array([point for point in prompts_dict[k]]) for k in prompts_dict.keys() if k != 'bkg']
        #
        # # batching points and labels
        # pts_batch.append(pts)
        # lbs_batch.append(lbs)
        #

        # Get the name from img_name[-1]
        file_name = img_name[-1][:-4]

        # Save lbs and pts to the specified folder
        # np.save(os.path.join(output_dir, f"{file_name}_lbs.npy"), lbs)
        # np.save(os.path.join(output_dir, f"{file_name}_pts.npy"), pts)
        np.save(os.path.join(output_dir, f"{file_name}_rect.npy"), rectangle_coords)

    #assert len(pts_batch[0]) != 0, "You must select at least 1 point..."


if __name__ == "__main__":
    main()

