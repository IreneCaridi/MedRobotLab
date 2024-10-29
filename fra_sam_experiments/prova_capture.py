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
from utils.active_graphic import annotate_image, obj2lab
from utils import random_state

random_state()

# placing myself in sam2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(Path(parent_dir) / 'sam2')


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# # Function to open a file dialog to select images
# def select_images():
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window
#     file_paths = filedialog.askopenfilenames(title="Select Images",
#                                              filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
#     return list(file_paths)
#
# # Function to display image and capture user points
# def draw_points(image_path):
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying in matplotlib
#
#     # List to store the points
#     points = []
#
#     def onclick(event):
#         # Store clicked points
#         ix, iy = event.xdata, event.ydata
#         if ix is not None and iy is not None:
#             points.append([int(ix), int(iy)])
#             print(f"Point selected: ({ix:.2f}, {iy:.2f})")
#             # Draw a small circle at the clicked point
#             plt.scatter(ix, iy, c='r', s=40)
#             plt.draw()
#
#     # Set up the plot and the callback for clicking
#     fig, ax = plt.subplots()
#     ax.imshow(image_rgb)
#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
#
#     plt.title("Click to select points on the image. Close when done.")
#     plt.show()
#
#     # After closing the plot, return the collected points
#     return np.array(points)
#
# # Main function to run the tool
# def main():
#     device = check_device()
#
#     sam2_checkpoint = r'C:\Users\Utente\Documents\SAM2\sam2\checkpoints\sam2.1_hiera_large.pt'
#     model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#
#     sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
#     predictor = SAM2ImagePredictor(sam2_model)
#
#     print("Please select images to annotate:")
#     image_paths = select_images()
#
#     imgs = []
#     pts_batch = []
#     lbs_batch = []
#     for image_path in image_paths:
#         image = Image.open(image_path)
#         # batching images
#         imgs.append(np.array(image.convert("RGB")))
#
#         print(f"\nAnnotating image: {image_path}")
#         labeled_points = annotate_image(image_path)
#         print(f"Points selected for {image_path}:")
#         p = []
#         l = []
#         l0 = []
#         p0 = []
#         old = 1
#         for points, label in labeled_points:
#
#             if label == old:
#                 old = label
#                 l0.append(obj2lab(label))
#                 p0.append(points)
#             else:
#                 old = label
#                 l.append(l0)
#                 p.append(p0)
#                 l0 = [obj2lab(label)]
#                 p0 = [points]
#         l.append(l0)
#         p.append(p0)
#         p = np.array(p)
#         pts_batch.append(p)
#         lbs_batch.append(np.array(l))
#
#     predict_and_plot(predictor, imgs, pts_batch, lbs_batch)
#
#     # MI PRODUCE SEMPRE SOLO UNA MAPPA... GUARDA PREDICT_AND_PLOT E RISOLVIIIIIIIIII
#
# if __name__ == "__main__":
#     main()
#
