# import useful libraries
import os
import sys
from pathlib import Path
from glob import glob

# Set path to the sam2 folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(Path(parent_dir) / 'sam2')

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.general import check_device
import torch
from PIL import Image
import numpy as np
from functions.sam2_functions import show_mask, show_box
from functions.creating_list import exctract_paths, load_boxes, load_points
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from functions.mask import process_and_save_contour, visualize_contours


# Select the device for computation
#device = check_device()
device = torch.device("cpu")

if __name__ == '__main__':
    # Load predictor
    sam2_checkpoint = Path(parent_dir) / r'sam2/checkpoints/sam2.1_hiera_tiny.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Define directories
    folder_images = '../image/dataset_mmi/images/test'
    folder_points = '../image/dataset_mmi/points/test'
    folder_labels = '../image/dataset_mmi/classes/test'
    folder_bbox = '../image/dataset_mmi/bbox/test'
    folder_true = '../image/dataset_mmi/labels/test'
    folder_three_pts = '../image/dataset_mmi/three_points/test'

    matching_image_paths, matching_bbox_paths, matching_pts_paths, matching_lbs_paths, matching_true_paths, matching_three_pts_paths = exctract_paths(folder_images, folder_bbox, folder_points, folder_labels, folder_true, folder_three_pts)

    # matching_bbox_paths = ['../image/dataset_mmi/bbox/test/image_0008.txt']
    # matching_image_paths = ['../image/dataset_mmi/images/test/image_0008.png']
    # matching_lbs_paths = ['../image/dataset_mmi/classes/test/image_0008.txt']
    # matching_pts_paths = ['../image/dataset_mmi/points/test/image_0008.txt']

    # Use sam2 with mask
    output_folder = '../image/dataset_mmi/masks/test_bbox'
    for box, lbs, image_path in zip(matching_bbox_paths, matching_lbs_paths, matching_image_paths):
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        image_width, image_height = image.size
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)

        bbox = load_boxes(box, image_width, image_height)

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False,
        )

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #    show_mask(mask.squeeze(0), plt.gca(), random_color=True)
        # for box in bbox:
        #    show_box(box, plt.gca())
        # plt.savefig("contour.png")

        process_and_save_contour(masks, image_path, output_folder, lbs, image_width, image_height)


    # Use sam2 with center points
    output_folder = '../image/dataset_mmi/masks/test_pts'
    for pts_path, image_path, lbs in zip(matching_pts_paths, matching_image_paths, matching_lbs_paths):
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        image_width, image_height = image.size
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)
        
        # Load the points (coordinates)
        input_point = load_points(pts_path, image_width, image_height) 
        predictor.set_image(image)
        
        masks_file = []
        # Iterate over each point (create one-hot label for each point)
        for idx in range(len(input_point)):
            # Create a one-hot encoded label where only the current point has "1"
            input_label = np.zeros(len(input_point))
            input_label[idx] = 1

            # Make the prediction for the current label configuration
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            best_mask = masks[0]

            masks_file.append(best_mask)
           
        # Show the results
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks_file:
        #     show_mask(mask, plt.gca(), random_color=True)
        # plt.show()
        
        process_and_save_contour(masks_file, image_path, output_folder, lbs, image_width, image_height)

    #contour_file = '../image/dataset_mmi/masks/test_pts/image_0008.txt'
    #visualize_contours(image_path, contour_file, image_width, image_height)