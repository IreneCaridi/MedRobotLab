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

from utils.general import check_device
import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from functions.mask import read_mask, compare_masks, visualize_contours_file
from functions.creating_list import get_base_filename


# Select the device for computation
device = check_device()
# device = torch.device("cpu")


if __name__ == '__main__':
     # Define the directory containing images, masks_sam, masks_original
    folder_images = '../image/dataset_mmi/images/test/'
    folder_masks_sam = '../image/dataset_mmi/masks/test_bbox'
    folder_masks_original = '../image/dataset_mmi/labels/test'

    # Get lists of all mask files
    masks_sam_paths = glob(os.path.join(folder_masks_sam, '*.txt'))
    masks_original_paths = glob(os.path.join(folder_masks_original, '*.txt'))

    # Create sets of base filenames for SAM and original masks
    sam_names = {get_base_filename(path) for path in masks_sam_paths}
    original_names = {get_base_filename(path) for path in masks_original_paths}

    # Find intersection of base names to get matching masks
    matching_names = sam_names & original_names  # Intersection of SAM and original mask names

    # Filter lists to keep only paths that have matching SAM and original masks
    matching_masks_sam_paths = [path for path in masks_sam_paths if get_base_filename(path) in matching_names]
    matching_masks_original_paths = [path for path in masks_original_paths if get_base_filename(path) in matching_names]

    # Sort paths to ensure they are in the same order
    matching_masks_sam_paths.sort(key=get_base_filename)
    matching_masks_original_paths.sort(key=get_base_filename)

    # Print the number of matches found
    print(f"Number of matching SAM masks: {len(matching_masks_sam_paths)}")
    print(f"Number of matching original masks: {len(matching_masks_original_paths)}")

    # matching_masks_sam_paths = ['../image/dataset_mmi/masks/test_pts/image_0008.txt']
    # matching_masks_original_paths = ['../image/dataset_mmi/labels/test/image_0008.txt']
    results = []

    # Compare each pair of corresponding masks
    for mask_sam_path, mask_original_path in zip(matching_masks_sam_paths, matching_masks_original_paths):
        print(mask_original_path)
        print(mask_sam_path)
        chosen_image = get_base_filename(mask_sam_path)
        image_path = folder_images + chosen_image + ".png"

        # Load the image to get dimensions
        image = plt.imread(image_path)
        img_height, img_width = image.shape[:2]


        # Load the masks
        mask_sam = read_mask(mask_sam_path, img_width, img_height)
        mask_original = read_mask(mask_original_path, img_width, img_height)
        
        # Compare the masks and print results
        comparison_results = compare_masks(
            image_name=chosen_image,
            mask_pred=mask_sam,
            mask_true=mask_original,
            iou_threshold=0.3
        )

        results.extend(comparison_results)

        # print(f"Comparison results for {chosen_image}:")
        # for result in comparison_results:
        #     print(f" Image: {result['image']}, Class: {result['class']}, IoU: {result['iou']:.2f}, Similar: {result['is_similar']}")

        # visualize_contours_file(image_path, mask_original_path, img_width, img_height)
        # visualize_contours_file(image_path, mask_sam_path, img_width, img_height)

    df_iou = pd.DataFrame(results)

    folder_path = "../image/dataset_mmi/IoU"
    file_name = "iou_results_test_bbox_final.csv"

    # Ensure the folder exists; if not, create it
    os.makedirs(folder_path, exist_ok=True)

    # Save the DataFrame to the specified folder as a CSV file
    file_path = os.path.join(folder_path, file_name)
    df_iou.to_csv(file_path, index=False)
