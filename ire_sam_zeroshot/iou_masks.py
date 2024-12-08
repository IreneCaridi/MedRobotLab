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
from torchmetrics import JaccardIndex
from PIL import Image
import numpy as np
import pandas as pd
from functions.sam2_functions import show_mask, show_points, show_box, show_masks
from functions.creating_list import get_base_filename, load_boxes, load_points, load_three_points, exctract_paths
import pickle
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import cv2
from functions.mask import extract_true_mask_json


# Select the device for computation
device = check_device()
#device = torch.device("cpu")

if __name__ == '__main__':
    # Load predictor
    sam2_checkpoint = Path(parent_dir) / r'sam2/checkpoints/sam2.1_hiera_tiny.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Define directories
    folder_images = '../image/Cholect_dataset/images/test'
    folder_points = '../image/cholect_annotation/points/test'
    folder_labels = '../image/cholect_annotation/classes/test'
    folder_bbox = '../image/cholect_annotation/bbox/test'
    folder_true = '../image/Cholect_dataset/labels/test'
    folder_three_pts = '../image/cholect_annotation/three_points/test'

    matching_image_paths, matching_bbox_paths, matching_pts_paths, matching_lbs_paths, matching_true_paths, matching_three_pts_paths = exctract_paths(folder_images, folder_bbox, folder_points, folder_labels, folder_true, folder_three_pts)

    # matching_bbox_paths = ['../image/dataset_mmi/bbox/test/image_0008.txt', '../image/dataset_mmi/bbox/test/image_0008.txt']
    # matching_image_paths = ['../image/dataset_mmi/images/test/image_0008.png', '../image/dataset_mmi/images/test/image_0021.png']
    # matching_lbs_paths = ['../image/dataset_mmi/classes/test/image_0008.txt', '../image/dataset_mmi/classes/test/image_0021.txt']
    # matching_pts_paths = ['../image/dataset_mmi/points/test/image_0008.txt', '../image/dataset_mmi/points/test/image_0021.txt']
    # matching_true_paths = ['../image/dataset_mmi/labels/test/image_0008.txt', '../image/dataset_mmi/labels/test/image_0021.txt']
    # matching_three_pts_paths = ['../image/dataset_mmi/three_points/test/image_0008.txt', '../image/dataset_mmi/three_points/test/image_0021.txt']

    option = 0
    if option == 0:
        # Use sam2 with mask
        output_folder = '../image/cholect_annotation/IoU/iou_mask_bbox.csv'
        all_results = []
        for box, lbs, image_path, true_contour in zip(matching_bbox_paths, matching_lbs_paths, matching_image_paths, matching_true_paths):
            print(f"Processing image: {image_path}")
            image = Image.open(image_path)
            image_width, image_height = image.size
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)

            bbox = load_boxes(box, image_width, image_height, 0)

            pred_masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox,
                multimask_output=False,
            )

            # Open the true contour file and extract masks
            true_masks = extract_true_mask_json(true_contour, image_width, image_height)

            # Convert masks to PyTorch tensors
            if pred_masks.shape[1] == 1:
                pred_masks = pred_masks.squeeze(1)
            pred_mask_tensor = torch.tensor(pred_masks)
            true_mask_tensor = torch.tensor(true_masks)
            # Verify shapes match
            assert pred_mask_tensor.shape == true_mask_tensor.shape, "mask and true_mask must have the same shape"

            # Initialize Jaccard Index metric for binary masks
            jaccard = JaccardIndex(task="binary", num_classes=2)

            # Compute Jaccard Index for each pair of masks
            iou_scores = []
            for i in range(pred_mask_tensor.shape[0]):
                iou = jaccard(pred_mask_tensor[i], true_mask_tensor[i])
                iou_scores.append(iou.item())

            # Read lbs file
            with open(lbs, 'r') as file:
                lbs = file.readlines()

            # Aggiungi i risultati per ogni maschera nell'immagine
            for i, iou in enumerate(iou_scores):
                all_results.append({
                    "Image": get_base_filename(image_path),
                    "Mask": lbs[i].strip(),
                    "IoU": iou,
                    "is_similar": iou > 0.5,
                })

        # Crea un DataFrame con i risultati
        results_df = pd.DataFrame(all_results)

        # Salva il DataFrame in un file CSV
        results_df.to_csv(output_folder, index=False)

    elif option == 1:
        # Use sam2 with points
        output_folder = '../image/dataset_mmi/IoU/iou_mask_pts.csv'
        all_results = []
        for pts_path, image_path, lbs, true_contour in zip(matching_pts_paths, matching_image_paths, matching_lbs_paths, matching_true_paths):
            image = Image.open(image_path)
            image_width, image_height = image.size
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)
            print(f"Processing image: {image_path}")

            # Load the points (coordinates)
            points = load_points(pts_path, image_width, image_height, 0) 
            input_point = np.array(points)
            predictor.set_image(image)
            
            pred_masks = []
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

                pred_masks.append(best_mask)

            # Convert the list to a NumPy array for further processing
            pred_masks = np.stack(pred_masks)

            # Open the true contour file and extract masks
            true_masks = extract_true_mask_json(true_contour, image_width, image_height)
            
            # Convert masks to PyTorch tensors
            pred_mask_tensor = torch.tensor(pred_masks)
            true_mask_tensor = torch.tensor(true_masks)
            # Verify shapes match
            assert pred_mask_tensor.shape == true_mask_tensor.shape, "mask and true_mask must have the same shape"

            # Initialize Jaccard Index metric for binary masks
            jaccard = JaccardIndex(task="binary", num_classes=2)

            # Compute Jaccard Index for each pair of masks
            iou_scores = []
            for i in range(pred_mask_tensor.shape[0]):
                iou = jaccard(pred_mask_tensor[i], true_mask_tensor[i])
                iou_scores.append(iou.item())

            # Read lbs file
            with open(lbs, 'r') as file:
                lbs = file.readlines()

            # Aggiungi i risultati per ogni maschera nell'immagine
            for i, iou in enumerate(iou_scores):
                all_results.append({
                    "image": get_base_filename(image_path),
                    "class": lbs[i].strip(),
                    "iou": iou,
                    "is_similar": iou > 0.5,
                })

        # Crea un DataFrame con i risultati
        results_df = pd.DataFrame(all_results)

        # Salva il DataFrame in un file CSV
        results_df.to_csv(output_folder, index=False)

    elif option == 2:
        # Use sam2 with three points
        output_folder = '../image/dataset_mmi/IoU/iou_mask_three_pts.csv'
        all_results = []
        for pts_path, image_path, lbs, true_contour in zip(matching_three_pts_paths, matching_image_paths, matching_lbs_paths, matching_true_paths):
            image = Image.open(image_path)
            image_width, image_height = image.size
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)
            print(f"Processing image: {image_path}")

            # Load the three points
            points = load_three_points(pts_path, image_width, image_height, 0) 
            input_point = np.array(points)
            predictor.set_image(image)
            
            pred_masks = []
            input_labels = []
            # Iterate over the points in steps of 3
            for idx in range(0, len(input_point), 3):
                # Create a label array with zeros
                input_label = np.zeros(len(input_point))
                # Set the 3 consecutive points to 1 (idx, idx+1, idx+2)
                input_label[idx:idx+3] = 1
                # Append the label array to the list of labels
                input_labels.append(input_label)

            # Convert the list of labels into a numpy array after the loop
            input_labels = np.array(input_labels)

            # Iterate over the input_labels for prediction
            pred_masks = []
            for input_label in input_labels:
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

                # Show the results
                # plt.figure(figsize=(10, 10))
                # plt.imshow(image)
                # show_mask(best_mask, plt.gca(), random_color=True)
                # show_points(input_point, input_label, plt.gca())
                # plt.show()

                pred_masks.append(best_mask)

            # Convert the list to a NumPy array for further processing
            pred_masks = np.stack(pred_masks)

            # Open the true contour file and extract masks
            true_masks = extract_true_mask_json(true_contour, image_width, image_height)
            
            # Convert masks to PyTorch tensors
            if pred_masks.shape[1] == 1:
                    pred_masks = pred_masks.squeeze(1)
            pred_mask_tensor = torch.tensor(pred_masks)
            true_mask_tensor = torch.tensor(true_masks)

            # Verify shapes match
            assert pred_mask_tensor.shape == true_mask_tensor.shape, "mask and true_mask must have the same shape"

            # Initialize Jaccard Index metric for binary masks
            jaccard = JaccardIndex(task="binary", num_classes=2)

            # Compute Jaccard Index for each pair of masks
            iou_scores = []
            for i in range(pred_mask_tensor.shape[0]):
                iou = jaccard(pred_mask_tensor[i], true_mask_tensor[i])
                iou_scores.append(iou.item())

            # Read lbs file
            with open(lbs, 'r') as file:
                lbs = file.readlines()

            # Aggiungi i risultati per ogni maschera nell'immagine
            for i, iou in enumerate(iou_scores):
                all_results.append({
                    "image": get_base_filename(image_path),
                    "class": lbs[i].strip(),
                    "iou": iou,
                    "is_similar": iou > 0.5,
                })

        # Crea un DataFrame con i risultati
        results_df = pd.DataFrame(all_results)

        # Salva il DataFrame in un file CSV
        results_df.to_csv(output_folder, index=False)