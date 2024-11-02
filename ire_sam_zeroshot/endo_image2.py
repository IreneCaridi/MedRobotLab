# import useful libraries
import os
import sys
from pathlib import Path
from glob import glob

# Set directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(Path(parent_dir) / 'sam2')

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import modules from sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import libraries
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import json
from collections import defaultdict

# Select the device for computation
device = torch.device("cpu")
print(f"Using device: {device}")

# Define utils
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def sam_function(image, predictor, input_box):
    # Show features
    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

    # Predict with SAM2ImagePredictor.predict

    # Predict using points
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True)

    # Predict using bbox
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # Show masks

    # Show masks using points
    # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

    # Show masks using bbox
    show_masks(image, masks, scores, box_coords=input_box)

def convert_xywh_to_xyxy(bbox):
    center_x, center_y, width, height = bbox

    # Calculate min and max coordinates
    x_min = center_x
    y_min = center_y
    x_max = center_x + height
    y_max = center_y + width

    return (x_min, y_min, x_max, y_max)


if __name__ == '__main__':
    # Load predictor
    # predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    sam2_checkpoint = Path(parent_dir) / r'sam2/checkpoints/sam2.1_hiera_small.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Define the directory containing images and annotation file
    image_folder = '../image/endoscapes/train'
    annotation_path = '../../image/endoscapes/train/annotation_coco.json'

    # Get a list of all image files in the directory
    image_paths = glob(os.path.join(image_folder, '*.jpg'))
    image_files = {os.path.basename(path) for path in image_paths}
    print(f"Number of images in directory: {len(image_files)}")

    # Load the COCO annotations file
    with open(annotation_path, 'r') as file:
        coco_annotations = json.load(file)
    print("Keys in COCO annotations:", coco_annotations.keys())

    # Create dictionaries for easy lookup
    image_id_map = {img['id']: img for img in coco_annotations['images'] if img['file_name'] in image_files}
    annotations_map = defaultdict(list)
    for ann in coco_annotations['annotations']:
        annotations_map[ann['image_id']].append(ann)
    categories_map = {cat['id']: cat for cat in coco_annotations['categories']}

    # Displaying categories, images, and annotations
    print("\nNumber of categories:", len(coco_annotations['categories']))
    print("Number of images:", len(coco_annotations['images']))
    print("Number of annotations:", len(coco_annotations['annotations']))

    # Display a sample from each key
    print("\nSample category:", coco_annotations['categories'][0])
    print("Sample image:", coco_annotations['images'][0])
    print("Sample annotation:", coco_annotations['annotations'][0])

    # Extract information for each image file in the folder, filtering for 'tool' supercategory
    results = []
    for img_id, img_info in image_id_map.items():
        img_annotations = annotations_map[img_id]

        # Collect associated category details, only for 'tool' supercategory
        img_data = {
            "file_name": img_info['file_name'],
            "image_id": img_id,
            "annotations": []
        }
        for ann in img_annotations:
            category_id = ann['category_id']
            category_info = categories_map.get(category_id, {})

            # Filter for annotations with 'supercategory' equal to 'tool'
            if category_info.get("supercategory") == "tool":
                img_data["annotations"].append({
                    "annotation_id": ann['id'],
                    "bbox": convert_xywh_to_xyxy(ann['bbox']),
                    "area": ann['area'],
                    "category": category_info.get("name"),
                    "supercategory": category_info.get("supercategory")
                })

        # Only add to results if there are matching 'tool' annotations
        if img_data["annotations"]:
            results.append(img_data)
    print(f"Number of matching images with 'tool' annotations: {len(results)}")
    print("Sample result:", results[0:3])

    # Process each image with 'tool' annotations
    for img_data in results[0:1]:
        image_path = os.path.join(image_folder, img_data['file_name'])
        print(image_path)
        image = Image.open("../../image/endoscapes/train/8_14775.jpg")
        image = np.array(image.convert("RGB"))
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.show()
        print(f"Image type: {type(image)}, Image shape: {image.shape}, Image dtype: {image.dtype}")

        # Process the image
        predictor.set_image(image)

        for annotation in img_data["annotations"]:
            box = annotation["bbox"]

            # Visualize the box on the image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_box(box, plt.gca())
            plt.axis('on')
            plt.show()

            # Call sam_function for the current image and bounding box
            sam_function(image, predictor, box)

