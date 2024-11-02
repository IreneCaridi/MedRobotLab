# import useful libraries
import os
import sys
from pathlib import Path
from glob import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(Path(parent_dir) / 'sam2')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import modules from sam2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

# Select the device for computation
device = torch.device("cpu")
print(f"Using device: {device}")

# Define utils (as you already have)
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

def sam_function(image, predictor, input_point, input_label):

    # Show features
    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

    # Predict with SAM2ImagePredictor.predict
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # Show masks
    show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)


if __name__ == '__main__':

    # Load predictor
    # predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    sam2_checkpoint = Path(parent_dir) / r'sam2/checkpoints/sam2.1_hiera_tiny.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Define the directory containing images
    image_folder = '../image'

    # Get a list of all image files in the directory
    image_paths = glob(os.path.join(image_folder, '*.png'))

    # Define input points and labels
    input_points = [np.array([[410, 100]]), np.array([[200, 75]])]
    input_labels = [np.array([1])] * len(image_paths)

    for image_path, input_point, input_label in zip(image_paths, input_points, input_labels):
        # Load image
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        # Process the image
        predictor.set_image(image)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show()

        # Call sam_function for the current image
        sam_function(image, predictor, input_point, input_label)