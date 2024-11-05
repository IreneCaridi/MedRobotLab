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
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from functions.sam2_functions import show_mask, show_points, show_box, show_masks, sam_predictor

# Select the device for computation
device = torch.device("cuda")
print(f"Using device: {device}")

if __name__ == '__main__':
    # Load predictor
    sam2_checkpoint = Path(parent_dir) / r'sam2/checkpoints/sam2.1_hiera_tiny.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)


    # Define the directory containing images
    folder = '../image/mmi'
    # Get a list of all image files in the directory
    image_paths = glob(os.path.join(folder, '*.png'))
    bbox_paths = glob(os.path.join(folder, '*_rect.npy'))
    pts_paths = glob(os.path.join(folder, '*_pts.npy'))
    lbs_paths = glob(os.path.join(folder, '*_lbs.npy'))


    # Load image
    for im, box, pts, lbs in zip(image_paths, bbox_paths, pts_paths, lbs_paths):
        image = Image.open(im)
        image_width, image_height = image.size
        image = np.array(image.convert("RGB"))
        pts = np.load(pts)
        lbs = np.load(lbs)

        box = np.load(box)
        boxes = box[:, :4].astype(float)

        # Process the image
        predictor.set_image(image)

        # plt.figure()
        # plt.imshow(image)
        # for box in boxes:
        #     show_box(box, plt.gca())
        # show_points(pts, lbs, plt.gca())
        # plt.axis('on')
        # plt.show()

        pred_model = 3
        n = 0
        m = 1

        if pred_model == 1:
            selected_pts = pts[:, n:m]
            selected_lbs = lbs[:, n:m]
            box = 0

            sam_predictor(image, predictor, pred_model, selected_pts, selected_lbs, box)
        elif pred_model == 2:
            selected_pts = 0
            selected_lbs = 0
            for box in boxes:
                sam_predictor(image, predictor, pred_model, selected_pts, selected_lbs, box)
        else:
            for i, box in enumerate(boxes):
                if i == 0:
                    selected_pts = pts[:, n:m]
                    selected_lbs = lbs[:, n:m]
                else:
                    selected_pts = pts[:, -3:]
                    selected_lbs = lbs[:, -3:]

                sam_predictor(image, predictor, pred_model, selected_pts, selected_lbs, box)
