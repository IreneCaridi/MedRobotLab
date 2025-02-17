
import numpy as np
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
sys.path.append(str(Path(parent_dir) / 'sam2'))

# import pkgutil
#
# for module in pkgutil.iter_modules():
#     if 'sam' in module.name:
#         print(module.name)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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

    sam2_checkpoint = Path(parent_dir) / r'sam2\checkpoints\sam2.1_hiera_tiny.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

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

    assert len(pts_batch[0]) != 0, "You must select at least 1 point..."

    try:
        predict_and_plot(predictor, imgs, pts_batch, lbs_batch, img_name=img_name)
    except ValueError:
        raise ValueError('Probably you selected different n° of points per class. '
              '\n You can select only equal n° of points per class, excluding badckground, per image. '
              '\n (That means that different images can have different n° of points per class)')

if __name__ == "__main__":
    main()

