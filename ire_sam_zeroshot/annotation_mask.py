import os
import glob
from functions.creating_point_bbox import process_contours_json, process_contours
import matplotlib
matplotlib.use("tkagg")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


if __name__ == '__main__':
    folder = '../image/Cholect_dataset/labels/test'

    mask_paths = glob.glob(os.path.join(folder, '*.json'))

    for image_path in mask_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        process_contours_json(image_path, image_name)
        print(f"Processed {image_name}.")


