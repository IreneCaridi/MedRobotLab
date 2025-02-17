import os
from pathlib import Path
import json
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


map_dict = {'grasper': 1,
            'snare': 2,
            'irrigator': 3,
            'clipper': 4,
            'scissors': 5,
            'bipolar': 6,
            'hook': 7}


def get_mask_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    masks = []
    for m in range(len(data['shapes'])):
        polygon_coords = np.array(data['shapes'][m]['points'])
        lab = data['shapes'][m]['label']
        masks.append((polygon_coords, map_dict[lab]))

    return masks  # to be consistent with format I'm using to handle masks for images (1 => instrument)


def plot_mask_from_json(image_path, json_path, ax):
    """
        For CholectInstanceSeg dataset plots mask over corresponding image on a specified ax object
    args:
        - image_path: path to image
        - json_path: path to corresponding json file
        - ax: matplotlib ax object

    """

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Open image
    image = Image.open(image_path)

    ax.imshow(image)
    ax.set_title(f'{Path(image_path).name}')

    for m in range(len(data['shapes'])):
        if data['shapes'][m]['points']:
            polygon_coords = np.array(data['shapes'][m]['points'])

            # Plot polygon
            polygon = Polygon(polygon_coords, closed=True, edgecolor='black', facecolor='red', alpha=0.4)
            ax.add_patch(polygon)



def random_plot_from_json(dataset_path, n=6):
    """

    args:
        - dataset_path: path to CholectInstanceSeg dataset
        - n: n° of images to plot together

    """

    dataset_path = Path(dataset_path) / 'train'
    folders = os.listdir(dataset_path)

    # randomly select one folder
    i = np.random.randint(0, len(folders))
    now_folder = folders[i]

    # randomly select n images
    files = os.listdir(dataset_path / now_folder / 'img_dir')
    imgs_names = [files[x] for x in np.random.randint(0, len(files), n)]
    ann_names = [Path(x).stem + '.json' for x in imgs_names]

    f, axs = plt.subplots(int(np.floor( n /(n // 2))), n // 2)
    f.suptitle(f'from train/{now_folder}')

    for i, (im, lb) in enumerate(zip(imgs_names, ann_names)):
        plot_mask_from_json(dataset_path / now_folder / 'img_dir' / im, dataset_path / now_folder / 'ann_dir' / lb,
                            axs.flatten()[i])

    plt.show()
