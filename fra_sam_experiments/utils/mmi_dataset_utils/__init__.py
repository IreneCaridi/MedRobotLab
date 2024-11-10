import numpy as np
from collections import defaultdict


def load_masks(file_path, img_shape: tuple = (1080, 1920)):
    """
    Loads polygonal mask annotations from txt and organizes them into a dictionary
    of numpy arrays, grouped by class.

    args:
        - file_path: path to the text file containing mask annotations.

    Returns:
        - mask_dict: dict, where keys are class labels and values are lists of numpy arrays
                     with shape (n_points, 2) for each polygon.
    """
    mask_dict = defaultdict(list)
    y, x = img_shape

    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            class_label = int(data[0])  # First item is the class label

            # Extract (x, y) pairs and convert them to float
            coordinates = np.array([(int(float(data[i]) * x), int(float(data[i+1]) * y)) for i in range(1, len(data), 2)])

            # Append polygon to the respective class list in the dictionary
            mask_dict[class_label].append(coordinates)

    return mask_dict

