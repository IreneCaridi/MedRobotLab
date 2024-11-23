import numpy as np
from collections import defaultdict


def get_mask_from_txt(file_path, img_shape: tuple = (1080, 1920), return_dict=False):
    """
    Loads polygonal mask annotations from txt and organizes them into a dictionary
    of numpy arrays, grouped by class or list of tuple with class - polygons coupling.

    args:
        - file_path: path to the text file containing mask annotations.
        - img_shape: image shape as tuple
        - return_dict: if True returns a dict else a list

    Returns: (exclusive)
        - mask_dict: dict, where keys are class labels and values are lists of numpy arrays
                     with shape (n_points, 2) for each polygon.
        - mask_list: list, of len == N° classes where each entry is a tuple with (list of np.array polys, class_id).
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

    mask_list = [(mask_dict[k], k + 1) for k in mask_dict.keys()]  # N.B. k+1 is done as classes are 0-2 but 0 is bkg

    if return_dict:
        return mask_dict
    else:
        return mask_list

