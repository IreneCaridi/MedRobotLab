import numpy as np
import cv2
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import torch


def int2color(integer, max_value=10):
    """
    Convert an integer to a distinct color using a colormap.

    Args:
        - integer: int, the integer to convert to a color.
        - max_value: int, the maximum possible value of the integer (for normalizing the color range).

    Returns:
           - color: str or tuple, the color suitable for plt.text.
    """
    if integer == 'bkg':
        integer = 0

    # Normalize the integer between 0 and 1 based on the max_value
    normalized_value = integer / max_value

    # Use a colormap (e.g., 'tab10', which has distinct colors) to map the integer to a color
    cmap = plt.get_cmap('tab10')  # You can choose different colormaps here (e.g., 'viridis', 'rainbow', etc.)

    # Get the color from the colormap
    color = cmap(normalized_value % 1)  # Using % 1 to keep values within the colormap range

    return color


def pad_and_resize(image: np.ndarray, target_size: int):
    """
    Pads the smaller dimension of an image symmetrically with zeros to make it square,
    then resizes it to the specified target size.

    args:
        - image: np.array, input image in numpy array format.
        - target_size: int, the desired size for both width and height after resizing.

    Returns:
        - resized_image: np.array, the padded and resized square image.
    """
    # Get current dimensions
    h, w = image.shape[:2]

    # Calculate padding
    if h > w:
        pad_width = (h - w) // 2
        padding = ((0, 0), (pad_width, h - w - pad_width), (0, 0))  # (top-bottom, left-right, color channels)
    else:
        pad_height = (w - h) // 2
        padding = ((pad_height, w - h - pad_height), (0, 0), (0, 0))

    # Pad the image
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)

    # Resize to target size
    resized_image = cv2.resize(padded_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return resized_image


def center_crop_and_resize(image: np.ndarray, target_size: int):
    """

    creates a square crop centered in the middle of the image and resizes it to target dimension

    args:
        - image: image to crop as np.array
        - target_size: final target size

    """

    h, w = image.shape[:2]
    min_dim = min(h, w)

    # Find the center crop
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Resize to target size
    resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return resized_image


def crop_mask_resize(polygons, img_shape, target_size):
    """

    takes the masks and reshapes them to new shape when reshaping is than with "crop_and_resize"

    args:
        - polygons: list of masks in image as tuples with corresponding class:
                    i.e. len(polygons) == n° classes and polygons[i] = (masks, c)
                         where masks is a list of np.ndarray polygons of c class
        - img_shape: initial shape of img
        - target_size: finale shape of img

    """

    h, w = img_shape[:2]
    min_dim = min(h, w)

    # Define the crop area
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    crop_box = [start_x, start_y, start_x + min_dim, start_y + min_dim]

    # Calculate the scale factor for resizing
    scale = target_size / min_dim

    transformed_polygons = []
    for polygon_list, c in polygons:
        transformed_polygon = []
        for polygon in polygon_list:
            # Shift the point based on crop and scale it to target size
            new_poly = np.array([[(x - start_x) * scale, (y - start_y) * scale] for x, y in polygon])

            # WARNING:
            # mi tiene anche punti della mask fuori dalla crop... capire come fare (perchè li plotta giusti...)
            # if crop_box[0] <= x <= crop_box[2] and crop_box[1] <= y <= crop_box[3]

            if len(new_poly) >= 3:
                transformed_polygon.append(new_poly)
        if transformed_polygon:
            transformed_polygons.append((transformed_polygon, c))

    return transformed_polygons


def pad_mask_resize(polygons, img_shape, target_size):
    """

    takes the masks and reshapes them to new shape when reshaping is than with "pad_and_resize"

    args:
        - polygons: list of masks in image as tuples with corresponding class:
                    i.e. len(polygons) == n° classes and polygons[i] = (masks, c)
                         where masks is a list of np.ndarray polygons of c class
        - img_shape: initial shape of img
        - target_size: finale shape of img

    """

    h, w = img_shape[:2]

    # Calculate scale to fit the longest side of the image to the target size
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Calculate padding needed to center the image in a square
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    # Adjust polygon coordinates
    transformed_polygons = []
    for polygon_list, c in polygons:
        transformed_polygon = []
        for polygon in polygon_list:
            # Scale and shift coordinates

            transformed_polygon.append((np.array([[x * scale + pad_left, y * scale + pad_top] for x, y in polygon])))

        transformed_polygons.append((transformed_polygon, c))

    return transformed_polygons


def plot_mask_over_image(image, masks: list, ax):
    """

    args:
        - image: image to plot
        - masks: list of masks in image as tuples with corresponding class:
                 i.e. len(masks) == n° classes and masks[i] = (m, c)
                      where m is a list of np.ndarray polygons of c class
        - ax: axis object to plot on

    """

    ax.imshow(image)

    for ms, c in masks:
        for m in ms:
            # Plot polygon

            polygon = Polygon(m, closed=True, edgecolor='black', facecolor=int2color(c), alpha=0.4)
            ax.add_patch(polygon)


def mask_list_to_array(mask_list, img_shape):
    """
        Convert the mask-polygons list of an image into a np.array with masks plotted

    Args:
        mask_list: List of tuples, where each tuple is (list of np.array polygons, class_id).
                   Each polygon is a numpy array of shape (num_points, 2).
        img_shape: tuple with shape of image

    Returns:
        mask: a np.array with integer masks of shape HxW (good for pytorch losses)
    """

    mask = np.zeros(img_shape, dtype=np.uint8)

    for class_idx, (polygons, class_id) in enumerate(mask_list):
        for polygon in polygons:

            mask = cv2.fillPoly(mask, [polygon.astype(np.int32)], color=class_id)

    return mask.sum(-1)


def bbox_from_poly(masks_batch, return_dict=False):
    """
    Retrieves bounding boxes for each give polygonals.

    args:
        - masks_batch: list of masks polygons lists (masks polygons in same format as above [[(polys, id)...]...] )
        - return_dict: if True it returns a list of dicts (1 per img) where each key is a class containing list of top_left
                       and bottom_right corners of bboxes. Else it returns a list like [[(bbox, id)...]...] )
                       where bbox is a tuple containing xyxy coord of bbox
                       (NOTE it is different from polys, here 1 tuple x box)
    Returns:
        bboxes_list:
    """

    bboxes_list = []
    for masks_instance in masks_batch:
        bbox_dict = {}
        for masks, class_id in masks_instance:
            bboxes = []

            for polygon in masks:

                min_x, min_y = np.min(polygon, axis=0)
                max_x, max_y = np.max(polygon, axis=0)

                bboxes.append((min_x, min_y, max_x, max_y))
                bbox_dict[class_id] = bboxes

        # sorting classes for consistency
        if return_dict:
            bboxes_list.append({k: bbox_dict[k] for k in sorted(bbox_dict.keys())})
        else:
            bb = []
            for k in sorted(bbox_dict.keys()):
                bb += [(x, k) for x in bbox_dict[k]]
            bboxes_list.append(bb)
    return bboxes_list

