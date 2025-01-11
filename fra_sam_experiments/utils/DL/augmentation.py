import numpy as np
import random


def square_crop(image: np.ndarray, crop_size: int, center: tuple = None):
    """
    Extracts a random square crop from the given image (numpy array).

    Args:
        - image (np.ndarray): The input image as a numpy array with shape (H, W) or (H, W, C).
        - crop_size (int): The desired size of the square crop.
        - center (tuple): If specified, it creates a crop around it

    Return:
        - np.ndarray: A randomly cropped square region of the input image as a numpy array.
    """
    h, w = image.shape[:2]

    # Ensure crop size does not exceed image dimensions
    if crop_size > w or crop_size > h:
        raise ValueError("Crop size is larger than image dimensions")

    dx = crop_size//2

    if not center:
        # Randomly select the center of the crop (the numbers are used be inside fov of dataset_mmi_2)
        x = random.randint(214 + dx, 1800 - dx)
        y = random.randint(114 + dx, 880 - dx)
    else:
        x, y = center
    # Crop the image

    crop = image[y - dx:y + dx, x - dx:x + dx]
    return crop, (x, y)


def get_grid_patches(image, patch_size):
    h, w = image.shape[0:2]
    crops = []

    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size

    if len(image.shape) == 3:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    padded_h, padded_w = padded_image.shape[0:2]

    for y in range(0, padded_h, patch_size):
        for x in range(0, padded_w, patch_size):
            if len(image.shape) == 3:
                crop = padded_image[y:y + patch_size, x:x + patch_size, :]
            else:
                crop = padded_image[y:y + patch_size, x:x + patch_size]
            crops.append(crop)

    return crops



