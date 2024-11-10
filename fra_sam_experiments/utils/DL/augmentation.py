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

