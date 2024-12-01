import numpy as np
from functions.sam2_functions import show_mask
from functions.poly2mask import polytomask

label_folder = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\labels\train"
image_folder = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\images\train"
output_folder = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\labels\tip_masks"

polytomask(label_folder = label_folder, image_folder= image_folder, output_folder= output_folder, just_one_class= 2)
