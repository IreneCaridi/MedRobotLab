import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_batched(image_dir, batch_size):
    # Initialize the variables
    sum_r, sum_g, sum_b = 0, 0, 0
    sum_sq_r, sum_sq_g, sum_sq_b = 0, 0, 0
    total_pixels = 0

    file_names = os.listdir(image_dir)

    for i in range(0, len(file_names), batch_size):
        batch_files = file_names[i:i+batch_size]
        for file_name in batch_files:
            file_path = os.path.join(image_dir, file_name)
            try:
                # Load image and convert to RGB
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img, dtype=np.float64)  # Ensure no overflow

                # Update sums and counts
                sum_r += np.sum(img_array[:, :, 0])
                sum_g += np.sum(img_array[:, :, 1])
                sum_b += np.sum(img_array[:, :, 2])

                sum_sq_r += np.sum(img_array[:, :, 0] ** 2)
                sum_sq_g += np.sum(img_array[:, :, 1] ** 2)
                sum_sq_b += np.sum(img_array[:, :, 2] ** 2)

                total_pixels += img_array.shape[0] * img_array.shape[1]
            except Exception as e:
                print(f"Errore nell'elaborare {file_name}: {e}")
            progress_bar.update(1)
    return sum_r, sum_g, sum_b, sum_sq_r, sum_sq_g, sum_sq_b, total_pixels


data_dir = r"C:\Users\User\Desktop\datasets\dataset_self"

# Count total files
tot_files = 0
for dir_1 in os.listdir(data_dir):
    single_dataset = os.path.join(data_dir, dir_1)
    for dir_2 in os.listdir(os.path.join(single_dataset, "images")):
        if dir_2 in ['train', 'val']:
            img_dir = os.path.join(single_dataset, "images", dir_2)
            file_count = sum(len(files) for _, _, files in os.walk(img_dir))
            tot_files += file_count

# Initialize totals
tot_sum_b, tot_sum_g, tot_sum_r = 0, 0, 0
tot_sq_b, tot_sq_g, tot_sq_r = 0, 0, 0
sum_of_all_pixels = 0

progress_bar = tqdm(total=tot_files, desc="sto pensando forte")
for dir_1 in os.listdir(data_dir):
    single_dataset = os.path.join(data_dir, dir_1)
    for dir_2 in os.listdir(os.path.join(single_dataset, "images")):
        if dir_2 in ['train', 'val']:
            img_dir = os.path.join(single_dataset, "images", dir_2)
            sum_r, sum_g, sum_b, sum_sq_r, sum_sq_g, sum_sq_b, total_pixels = process_batched(img_dir, 32)
            tot_sum_r += sum_r
            tot_sum_g += sum_g
            tot_sum_b += sum_b
            tot_sq_r += sum_sq_r
            tot_sq_g += sum_sq_g
            tot_sq_b += sum_sq_b
            sum_of_all_pixels += total_pixels

# Calculate global mean and variance
mean_r = tot_sum_r / sum_of_all_pixels
mean_g = tot_sum_g / sum_of_all_pixels
mean_b = tot_sum_b / sum_of_all_pixels

var_r = (tot_sq_r / sum_of_all_pixels) - mean_r ** 2
var_g = (tot_sq_g / sum_of_all_pixels) - mean_g ** 2
var_b = (tot_sq_b / sum_of_all_pixels) - mean_b ** 2

print(f"Media: R={mean_r}, G={mean_g}, B={mean_b}")
print(f"Varianza: R={var_r}, G={var_g}, B={var_b}")
