import os
import numpy as np

# Paths to input folders and output folder
input_folder_1 = r"C:\Users\User\Desktop\uni_matteo\quinto_anno\laboratorio_robotics\project\predictions_zeroshot_sam2_video_mmi\single_masks\video_segments_shaft_1"
input_folder_2 = r"C:\Users\User\Desktop\uni_matteo\quinto_anno\laboratorio_robotics\project\predictions_zeroshot_sam2_video_mmi\single_masks\video_segments_shaft_2"
output_folder = r"C:\Users\User\Desktop\uni_matteo\quinto_anno\laboratorio_robotics\project\predictions_zeroshot_sam2_video_mmi\merged_masks\shaft"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get the list of file names from both folders
files1 = set(os.listdir(input_folder_1))
files2 = set(os.listdir(input_folder_2))

# Find common files between the two folders
common_files = files1.intersection(files2)

# Merge masks and save them
for file_name in common_files:
    if file_name.endswith('.npz'):  # Ensure only .npz files are processed
        # Load the two masks
        mask1_path = os.path.join(input_folder_1, file_name)
        mask2_path = os.path.join(input_folder_2, file_name)



        mask1 = np.load(mask1_path)['1']  # Load array from npz
        mask2 = np.load(mask2_path)['1']

        # Ensure masks are boolean
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)

        # Merge the masks with logical OR
        merged_mask = np.logical_or(mask1, mask2)

        # Save the merged mask as a compressed .npz file
        output_path = os.path.join(output_folder, file_name)
        np.savez_compressed(output_path, merged_mask)

print(f"Merged masks saved to {output_folder} in .npz format")