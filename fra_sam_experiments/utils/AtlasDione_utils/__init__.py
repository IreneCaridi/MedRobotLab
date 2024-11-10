
from pathlib import Path
import os
import cv2



def save_images_from_video(dataset_path, output_folder, frame_rate=10):
    """
    Extract frames from every video and save them as images.

    args:
        - dataset_path: path to AtlasDione videos folder
        - output_folder: directory to save the extracted frames.
        - frame_rate: int, saves every `frame_rate`-th frame.
    """

    dataset_path = Path(dataset_path)
    output_folder = Path(output_folder)
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for vid in os.listdir(dataset_path):

        dst = output_folder / Path(vid).stem

        # Create the output directory if it doesn't exist
        os.makedirs(dst, exist_ok=True)

        # Load video
        cap = cv2.VideoCapture(str(dataset_path / vid))

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_count = 0
        saved_count = 0

        while True:
            # Read frame
            ret, frame = cap.read()

            # If no frame is returned, we've reached the end of the video
            if not ret:
                break

            # Save frame only if it is at the specified interval
            if frame_count % frame_rate == 0:
                frame_filename = os.path.join(dst, f"{Path(vid).stem}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

        # Release the video capture object
        cap.release()
        print(f"Extracted {saved_count} frames and saved to '{dst}'.")


