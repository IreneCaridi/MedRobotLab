import os
from glob import glob
from PIL import Image
import numpy as np


# Function to get the base filename without extension for comparison
def get_base_filename(path):
    """
    Extracts the base filename (without extension) from a given file path.

    Parameters:
        path (str): The file path from which the base filename should be extracted.

    Returns:
        str: The base filename without the file extension.
    """
    #return os.path.splitext(os.path.basename(path))[0]
    return path.split('/')[-1].split('.')[0]

def exctract_paths(folder_images, folder_bbox, folder_points, folder_labels, folder_true, folder_three_pts):
    """
    Extracts the paths of matching files from the specified folders.
    """
    # Get lists of files
    image_paths = glob(os.path.join(folder_images, '*.png'))
    bbox_paths = glob(os.path.join(folder_bbox, '*.txt'))
    pts_paths = glob(os.path.join(folder_points, '*.txt'))
    lbs_paths = glob(os.path.join(folder_labels, '*.txt'))
    true_paths = glob(os.path.join(folder_true, '*.json'))
    three_pts_paths = glob(os.path.join(folder_three_pts, '*.txt'))
    print('Number of files in each folder:', len(image_paths), len(bbox_paths), len(pts_paths), len(lbs_paths), len(true_paths), len(three_pts_paths))

    # Get base filenames (without extensions) for matching
    image_names = {get_base_filename(path) for path in image_paths}
    bbox_names = {get_base_filename(path) for path in bbox_paths}
    pts_names = {get_base_filename(path) for path in pts_paths}
    lbs_names = {get_base_filename(path) for path in lbs_paths}
    true_names = {get_base_filename(path) for path in true_paths}
    three_names = {get_base_filename(path) for path in three_pts_paths}

    # Find the intersection of all sets to ensure all files have corresponding matches
    common_names = image_names & bbox_names & pts_names & lbs_names & true_names & three_names
    print('Number of matching files:', len(common_names))

    # Filter paths to include only those that have all required files
    matching_image_paths = [path for path in image_paths if get_base_filename(path) in common_names]
    matching_bbox_paths = [path for path in bbox_paths if get_base_filename(path) in common_names]
    matching_pts_paths = [path for path in pts_paths if get_base_filename(path) in common_names]
    matching_lbs_paths = [path for path in lbs_paths if get_base_filename(path) in common_names]
    matching_true_paths = [path for path in true_paths if get_base_filename(path) in common_names]
    matching_three_pts_paths = [path for path in three_pts_paths if get_base_filename(path) in common_names]

    # Sort paths to ensure they are in the same order
    matching_image_paths.sort(key=get_base_filename)
    matching_bbox_paths.sort(key=get_base_filename)
    matching_pts_paths.sort(key=get_base_filename)
    matching_lbs_paths.sort(key=get_base_filename)
    matching_true_paths.sort(key=get_base_filename)
    matching_three_pts_paths.sort(key=get_base_filename)

    return matching_image_paths, matching_bbox_paths, matching_pts_paths, matching_lbs_paths, matching_true_paths, matching_three_pts_paths


# Functions to load bounding boxes
def load_boxes(matching_bbox_paths, image_width, image_height, norm):
    """
    Loads bounding boxes from text files, converts them from normalized
    coordinates to pixel coordinates based on the image size, and returns
    a batch of bounding boxes.

    Args:
        matching_bbox_paths (str): Path to the folder containing bounding box files.

    Returns:
        np.array: A list of numpy arrays, each containing the bounding boxes for an image.
    """

    # Read the bounding box file and convert to pixel coordinates
    with open(matching_bbox_paths, 'r') as f:
        boxes = []
        for line in f:
            # Extract normalized coordinates from the file
            xmin, ymin, xmax, ymax = map(float, line.strip().split(','))

            if norm == 1:
                xmin = xmin*image_width
                ymin = ymin*image_height
                xmax = xmax*image_width
                ymax = ymax*image_height

            # Add the bounding box to the list
            boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        boxes_array = np.array(boxes)
    return boxes_array

def load_points(file_path, image_width, image_height, norm):
    """
    Loads normalized point coordinates from a file, scales them according to the 
    specified image dimensions, and returns them as an array.

    Parameters:
        file_path (str): Path to the file containing normalized point coordinates.
        image_width (int): Width of the image to scale the x-coordinates.
        image_height (int): Height of the image to scale the y-coordinates.

    Returns:
        ndarray: A NumPy array of shape (N, 2), where N is the number of points. 
                 Each row contains the scaled x and y coordinates of a point.
    """

    with open(file_path, 'r') as f:
        points = []
        for line in f:
            # Extract normalized coordinates from the file
            x, y = map(float, line.strip().split(','))

            if norm == 1:
                x = x*image_width
                y = y*image_height


            # Add the bounding box to the list
            points.append([int(x), int(y)])
        points_array = np.array(points)
    return points_array


def load_three_points(file_path, image_width, image_height, norm):
    """
    Loads normalized point coordinates from a file, scales them according to the 
    specified image dimensions, and returns them as an array.

    Parameters:
        file_path (str): Path to the file containing normalized point coordinates.
        image_width (int): Width of the image to scale the x-coordinates.
        image_height (int): Height of the image to scale the y-coordinates.

    Returns:
        ndarray: A NumPy array of shape (N, 6), where N is the number of points. 
                 Each row contains the scaled x and y coordinates of a point.
    """

    with open(file_path, 'r') as f:
        points = []
        for line in f:
            # Extract normalized coordinates from the file
            x, y, x_min, y_min, x_max, y_max = map(float, line.strip().split(','))

            if norm == 1:
                x = x*image_width
                y = y*image_height
                x_min = x_min*image_width
                y_min = y_min*image_height
                x_max = x_max*image_width
                y_max = y_max*image_height


            points.append([int(x), int(y)])
            points.append([int(x_min), int(y_min)])
            points.append([int(x_max), int(y_max)])
            
        points_array = np.array(points)
    return points_array



