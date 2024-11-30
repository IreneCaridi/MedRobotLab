from cProfile import label

import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import explain_validity

def get_class_label(label_path):
    """Reads the class label from a file."""
    with open(label_path, 'r') as f:
        return f.read().strip()

def get_class_labels_from_file(label_file):
    """Reads a label file and returns a list of class labels."""
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def process_and_save_contour(masks, image_path, output_folder, label_file, image_width, image_height):
    """
    Processes multiple masks, extracts contour points for each mask, associates each with a class label, 
    and saves the results in a specified output folder as a text file. Each line in the output file contains
    a class label followed by normalized contour coordinates for a mask.

    Parameters:
        masks (list of ndarray): List of binary masks to process, each corresponding to a region of interest.
        image_path (str): Path to the image file associated with the masks, used to name the output file.
        output_folder (str): Directory where the output text file with contour points will be saved.
        label_file (str): Path to the file containing class labels, with each line representing a label
                          that corresponds to a mask in `masks`.
                          
    Raises:
        ValueError: If the number of class labels in the label file is less than the number of masks provided.
    """

    image_names = os.path.splitext(os.path.basename(image_path))[0]
    filename = os.path.join(output_folder, f"{image_names}.txt")

    os.makedirs(output_folder, exist_ok=True)

    with open(label_file, 'r') as f:
        class_labels = [line.strip() for line in f]

    if len(class_labels) < len(masks):
        raise ValueError("Mismatch: Not enough class labels provided for the number of masks.")
    with open(filename, "w") as file:
        for idx, mask_array in enumerate(masks):
            mask = mask_array.squeeze()
            class_label = class_labels[idx]

            # print(f"Processing: Image {image_names} - Mask {idx} - Class Label: {class_label}")

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

            # Normalize the contour coordinates to [0, 1] range based on the image width and height
            for contour in contours:
                # If the contour doesn't define a closed region, skip it
                if len(contour) < 3:
                    continue

                normalized_contour = []
                for point in contour:
                    x, y = point[0]  # Extract the (x, y) coordinates
                    normalized_x = x / image_width
                    normalized_y = y / image_height
                    normalized_contour.append(f"{normalized_x:.6f} {normalized_y:.6f}")

                file.write(f"{class_label} {' '.join(normalized_contour)}\n")

        print(f"Saved: {filename}")


def visualize_contours(image_path, contour_file, image_width, image_height):
    """
    Visualizes contours on an image based on the information stored in a contour file.

    Parameters:
        image_path (str): Path to the image file to display contours on.
        contour_file (str): Path to the text file containing normalized contour coordinates and class labels.
        image_width (int): Width of the image, used to re-normalize contour coordinates.
        image_height (int): Height of the image, used to re-normalize contour coordinates.
    """
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for display in matplotlib

    # Create a copy of the image to draw the contours
    contour_image = image.copy()
    label_to_color = {
        "0": (255, 0, 0),  # Red
        "1": (0, 255, 0),  # Green
        "2": (0, 0, 255),  # Blue
    }

    # Open the contour file and process each line
    with open(contour_file, 'r') as f:
        for line in f:
            # Split the line into class label and contour points
            parts = line.strip().split(' ', 1)
            class_label = parts[0]
            
            contour_data = parts[1].split(' ')

            # Create a list to hold the contour points
            contour_points = []

            # Convert normalized contour coordinates to original image scale
            for i in range(0, len(contour_data), 2):
                normalized_x = float(contour_data[i])
                normalized_y = float(contour_data[i + 1])

                # Re-normalize the contour points to the original image dimensions
                x = int(normalized_x * image_width)
                y = int(normalized_y * image_height)

                # Append the contour point as a tuple (x, y) to the contour_points list
                contour_points.append([[x, y]])

            # Convert the contour points into a numpy array suitable for OpenCV (list of points)
            contour_array = np.array(contour_points, dtype=np.int32)

            # Draw the contours on the image using a random color
            cv2.drawContours(contour_image, [contour_array], -1, label_to_color[class_label], 3)


    # Display the image with contours
    plt.imshow(contour_image)
    plt.axis('off')  # Turn off axis
    plt.show()


# Function to read and scale mask points from file
def read_mask(file_path, img_width, img_height):
    """
    Reads a mask file containing class labels and normalized contour points, 
    scales the contour points to the specified image dimensions, and returns 
    the data as a list of tuples with class labels and contour coordinates.

    Parameters:
        file_path (str): Path to the file containing class labels and contour points.
        img_width (int): Width of the image, used to scale the contour points.
        img_height (int): Height of the image, used to scale the contour points.
        
    Returns:
        list of tuples: Each tuple contains:
            - class_label (int): The class label associated with a contour.
            - coordinates (ndarray): An array of shape (N, 2) with scaled contour points, 
              where N is the number of points.

    Raises:
        ValueError: If the file format is invalid, such as having insufficient data, 
                    non-integer class labels, or an odd number of coordinates.
    """
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
    mask_data = []
    for line in lines:
        # Remove any surrounding whitespace and split by space or comma
        line = line.strip()
        parts = line.replace(',', ' ').split()

        if len(parts) < 3:
            raise ValueError(f"Insufficient data in line: '{line}' in file {file_path}")

        # Convert the first part to an integer for the class label
        try:
            class_label = int(parts[0])
        except ValueError:
            raise ValueError(f"Unexpected format in file {file_path}: '{parts[0]}' is not a valid integer.")

        # Convert the remaining parts to coordinates and scale them
        try:
            coords = [float(coord) for coord in parts[1:]]
            if len(coords) % 2 != 0:
                raise ValueError(f"Odd number of coordinates in line: '{line}' in file {file_path}")
            coordinates = np.array(coords).reshape(-1, 2)
            # Scale coordinates by image dimensions
            coordinates[:, 0] *= img_width
            coordinates[:, 1] *= img_height
        except ValueError as e:
            raise ValueError(f"Error parsing coordinates in file {file_path}: {e}")

        mask_data.append((class_label, coordinates))
    return mask_data


# Function to plot an image with SAM and ground truth masks, including contour order
def plot_image_with_masks(image_path, sam_mask, ground_truth_mask):
    """
    Plots an image with superimposed SAM and ground truth mask contours. Each contour 
    is labeled with a sequential number for easy identification, with SAM mask contours 
    in blue and ground truth mask contours in green.

    Parameters:
        image_path (str): Path to the image file to be displayed as the background.
        sam_mask (list of tuples): Each tuple contains:
            - class_label (int): The class label for the SAM mask contour.
            - contour_points (ndarray): Array of shape (N, 2) with contour points, where N 
              is the number of points in the contour.
        ground_truth_mask (list of tuples): Each tuple contains:
            - class_label (int): The class label for the ground truth mask contour.
            - contour_points (ndarray): Array of shape (N, 2) with contour points, where N 
              is the number of points in the contour.

    Displays:
        A plot of the image with SAM and ground truth contours. Each contour is marked 
        with its sequence number. The SAM mask contours are in blue, and ground truth 
        contours are in green, with a legend distinguishing them.
    """

    # Load and plot the image
    image = plt.imread(image_path)
    img_height, img_width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Plot SAM mask contours with numbering
    for idx, (class_label, contour_points) in enumerate(sam_mask):
        polygon = MplPolygon(contour_points, closed=True, fill=False, edgecolor='blue', linewidth=2,
                             label='SAM Mask' if idx == 0 else "")
        ax.add_patch(polygon)

        # Add contour number to the plot
        centroid = np.mean(contour_points, axis=0)  # Find the centroid to place the label
        ax.text(centroid[0], centroid[1], str(idx+1), color='blue', fontsize=10, ha='center', va='center')

    # Plot ground truth mask contours with numbering
    for idx, (class_label, contour_points) in enumerate(ground_truth_mask):
        polygon = MplPolygon(contour_points, closed=True, fill=False, edgecolor='green', linewidth=2,
                             label='Ground Truth Mask' if idx == 0 else "")
        ax.add_patch(polygon)

        # Add contour number to the plot
        centroid = np.mean(contour_points, axis=0)  # Find the centroid to place the label
        ax.text(centroid[0], centroid[1], str(idx+1), color='green', fontsize=10, ha='center', va='center')

    # Create custom legend to avoid duplicate labels
    handles = [
        plt.Line2D([0], [0], color='blue', lw=2, label='SAM Mask'),
        plt.Line2D([0], [0], color='green', lw=2, label='Ground Truth Mask')
    ]
    ax.legend(handles=handles, loc='upper right')

    ax.set_title(f"Image with SAM and Ground Truth Masks: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

def compute_iou(polygon1, polygon2):
    """
    Calculates the Intersection over Union (IoU) between two polygons. IoU is a metric 
    used to evaluate the overlap between two shapes, with values ranging from 0 to 1, 
    where 1 indicates complete overlap.

    Parameters:
        polygon1 (Polygon): First polygon as a Shapely Polygon object.
        polygon2 (Polygon): Second polygon as a Shapely Polygon object.

    Returns:
        float: The IoU score between the two polygons. Returns 0.0 if either polygon 
               is invalid or if the union of the two polygons has zero area.
    """

    if not polygon1.is_valid or not polygon2.is_valid:
        return 0.0
    intersection = polygon1.intersection(polygon2).area
    union = unary_union([polygon1, polygon2]).area
    return intersection / union if union != 0 else 0

# Function to compare two masks
def compare_masks(image_name, mask_pred, mask_true, iou_threshold=0.5):
    """
    Compares two sets of masks by calculating the Intersection over Union (IoU) between
    contours in `mask1` and `mask2` that share the same class label. Each contour in 
    `mask1` is matched to the best IoU contour in `mask2` that has the same class label, 
    if the IoU meets or exceeds a specified threshold. Results include IoU scores and 
    similarity status.

    Parameters:
        image_name (str): Name of the image associated with the masks.
        mask1 (list of tuples): Each tuple contains:
            - class1 (int): Class label for a contour in the first mask.
            - contour1 (ndarray): Array of shape (N, 2) with points of the contour.
        mask2 (list of tuples): Each tuple contains:
            - class2 (int): Class label for a contour in the second mask.
            - contour2 (ndarray): Array of shape (N, 2) with points of the contour.
        iou_threshold (float): Threshold for considering two contours as similar. 
                               Default is 0.5.

    Returns:
        list of dict: Each dictionary contains:
            - 'image' (str): Name of the image.
            - 'class' (int): The class label of the contour.
            - 'iou' (float): Highest IoU achieved between contours of the same class.
            - 'is_similar' (bool): True if the highest IoU meets or exceeds the threshold.
    """
    # Group the true masks by class label for easier comparison
    label_to_true_masks = {}
    for true_label, true_contour in mask_true:
        if true_label not in label_to_true_masks:
            label_to_true_masks[true_label] = []
        label_to_true_masks[true_label].append(true_contour)

    # Group the predicted masks by class label for easier comparison
    label_to_pred_masks = {}
    for pred_label, pred_contour in mask_pred:
        if pred_label not in label_to_pred_masks:
            label_to_pred_masks[pred_label] = []
        label_to_pred_masks[pred_label].append(pred_contour)


    # Compute IoU for each class label
    similarity_results = []
    for label in set(label_to_true_masks.keys()) | set(label_to_pred_masks.keys()):
        true_contours = label_to_true_masks.get(label, [])
        pred_contours = label_to_pred_masks.get(label, [])

        # If either set of contours is empty, set IoU to 0 and similarity to False
        if len(true_contours) == 0 or len(pred_contours) == 0:
            similarity_results.append({
                'image': image_name,
                'class': label,
                'iou': 0.0,
                'is_similar': False,
            })
            continue

        # Validate and create polygons
        def validate_and_fix(contours):
            valid_polygons = []
            for contour in contours:
                poly = ShapelyPolygon(contour)
                if not poly.is_valid:
                    print(f"Invalid polygon for class {label}: {explain_validity(poly)}")
                    poly = poly.buffer(0)  # Attempt to fix the polygon
                if poly.is_valid:
                    valid_polygons.append(poly)
            return valid_polygons

        true_polygons = validate_and_fix(true_contours)
        pred_polygons = validate_and_fix(pred_contours)

        # If there are no valid polygons, continue
        if not true_polygons or not pred_polygons:
            similarity_results.append({
                'image': image_name,
                'class': label,
                'iou': 0.0,
                'is_similar': False,
            })
            continue

        # Create union polygons
        true_polygon = unary_union(true_polygons)
        pred_polygon = unary_union(pred_polygons)
        
        # Calculate IoU between the two polygons
        iou = compute_iou(true_polygon, pred_polygon)

        # Save the results
        similarity_results.append({
            'image': image_name,
            'class': label,
            'iou': iou,
            'is_similar': iou >= iou_threshold,
        })

    return similarity_results

def extract_true_mask(true_contour, image_width, image_height):
    # Open the true contour file
    with open(true_contour, 'r') as file:
        lines = file.readlines()
    
    # Convert each line of normalized coordinates into pixel coordinates, save them in a list
    pixel_coords = []
    for line in lines:
        data = line.split()
        normalized_coords = np.array(data[1:], dtype=float)
        line_pixel_coords = [(int(x * image_width), int(y * image_height)) 
                        for x, y in zip(normalized_coords[::2], normalized_coords[1::2])]
        pixel_coords.append(line_pixel_coords)

    # Initialize a list to hold each mask
    true_masks = []

    # Loop through each set of pixel coordinates
    for line_pixel_coords in pixel_coords:
        # Create an empty mask for the current line
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        # Fill the polygon for the current line
        cv2.fillPoly(mask, [np.array(line_pixel_coords, dtype=np.int32)], 1)
        # Append the mask to the list
        true_masks.append(mask)

    # Convert the list to a NumPy array for further processing
    true_masks = np.stack(true_masks)

    return true_masks