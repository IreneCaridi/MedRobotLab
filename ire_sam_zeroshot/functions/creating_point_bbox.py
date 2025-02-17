import numpy as np
import os
import glob
import math
import json


# Function to calculate the center point inside the contour
def get_center_point(contour_points):
    """
    Calculates the center point of a given contour by averaging the x and y coordinates.

    Parameters:
        contour_points (list of tuples): List of (x, y) points representing the contour.

    Returns:
        tuple: The center point (center_x, center_y) of the contour.
    """

    x_coords, y_coords = zip(*contour_points)
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (center_x, center_y)


# Function to calculate the bounding box of the contour
def get_bounding_box(contour_points):
    """
    Calculates the bounding box of a given contour. The bounding box is the smallest 
    rectangle that contains all the points of the contour.

    Parameters:
        contour_points (list of tuples): List of (x, y) points representing the contour.

    Returns:
        tuple: The bounding box (min_x, min_y, max_x, max_y) coordinates of the contour.
    """

    x_coords, y_coords = zip(*contour_points)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return (min_x, min_y, max_x, max_y)

def calculate_proportional_distance(bounding_box, scale_factor):
    """
    Calcola una distanza proporzionale alla diagonale del bounding box del contorno.

    Parameters:
        contour_points (list of tuples): Lista di punti (x, y) del contorno.
        scale_factor (float): Frazione della diagonale da usare come distanza.

    Returns:
        float: Distanza proporzionale.
    """
    # Calcola il bounding box
    min_x, min_y, max_x, max_y = bounding_box

    # Diagonale del bounding box
    diagonal = ((max_x - min_x)**2 + (max_y - min_y)**2)**0.5

    # Distanza proporzionale
    return scale_factor * diagonal


import math

def get_three_point_mediana(bounding_box, contour_points, center_point, scale_factor=0.1):
    """
    Calcola una linea mediana tra due punti opposti del contorno e due punti equidistanti lungo questa linea,
    con un controllo per verificare che i punti non escano dai limiti del contorno.

    Parameters:
        bounding_box (tuple): Coordinati del bounding box (min_x, min_y, max_x, max_y).
        contour_points (list of tuples): Lista di punti (x, y) che rappresentano il contorno.
        center_point (tuple): Centro del contorno (center_x, center_y).
        scale_factor (float): Frazione della diagonale da usare come distanza.

    Returns:
        tuple: Centro e due punti equidistanti (center_x, center_y, point1_x, point1_y, point2_x, point2_y).
    """
    center_x, center_y = center_point
    min_x, min_y, max_x, max_y = bounding_box

    # Direzione della diagonale del bounding box
    dir_x, dir_y = max_x - min_x, max_y - min_y
    magnitude = math.sqrt(dir_x**2 + dir_y**2)

    # Normalizza il vettore direzione
    dir_x, dir_y = dir_x / magnitude, dir_y / magnitude

    # Calcola una distanza proporzionale
    distance = calculate_proportional_distance(bounding_box, scale_factor)

    # Calcola i due punti equidistanti lungo la linea
    point1 = (center_x + distance * dir_x, center_y + distance * dir_y)
    point2 = (center_x - distance * dir_x, center_y - distance * dir_y)

    # Verifica che i punti siano dentro il contorno
    point1 = clip_to_contour(point1, contour_points)
    point2 = clip_to_contour(point2, contour_points)

    return (center_x, center_y) + point1 + point2

def clip_to_contour(point, contour_points):
    """
    Verifica se un punto è dentro il contorno e lo clippa dentro il contorno se necessario.

    Parameters:
        point (tuple): Un punto (x, y).
        contour_points (list of tuples): Lista di punti (x, y) che rappresentano il contorno.

    Returns:
        tuple: Un punto (x, y) che è dentro il contorno.
    """
    x, y = point

    # Controlla se il punto è dentro il contorno usando il test ray-casting
    if not is_point_in_polygon(x, y, contour_points):
        # Se il punto è fuori, clippalo al punto più vicino sul contorno
        point = closest_point_on_contour(x, y, contour_points)

    return point

def is_point_in_polygon(x, y, polygon):
    """
    Determina se un punto (x, y) è dentro un poligono usando il metodo di ray-casting.

    Parameters:
        x (float): La coordinata x del punto.
        y (float): La coordinata y del punto.
        polygon (list of tuples): Una lista di punti (x, y) che definiscono il poligono.

    Returns:
        bool: True se il punto è dentro il poligono, altrimenti False.
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def closest_point_on_contour(x, y, contour_points):
    """
    Trova il punto più vicino nel contorno al punto dato.

    Parameters:
        x (float): La coordinata x del punto.
        y (float): La coordinata y del punto.
        contour_points (list of tuples): Una lista di punti (x, y) che rappresentano il contorno.

    Returns:
        tuple: Il punto più vicino (x, y) sul contorno.
    """
    closest_point = None
    min_dist = float('inf')

    for (px, py) in contour_points:
        dist = math.sqrt((x - px)**2 + (y - py)**2)
        if dist < min_dist:
            min_dist = dist
            closest_point = (px, py)

    return closest_point


# Main function to process data and save the results
def process_contours(file_path, image_name):
    """
    Processes the contours from the given file, calculates the center points and bounding boxes, 
    and saves the results in specific output files for points, bounding boxes, and labels.

    Parameters:
        file_path (str): Path to the input file containing the contour data.
        image_name (str): Name of the image for which the contours are processed.
    """

    output_folder_pts = '../image/dataset_mmi/points/test'
    output_folder_cls = '../image/dataset_mmi/classes/test'
    output_folder_mks = '../image/dataset_mmi/bbox/test'
    output_folder_three_points = '../image/dataset_mmi/three_points/test'

    os.makedirs(output_folder_pts, exist_ok=True)
    os.makedirs(output_folder_cls, exist_ok=True)
    os.makedirs(output_folder_mks, exist_ok=True)
    os.makedirs(output_folder_three_points, exist_ok=True)
    print(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    centers_path = os.path.join(output_folder_pts, f"{image_name}.txt")
    boxes_path = os.path.join(output_folder_mks, f"{image_name}.txt")
    labels_path = os.path.join(output_folder_cls, f"{image_name}.txt")
    three_points_path = os.path.join(output_folder_three_points, f"{image_name}.txt")

    with open(centers_path, 'w') as f_centers, \
            open(boxes_path, 'w') as f_boxes, \
            open(labels_path, 'w') as f_labels, \
            open(three_points_path, 'w') as f_three_points:
        for line in lines:
            data = line.split()
            label = int(data[0])
            points = list(map(float, data[1:]))

            # Divide the points into x and y coordinates
            contour_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

            center_point = get_center_point(contour_points)
            bounding_box = get_bounding_box(contour_points)
            three_points = get_three_point_mediana(bounding_box, contour_points, center_point)

            # Write the results to the respective files
            f_centers.write(f"{center_point[0]:.6f}, {center_point[1]:.6f}\n")
            f_boxes.write(f"{bounding_box[0]:.6f}, {bounding_box[1]:.6f}, {bounding_box[2]:.6f}, {bounding_box[3]:.6f}\n")
            f_labels.write(f"{label}\n")
            f_three_points.write(
                f"{three_points[0]:.6f}, {three_points[1]:.6f}, {three_points[2]:.6f}, "
                f"{three_points[3]:.6f}, {three_points[4]:.6f}, {three_points[5]:.6f}\n"
            )
            

def process_contours_json(file_path, image_name):
    """
    Processes the contours from the given file, calculates the center points and bounding boxes, 
    and saves the results in specific output files for points, bounding boxes, and labels.

    Parameters:
        file_path (str): Path to the input file containing the contour data.
        image_name (str): Name of the image for which the contours are processed.
    """
    # Output folder paths
    output_folder_pts = '../image/cholect_annotation/points/test'
    output_folder_cls = '../image/cholect_annotation/classes/test'
    output_folder_mks = '../image/cholect_annotation/bbox/test'
    output_folder_three_points = '../image/cholect_annotation/three_points/test'

    # Create necessary directories if they don't exist
    os.makedirs(output_folder_pts, exist_ok=True)
    os.makedirs(output_folder_cls, exist_ok=True)
    os.makedirs(output_folder_mks, exist_ok=True)
    os.makedirs(output_folder_three_points, exist_ok=True)
    
    # Open the JSON file and load data
    with open(file_path, 'r') as f:
        lines = json.load(f)

    # File paths for output
    centers_path = os.path.join(output_folder_pts, f"{image_name}.txt")
    boxes_path = os.path.join(output_folder_mks, f"{image_name}.txt")
    labels_path = os.path.join(output_folder_cls, f"{image_name}.txt")
    three_points_path = os.path.join(output_folder_three_points, f"{image_name}.txt")

    # Open the output files for writing
    with open(centers_path, 'w') as f_centers, \
         open(boxes_path, 'w') as f_boxes, \
         open(labels_path, 'w') as f_labels, \
         open(three_points_path, 'w') as f_three_points:
        
        # Iterate over each shape in the JSON file
        for shape in lines.get('shapes', []):
            label = shape.get('label', '')
            points = shape.get('points', [])

            # Divide the points into x and y coordinates (assuming the points list is in pairs)
            contour_points = [(point[0], point[1]) for point in points]

            # Calculate the center, bounding box, and three points
            center_point = get_center_point(contour_points)
            bounding_box = get_bounding_box(contour_points)
            three_points = get_three_point_mediana(bounding_box, contour_points, center_point)

            # Write the results to the respective output files
            f_centers.write(f"{center_point[0]:.6f}, {center_point[1]:.6f}\n")
            f_boxes.write(f"{bounding_box[0]:.6f}, {bounding_box[1]:.6f}, {bounding_box[2]:.6f}, {bounding_box[3]:.6f}\n")
            f_labels.write(f"{label}\n")
            f_three_points.write(
                f"{three_points[0]:.6f}, {three_points[1]:.6f}, {three_points[2]:.6f}, "
                f"{three_points[3]:.6f}, {three_points[4]:.6f}, {three_points[5]:.6f}\n"
            )
