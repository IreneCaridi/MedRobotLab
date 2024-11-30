import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import matplotlib
matplotlib.use('tkagg')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

folder_image = '../image/dataset_mmi/images/test/image_0197.png'
centers_file = '../image/dataset_mmi/points/test/image_0197.txt'
boxes_file = '../image/dataset_mmi/bbox/test/image_0197.txt'
points_file = '../image/dataset_mmi/three_points/test/image_0197.txt'

# Carica l'immagine
image = cv2.imread(folder_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Carica i punti centrali
center_points = []
with open(centers_file, 'r') as f:
    for line in f:
        x, y = map(float, line.strip().split(','))
        center_points.append((x, y))

# Carica i rettangoli
bounding_boxes = []
with open(boxes_file, 'r') as f:
    for line in f:
        min_x, min_y, max_x, max_y = map(float, line.strip().split(','))
        bounding_boxes.append((min_x, min_y, max_x, max_y))

three_points = []
with open(points_file, 'r') as f:
    for line in f:
        x1, y1, x2, y2, x3, y3 = map(float, line.strip().split(','))
        three_points.append((x1, y1))
        three_points.append((x2, y2))
        three_points.append((x3, y3))


# Visualizza l'immagine con i punti e i rettangoli
fig, ax = plt.subplots()
ax.imshow(image)

# Disegna i punti centrali
for (x, y) in center_points:
    ax.plot(x * image.shape[1], y * image.shape[0], 'ro')  # Scala i punti alle dimensioni dell'immagine

# Disegna i tre punti
for (x, y) in three_points:
    ax.plot(x * image.shape[1], y * image.shape[0], 'ro')  # Scala i punti alle dimensioni dell'immagine

# Disegna i rettangoli
for (min_x, min_y, max_x, max_y) in bounding_boxes:
    # Calcola le coordinate scalate in base alla dimensione dell'immagine
    rect = patches.Rectangle(
        (min_x * image.shape[1], min_y * image.shape[0]),  # Posizione iniziale (scala alle dimensioni dell'immagine)
        (max_x - min_x) * image.shape[1],  # Larghezza del rettangolo
        (max_y - min_y) * image.shape[0],  # Altezza del rettangolo
        linewidth=1, edgecolor='b', facecolor='none'
    )
    ax.add_patch(rect)

plt.show()
# plt.savefig("annotation.png")
