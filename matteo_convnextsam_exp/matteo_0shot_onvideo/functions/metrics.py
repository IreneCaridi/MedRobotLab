from sklearn.metrics import jaccard_score
import numpy as np
from PIL import Image


def iou_onimages(pred_img, target_img, num_classes):
    # Flatten the 2D images into 1D arrays
    pred = Image.open(pred_img)  # Sostituisci con il percorso dell'immagine
    target = Image.open(target_img)

    # Converti l'immagine in array NumPy
    pred_array = np.array(pred)
    target_array = np.array(target)

    pred_flat = pred_array.flatten()
    target_flat = target_array.flatten()

    # Calculate IoU per class
    ious = jaccard_score(target_flat, pred_flat, average=None, labels=list(range(num_classes)))
    print("IoU per class:", ious)