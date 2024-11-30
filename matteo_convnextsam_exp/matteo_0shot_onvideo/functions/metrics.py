from sklearn.metrics import jaccard_score
import numpy as np
from PIL import Image
import os
import tqdm


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

def iou_onfoldernpz(pred_folder, target_folder):
    iou_values = []
    # Ottieni il numero totale di confronti
    total_files = len(os.listdir(target_folder))
    # Barra di avanzamento con tqdm
    with tqdm.tqdm(total=total_files, desc="Calcolo IoU", unit="file") as pbar:
        for target in os.listdir(target_folder):
            index_target = int(target.split("_")[1].split(".")[0])
            for prediction in os.listdir(pred_folder):
                index_prediction = int(prediction.split("_")[1].split(".")[0])
                if index_target == index_prediction:
                    tar_npy = np.load(os.path.join(target_folder, target))
                    pred_npy = np.load(os.path.join(pred_folder, prediction))
                    tar_flat = tar_npy["arr_0"].flatten()
                    pred_flat = pred_npy["arr_0"].flatten()
                    iou = jaccard_score(tar_flat, pred_flat)
                    iou_values.append(iou)
                    break  # Passa al prossimo target una volta trovata la corrispondenza
            pbar.update(1)  # Aggiorna la barra di avanzamento
    return iou_values