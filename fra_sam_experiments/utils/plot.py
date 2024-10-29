import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
from . import random_state

random_state()

def show_mask(mask, ax, random_color=False, borders = True):
    """
    inputs:
          -mask: mask as (h, w, 1)
          -ax: axes object into which doing the plot
          -random_color: whether to randomly color all masks
          -borders: bool whether to draw borders on masks or not
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask.reshape(h, w, 1),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(points, labels, ax, marker_size=30):

    for p, l in zip(points, labels):
        for i in range(len(l)):
            if l[i] != 0:
                ax.scatter(p[i, 0], p[i, 1], color='green', marker='o', s=marker_size,
                           edgecolor='white', linewidth=0.25)
            else:
                ax.scatter(p[i, 0], p[i, 1], color='red', marker='o', s=marker_size,
                           edgecolor='white', linewidth=0.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, img_name=None):
    """
    inputs:
          -image: RGB image as (w, h, 3)
          -masks: mask image  as (1, w, h)
          -scores: scores list
          -point_coord: points as (n° points, 1, 2)
          -box_coords: coordinates of bboxes
          -input_labels: labels as (n° points, 1)
          -borders: bool whether to draw borders on masks or not
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca(), random_color=True, borders=borders)
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
    if box_coords is not None:
        # boxes
        show_box(box_coords, plt.gca())

    plt.title(f'{img_name}: green points are foreground')
    plt.axis('off')
    plt.show()


def show_batch(img_batch, labels_batch, masks_batch, points_batch, scores_batch, img_name=None):
    """
    inputs:
          -img_batch: img as list of RGB images (BS, w, h, 3)
          -masks_batch: masks batch as output of 'model.predict_batch' (n° prompts, 1, h, w)
          -label_batch: labels as list of labels per point in image (BS, n° prompts, 1)
          -points_batch: points as list of points per image (BS, n° prompts, 1, 2)
          -scores_batch: scores as output of 'model.predict_batch'
    """
    for i, (image, labels, masks, points, scores) in enumerate(zip(img_batch, labels_batch, masks_batch, points_batch, scores_batch)):

        show_masks(image, masks, scores, points, input_labels=labels, img_name=img_name[i])


def predict_and_plot(model, img_batch, points_batch, label_batch, img_name=None):
    """
    inputs:
          -model: loaded sam2 model
          -img_batch: img as list of RGB images (BS, w, h, 3)
          -points_batch: points as list of points per image (BS, n° points, 1, 2)
          -label_batch: labels as list of labels per point in image (BS, n° points, 1)
    return:
          -prints the best mask per prompt of every input image
    """

    # passing the batch of images
    model.set_image_batch(img_batch)

    # predicting on the batch
    masks_batch, scores_batch, _ = model.predict_batch(points_batch, label_batch, multimask_output=False)

    # Plotting
    show_batch(img_batch, label_batch, masks_batch, points_batch, scores_batch, img_name=img_name)

