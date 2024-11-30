# import useful libraries
import os
import sys
from pathlib import Path

# Set path to the sam2 folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(Path(parent_dir) / 'sam2')

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from utils.general import check_device
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


# Select the device for computation
device = check_device()
#device = torch.device("cpu")


if __name__ == '__main__':
    iou_path = "../image/dataset_mmi/IoU/iou_mask_three_pts.csv"
    df_iou = pd.read_csv(iou_path)
    print(df_iou.head())

    # Plot boxplot for IoU per image
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='class', y='iou', data=df_iou, palette='Set2')
    plt.title('IoU per Image')
    plt.xlabel('Image')
    plt.ylabel('IoU')
    plt.savefig("iou_bplot_three_pts_mask.png")
    
    # Plot histogram of IoU values
    plt.figure(figsize=(10, 6))
    sns.histplot(df_iou['iou'], bins=10, kde=True, color='purple')
    plt.title('IoU Distribution Across All Classes and Images')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.savefig("iou_hist_three_pts_mask.png")

    # Pivot table for heatmap, with images as rows and classes as columns
    iou_pivot = df_iou.pivot_table(index='image', columns='class', values='iou', aggfunc='mean')

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(iou_pivot, annot=False, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title('IoU per Class and Image')
    plt.xlabel('Class')
    plt.ylabel('Image')
    plt.savefig("iou_hmap_three_pts_mask.png")