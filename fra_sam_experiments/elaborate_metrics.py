import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

src = Path('data/three_points_03')

iou_dict = {}

for n in os.listdir(src):
    name = n.split('_')[0]
    iou_dict[name] = np.load(src / n)

names = list(iou_dict.keys())
values = list(iou_dict.values())

plt.boxplot(values, tick_labels=names)
plt.show()