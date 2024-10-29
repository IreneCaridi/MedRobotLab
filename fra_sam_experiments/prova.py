import torch
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from PIL import Image
from pathlib import Path

a = np.array([[1, 3], [1, 4]])
print(a.shape)

a = a.reshape(1,-1,-1)
print(a)
print(a.shape)