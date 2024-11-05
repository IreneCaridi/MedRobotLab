import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image


class DummyLoader(torch.utils.data.Dataset):
    def __init__(self, img_path):
        super().__init__()

        self.img_path = Path(img_path)
        self.data = self.load_imgs()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    def load_imgs(self):
        imgs = []

        print(f'loading images form {self.img_path}...')

        for img in sorted(os.listdir(self.img_path)):

            img = np.array(Image.open(self.img_path / img).convert('RGB'))
            imgs.append(img)
        return imgs





