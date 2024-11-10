import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from ..image_handling import center_crop_and_resize, pad_and_resize


class DummyLoader(torch.utils.data.Dataset):
    def __init__(self, img_path, reshape_mode=None, reshaped_size=640):
        super().__init__()

        accepted_reshape_types = [None, 'crop', 'pad']

        assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'
        self.reshape_mode = reshape_mode
        self.reshape_size = reshaped_size
        self.img_path = Path(img_path)
        self.data = self.load_imgs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    def reshape(self, img):
        if self.reshape_mode == 'pad':
            return pad_and_resize(img, self.reshape_size)
        elif self.reshape_mode == 'crop':
            return center_crop_and_resize(img, self.reshape_size)
        else:
            return img

    def load_imgs(self):
        imgs = []

        print(f'loading images form {self.img_path}...')

        for img in sorted(os.listdir(self.img_path)):

            img = np.array(Image.open(self.img_path / img).convert('RGB'))
            imgs.append(self.reshape(img))

        return imgs





