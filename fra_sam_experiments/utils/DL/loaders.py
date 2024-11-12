import os
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from ..image_handling import center_crop_and_resize, pad_and_resize
from .. import random_state

random_state()


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

    def shuffle(self):
        random.shuffle(self.data)


class LoaderFromPath:
    def __init__(self, img_path, accepted_paths, reshape_mode=None, reshaped_size=640, p_val=0.2, test=False):
        accepted_reshape_types = [None, 'crop', 'pad']
        assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'
        self.reshape_mode = reshape_mode
        self.reshape_size = reshaped_size
        self.img_path = Path(img_path)
        self.data = self.load_imgs()

        self.test_flag = test
        self.train, self.val, self.test = self.split(p_val)

    def load_imgs(self):

        imgs = []
        accepted_folders = []  # here I should put the actual folder

        # select just some folders to load probabily...
        for folder in os.listdir(self.img_path):
            if folder in accepted_folders:
                print(f'loading images form {self.img_path / folder}...')

                for img in sorted(os.listdir(self.img_path)):

                    img = np.array(Image.open(self.img_path / folder / img).convert('RGB'))
                    imgs.append(self.reshape(img))
        return imgs

    def shuffle(self):
        random.shuffle(self.data)

    def split(self, p_val):
        self.shuffle()
        train, test = train_test_split(self.data, p_val, random_state=36)
        if self.test_flag:
            train, val = train_test_split(train, p_val, random_state=36)
            return train, val, test
        else:
            return train, test, None  # here test is actually a validation...




