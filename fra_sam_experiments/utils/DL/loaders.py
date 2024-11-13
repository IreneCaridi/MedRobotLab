import os
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict

from ..CholectinstanceSeg_utils import get_mask_from_json
from ..mmi_dataset_utils import get_mask_from_txt

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
    def __init__(self, data_path, reshape_mode=None, reshaped_size=640, test_flag=False, use_label=False):
        accepted_reshape_types = [None, 'crop', 'pad']
        assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'
        self.reshape_mode = reshape_mode
        self.reshape_size = reshaped_size

        self.img_path = Path(data_path) / 'images'
        self.lab_path = Path(data_path) / 'labels'

        self.use_label = use_label

        self.test_flag = test_flag
        self.train, self.val, self.test = self.load_imgs()

    def load_imgs(self):
        # fare qualcosa per gestire le labels che NON sono ancora reshapate

        imgs = defaultdict(list)

        for folder in os.listdir(self.img_path):
            # train, valid or test

            print(f'loading images from {self.img_path / folder}...')
            for img in sorted(os.listdir(self.img_path / folder)):

                img = np.array(Image.open(self.img_path / folder / img).convert('RGB'))
                if self.use_label:
                    lab = self.load_masks_labels(img)
                    imgs[folder].append((self.reshape_and_scale(img), lab))
                else:
                    imgs[folder].append(self.reshape_and_scale(img))
            print(f'loaded {len(imgs[folder])} images for {folder}')
        if self.test_flag:
            return imgs['train'], imgs['valid'], imgs['test']
        else:
            return imgs['train'], imgs['valid'], []

    def load_masks_labels(self, img):
        # masks are a list containing N classes tuples like (list of masks, class)
        #   --> [([mask_0 ... mask_n], class_id) ... xN_clsses]

        if '.json' in img:
            masks = get_mask_from_json(self.lab_path / Path(img).with_suffix('.json'))
            # elaborate
            return masks
        elif '.txt' in img:
            masks = get_mask_from_txt(self.lab_path / Path(img).with_suffix('.txt'), return_dict=False)
            # elaborare
            return masks
        else:
            raise ValueError(f'loading from {self.lab_path} not yet implemented...')

    def reshape_and_scale(self, img):
        if self.reshape_mode == 'pad':
            return pad_and_resize(img / 255, self.reshape_size)
        elif self.reshape_mode == 'crop':
            return center_crop_and_resize(img / 255, self.reshape_size)
        else:
            return img / 255




class LoaderFromData(torch.utils.data.Dataset):
    def __init__(self, data, augmentation=None):
        """
            gets the data loaded by the LoadersFromPaths and creates an iterable torch-like dataloader

            args:
                - data list of images as np.array or tuple (img, lab)
                - augmentation: probably I'll create an apposite object
        """
        super().__init__()

        self.data = data
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data[idx], tuple):
            raise ValueError('data as tuple means labels are given and that is not yet implemented...')
        else:
            data = self.data[idx]
            if self.augmentation:
                raise AttributeError('augmentetion not yet implemented :)....')
            data = torch.from_numpy(data.astype('float32')).permute(2, 0, 1)  # to CxHxW

        return data

    def transform(self):
        pass
        # here should make augmentation


def load_all(img_paths, reshape_mode=None, reshaped_size=640, batch_size=4, test_flag=False, use_label=False):
    """

        should load all data in img_paths and returns the dataloaders for train, val and eventually test

    """

    accepted_reshape_types = [None, 'crop', 'pad']
    assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'

    train = []
    val = []
    test = []

    for p in img_paths:
        # loading from all paths and splitting
        loader = LoaderFromPath(p, reshape_mode, reshaped_size, test_flag, use_label)
        train += loader.train
        val += loader.val
        test += loader.test

    train_loader = torch.utils.data.DataLoader(LoaderFromData(train), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(LoaderFromData(val), batch_size=batch_size, shuffle=True)
    if test_flag:
        test_loader = torch.utils.data.DataLoader(LoaderFromData(test), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader




