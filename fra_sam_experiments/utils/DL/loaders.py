import os
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
# from sklearn.model_selection import train_test_split
from collections import defaultdict
import sys

from ..CholectinstanceSeg_utils import get_mask_from_json
from ..mmi_dataset_utils import get_mask_from_txt

from ..image_handling import center_crop_and_resize, pad_and_resize, get_polygon_centroid
from .. import my_logger
from .collates import imgs_masks_polys, from_grid_crop
from .augmentation import get_grid_patches
from ..image_handling import bbox_from_poly, mask_list_to_array, get_three_points

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

        my_logger.info(f'loading images form {self.img_path}...')

        for img in sorted(os.listdir(self.img_path)):

            img = np.array(Image.open(self.img_path / img).convert('RGB'))
            imgs.append(self.reshape(img))

        return imgs

    def shuffle(self):
        random.shuffle(self.data)


class LoaderFromPath:
    def __init__(self, data_path, reshape_mode=None, reshaped_size=640, test_flag=False, use_label=False, store_imgs=False,
                 use_bbox=False):
        """
        gets the data path and loads images or path-to-images split datasets

        args:
            - data_path: path to dataset
            - reshape_mode: how to get square imgs
            - reshaped_size: target size as input to model
            - test_flag: whether to return the test set (it always split for it, simply it is not returned)
            - use_label: whether to load also labels
            - load_imgs: if True it directly loads images to RAM, else it stores paths-to-images
        """
        accepted_reshape_types = [None, 'crop', 'pad', 'grid']
        assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'
        self.reshape_mode = reshape_mode
        self.reshape_size = reshaped_size

        self.img_path = Path(data_path) / 'images'
        self.lab_path = Path(data_path) / 'labels'

        if use_label:
            self.lab_suffix = self.get_lab_type()
        else:
            self.lab_suffix = None

        self.store_imgs = store_imgs
        self.use_bbox = use_bbox

        self.test_flag = test_flag
        self.train, self.val, self.test = self.load_imgs()

    def load_imgs(self):
        # fare qualcosa per gestire le labels che NON sono ancora reshapate

        imgs = defaultdict(list)

        for folder in os.listdir(self.img_path):
            # train, valid or test

            my_logger.info(f'loading images from {self.img_path / folder}...')
            for img_n in sorted(os.listdir(self.img_path / folder)):
                if self.store_imgs:  # loads all dataset to ram
                    img = np.array(Image.open(self.img_path / folder / img_n).convert('RGB'))
                    if self.lab_suffix:
                        lab, poly = self.load_masks_labels(folder, Path(img_n).with_suffix(self.lab_suffix), img.shape[:2])
                        imgs[folder].append((self.reshape_and_scale(img), (lab, poly)))
                    else:
                        imgs[folder].append(self.reshape_and_scale(img))
                else:  # folders to only load batches
                    if self.lab_suffix:
                        lab_path = self.lab_path / folder / Path(img_n).with_suffix(self.lab_suffix)
                        imgs[folder].append((self.img_path / folder / img_n, lab_path))
                    else:
                        imgs[folder].append(self.img_path / folder / img_n)
            my_logger.info(f'loaded {len(imgs[folder])} images for {folder}')
        if self.test_flag:
            return imgs['train'], imgs['valid'], imgs['test']
        else:
            return imgs['train'], imgs['valid'], []

    def get_lab_type(self, folder='train'):  # every folder of a dataset should be same format...
        i = os.listdir(self.lab_path / folder)
        return Path(i[0]).suffix

    def load_masks_labels(self, folder, lab_name, img_shape):
        # masks are a np.array with integer coded masks (fine for torch)
        lab_name = str(lab_name)
        if '.json' in lab_name:
            poly = get_mask_from_json(self.lab_path / folder / lab_name)
            mask = mask_list_to_array(poly, img_shape)
            mask = self.reshape_masks(mask)
            return mask, poly
        elif '.txt' in lab_name:
            poly = get_mask_from_txt(self.lab_path / folder / lab_name, return_dict=False)
            mask = mask_list_to_array(poly, img_shape)
            mask = self.reshape_masks(mask)
            return mask, poly
        else:
            raise TypeError(f'{self.lab_suffix} labels are not accepted... ')

    def reshape_and_scale(self, img):
        if self.reshape_mode == 'pad':
            return pad_and_resize(img / 255, self.reshape_size)
        elif self.reshape_mode == 'crop':
            return center_crop_and_resize(img / 255, self.reshape_size)
        elif self.reshape_mode == 'grid':
            return get_grid_patches(img / 255, self.reshape_size)
        else:
            return img / 255

    def reshape_masks(self, y):
        if self.reshape_mode == 'pad':
            return pad_and_resize(y, self.reshape_size)
        elif self.reshape_mode == 'crop':
            return center_crop_and_resize(y, self.reshape_size)
        elif self.reshape_mode == 'grid':
            return get_grid_patches(y, self.reshape_size)
        else:
            return y


class LoaderFromData(torch.utils.data.Dataset):
    def __init__(self, data, augmentation=None, reshape_mode=None, reshaped_size=640):
        """
            gets the data loaded by the LoadersFromPaths and creates an iterable torch-like dataloader

            args:
                - data: list of images as np.array or tuple (img, lab)
                - augmentation: probably I'll create an apposite object
                - reshape_mode: how to get square imgs
                - reshaped_size: target size as input to model
        """
        super().__init__()

        accepted_reshape_types = [None, 'crop', 'pad', 'grid']
        assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'
        self.reshape_mode = reshape_mode
        self.reshape_size = reshaped_size

        self.data = data

        # check for labels
        if isinstance(self.data[0], tuple):
            i, _ = self.data[0]
            self.with_labels = True
        else:
            i = self.data[0]
            self.with_labels = False

        # check for loaded or not
        if isinstance(i, np.ndarray):
            self.already_loaded = True  # I have loaded images and labels
        else:
            self.already_loaded = False  # I have paths

        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.with_labels:
            if self.already_loaded:
                x, (y, p) = self.data[idx]  # imgs, labs, polys
                return self.transform(x, [y, p])
            else:
                x_p, y_p = self.data[idx]
                x = self.reshape_and_scale(np.array(Image.open(x_p).convert('RGB')))
                if self.reshape_mode == 'grid':
                    out = []
                    y, p = self.load_masks_labels(y_p, (480, 854))
                    for crop, crop_m in zip(x, y):
                        # if np.max(crop_m.ravel()) != 0:
                        out.append(self.transform(crop, [crop_m, p]))
                    return out
                else:
                    y, p = self.load_masks_labels(y_p, x.shape[:2])
                    return self.transform(x, [y, p])
        else:
            if self.already_loaded:  # check if paths or images are given
                x = self.data[idx]
            else:
                x = self.reshape_and_scale(np.array(Image.open(self.data[idx]).convert('RGB')))
            return self.transform(x)

    def transform(self, x, y=None):
        if self.augmentation:
            pass
            # do something for augmentation...
        else:
            x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1)

            return (x, torch.LongTensor(y[0]), y[1]) if y else x

    def reshape_and_scale(self, img):
        if self.reshape_mode == 'pad':
            return pad_and_resize(img / 255, self.reshape_size)
        elif self.reshape_mode == 'crop':
            return center_crop_and_resize(img / 255, self.reshape_size)
        elif self.reshape_mode == 'grid':
            return get_grid_patches(img / 255, self.reshape_size)
        else:
            return img / 255

    def reshape_masks(self, y):
        if self.reshape_mode == 'pad':
            return pad_and_resize(y, self.reshape_size)
        elif self.reshape_mode == 'crop':
            return center_crop_and_resize(y, self.reshape_size)
        elif self.reshape_mode == 'grid':
            return get_grid_patches(y, self.reshape_size)
        else:
            return y

    def load_masks_labels(self, lab_path, img_shape):
        # masks are a np.array with integer coded masks (fine for torch)

        lab_path = str(lab_path)
        if '.json' in lab_path:
            poly = get_mask_from_json(lab_path)
            mask = mask_list_to_array(poly, img_shape)
            mask = self.reshape_masks(mask)
            return mask, poly
        elif '.txt' in lab_path:
            poly = get_mask_from_txt(lab_path, return_dict=False)
            mask = mask_list_to_array(poly, img_shape)
            mask = self.reshape_masks(mask)
            return mask, poly
        else:
            raise TypeError(f'{self.lab_suffix} labels are not accepted... ')


def load_all(img_paths: list, reshape_mode=None, reshaped_size=1024, batch_size=4, test_flag=False, use_label=False,
             n_workers=7, pin_memory=True, store_imgs=False):
    """
        loads all data in img_paths and returns the dataloaders for train, val and eventually test

        args:
            - img_paths: list of paths from which data are loaded
            - reshape_mode: how to extract square imgs
            - reshaped_size: target shape to be fed to model
            - batch_size: self explained
            - test_flag: whether to return test set
            - use_label: whether to use labels
            - n_workers: number of workers for parallel dataloading (rule of thumb: n° cpu core - 1)
            - pin_memory: whether to pin memory for more efficient passage to gpu
        returns:
            - train_loader, val_loader, (optional) test_loader : torch iterable dataloaders
    """

    accepted_reshape_types = [None, 'crop', 'pad', 'grid']
    assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'

    train = []
    val = []
    test = []

    for p in img_paths:
        # loading from all paths and splitting
        loader = LoaderFromPath(p, reshape_mode, reshaped_size, test_flag, use_label, store_imgs=store_imgs)
        train += loader.train
        val += loader.val
        test += loader.test

    if use_label and reshape_mode != 'grid':
        collate_fun = imgs_masks_polys
    elif reshape_mode == 'grid':
        collate_fun = from_grid_crop
    else:
        collate_fun = None

    train_loader = torch.utils.data.DataLoader(LoaderFromData(train, reshape_mode=reshape_mode, reshaped_size=reshaped_size),
                                               batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                               num_workers=n_workers, collate_fn=collate_fun)
    val_loader = torch.utils.data.DataLoader(LoaderFromData(val, reshape_mode=reshape_mode, reshaped_size=reshaped_size),
                                             batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                             num_workers=n_workers, collate_fn=collate_fun)
    if test_flag:
        test_loader = torch.utils.data.DataLoader(LoaderFromData(test, reshape_mode=reshape_mode, reshaped_size=reshaped_size),
                                                  batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                                  num_workers=n_workers, collate_fn=collate_fun)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(str(Path(parent_dir) / 'EdgeSAM'))

from edge_sam.utils.transforms import ResizeLongestSide

class EdgeSAMLoader(torch.utils.data.Dataset):
    def __init__(self, src, lab_type='bbox', device='cpu'):
        super().__init__()

        assert lab_type in ['bbox', 'centroid', 'three_points']

        self.src = Path(src)
        self.src_lab =Path(str(self.src).replace('images', 'labels'))
        self.lab_type = lab_type

        self.device = device

        self.transform = ResizeLongestSide(1024)

        self.data= self.load_data()

    def prepare_image(self, image):
        image = self.transform.apply_image(image)
        image = torch.as_tensor(image, device=self.device)
        return image.permute(2, 0, 1).contiguous()

    def load_data(self):
        data = []
        for i in  os.listdir(self.src):
            img = np.array(Image.open(self.src / i).convert('RGB'))
            lab = get_mask_from_json(self.src_lab / i.replace('.png', '.json'))
            mask = mask_list_to_array(lab, img.shape, True)
            labs = [l for _, l in lab]
            if lab:
                if self.lab_type=='bbox':
                    bbox = bbox_from_poly([lab])
                    xyxy = torch.tensor([b for b, _ in bbox])
                    # print(xyxy.shape)
                    # print(xyxy.squeeze().numpy().tolist())

                    input_dict = {'image': self.prepare_image(img),
                                  'boxes': self.transform.apply_boxes_torch(xyxy, img.shape[:2]),
                                  'original_size': img.shape[:2],
                                  'prompt_init': xyxy.squeeze().numpy().tolist(),
                                  'original_image': img}

                elif self.lab_type=='centroid':
                    centroids_list = get_polygon_centroid(lab)
                    centroids = torch.tensor([[x] for x, y in centroids_list])

                    input_dict = {'image': self.prepare_image(img),
                                  'point_coords': self.transform.apply_coords_torch(centroids, img.shape[:2]),
                                  'point_labels': torch.ones((centroids.shape[0], 1)),
                                  'original_size': img.shape[:2],
                                  'prompt_init': centroids.numpy(),
                                  'original_image': img}

                elif self.lab_type=='three_points':
                    three_points_list = get_three_points(lab, 0.3)
                    points = torch.tensor([x for x, _ in three_points_list])
                    input_dict = {'image': self.prepare_image(img),
                                 'point_coords': self.transform.apply_coords_torch(points, img.shape[:2]),
                                 'point_labels': torch.ones((points.shape[:2])),
                                 'original_size': img.shape[:2],
                                  'prompt_init': points.numpy(),
                                  'original_image': img}

                else:
                    raise NotImplementedError
                data.append((input_dict, (torch.tensor(mask), labs)))
        return data



    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

