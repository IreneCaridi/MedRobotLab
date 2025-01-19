import numpy as np
import os
import sys
import torch
from pathlib import Path
from PIL import Image

from ..CholectinstanceSeg_utils import get_mask_from_json
from ..mmi_dataset_utils import get_mask_from_txt

from ..image_handling import (center_crop_and_resize, pad_and_resize, get_polygon_centroid, bbox_from_poly,
                              mask_list_to_array, get_three_points)
from .. import my_logger



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
        for i in  sorted(os.listdir(self.src)):
            img = np.array(Image.open(self.src / i).convert('RGB'))
            lab = get_mask_from_json(self.src_lab / i.replace(str(Path(i).suffix), '.json'))
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
                                  'original_image': img,
                                  'image_name': i
                                  }

                elif self.lab_type=='centroid':
                    centroids_list = get_polygon_centroid(lab)
                    centroids = torch.tensor([[x] for x, y in centroids_list])

                    input_dict = {'image': self.prepare_image(img),
                                  'point_coords': self.transform.apply_coords_torch(centroids, img.shape[:2]),
                                  'point_labels': torch.ones((centroids.shape[0], 1)),
                                  'original_size': img.shape[:2],
                                  'prompt_init': centroids.numpy(),
                                  'original_image': img,
                                  'image_name': i
                                  }

                elif self.lab_type=='three_points':
                    three_points_list = get_three_points(lab, 0.3)
                    points = torch.tensor([x for x, _ in three_points_list])
                    input_dict = {'image': self.prepare_image(img),
                                 'point_coords': self.transform.apply_coords_torch(points, img.shape[:2]),
                                 'point_labels': torch.ones((points.shape[:2])),
                                 'original_size': img.shape[:2],
                                  'prompt_init': points.numpy(),
                                  'original_image': img,
                                  'image_name': i
                                  }

                else:
                    raise NotImplementedError
                data.append((input_dict, (torch.tensor(mask), labs)))
        return data



    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

