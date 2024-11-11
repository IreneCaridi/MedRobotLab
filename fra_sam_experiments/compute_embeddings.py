import torch
import sys
import os
from pathlib import Path
from utils.general import check_device
from utils import random_state
from utils.DL.loaders import DummyLoader
from utils.DL.collates import keep_unchanged
import stat

random_state()

# placing myself in sam2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(str(Path(parent_dir) / 'sam2'))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



def main():

    img_paths = [Path(r'D:\poli_onedrive_backup\CholectInstanceSeg\train'),
                 Path(r'D:\poli_onedrive_backup\AtlasDione_30fps')]
    names = ['CholectInstanceSeg_train', 'AtlasDione_30fps']

    dst = Path(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\cropped_640')

    # setting up sam2
    device = check_device()

    sam2_checkpoint = Path(parent_dir) / r'sam2\checkpoints\sam2.1_hiera_large.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    for src, name in zip(img_paths, names):

        bs = 4

        embd_dict = {}

        if name == 'AtlasDione_30fps':
            for folder in os.listdir(src):
                embd_dict = {}
                images_names = [str(src / folder / x) for x in sorted(os.listdir(src / folder))]
                dataloader = DummyLoader(src / folder, 'crop')
                loader = torch.utils.data.DataLoader(dataloader, batch_size=bs, shuffle=False,
                                                     collate_fn=keep_unchanged)

                for b, data in enumerate(loader):

                    print(f'{b+1}/{len(loader)}')

                    predictor.set_image_batch(data)
                    emb = predictor.get_image_embedding()
                    emb = emb.cpu()

                    for i in range(emb.size()[0]):
                        embd_dict[images_names[b*bs + i]] = emb[i, :, :, :]
                torch.save(embd_dict, dst / f'{name}_{folder}.pth')
        elif name == 'CholectInstanceSeg_train':
            for folder in os.listdir(src):
                embd_dict = {}
                if 'seg' not in folder:
                    images_names = [str(src / folder / 'img_dir' /x) for x in sorted(os.listdir(src / folder / 'img_dir'))]
                    dataloader = DummyLoader(src / folder / 'img_dir', 'crop')
                    loader = torch.utils.data.DataLoader(dataloader, batch_size=bs, shuffle=False,
                                                         collate_fn=keep_unchanged)

                    for b, data in enumerate(loader):

                        print(f'{b + 1}/{len(loader)}')

                        predictor.set_image_batch(data)
                        emb = predictor.get_image_embedding()
                        emb = emb.cpu()

                        for i in range(emb.size()[0]):
                            embd_dict[images_names[b * bs + i]] = emb[i, :, :, :]
                    torch.save(embd_dict, dst / f'{name}_{folder}.pth')
        else:
            images_names = [str(src / x) for x in sorted(os.listdir(src))]

            dataloader = DummyLoader(src, 'crop')
            loader = torch.utils.data.DataLoader(dataloader, batch_size=bs, shuffle=False,
                                                 collate_fn=keep_unchanged)

            for b, data in enumerate(loader):

                print(f'{b + 1}/{len(loader)}')

                predictor.set_image_batch(data)
                emb = predictor.get_image_embedding()
                emb = emb.cpu()

                for i in range(emb.size()[0]):
                    embd_dict[images_names[b * bs + i]] = emb[i, :, :, :]

            torch.save(embd_dict, dst / f'{name}.pth')


if __name__ == "__main__":
    main()
