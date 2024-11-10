import torch
import sys
import os
from pathlib import Path
from utils.general import check_device
from utils import random_state
from utils.DL.loaders import DummyLoader
from utils.DL.collates import keep_unchanged

random_state()

# placing myself in sam2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(str(Path(parent_dir) / 'sam2'))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



def main():

    img_paths = [Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\dataset_mmi\images\train'),
                   Path(r'D:\poli_onedrive_backup\dataset_mmi_3\dataset_mmi\images\train'),
                   Path(r'D:\poli_onedrive_backup\AtlasDione_30fps'),
                   Path(r'D:\poli_onedrive_backup\CholectInstanceSeg\train')]
    names = ['mmi_2_train', 'mmi_3_train', 'AtlasDione_30fps', 'CholectInstanceSeg_train']

    dst = Path(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\cropped_640')

    for src, name in zip(img_paths, names):

        # setting up sam2
        device = check_device()

        sam2_checkpoint = Path(parent_dir) / r'sam2\checkpoints\sam2.1_hiera_large.pt'
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        bs = 4

        embd_dict = {}

        if name in names[3:]:
            for folder in os.listdir(src):

                images_names = [str(src / folder / x) for x in sorted(os.listdir(src / folder))]

                loader = torch.utils.data.DataLoader(DummyLoader(src / folder, 'crop'), batch_size=bs, shuffle=False,
                                                     collate_fn=keep_unchanged)

                for b, data in enumerate(loader):

                    print(f'{b+1}/{len(loader)}')

                    predictor.set_image_batch(data)
                    emb = predictor.get_image_embedding()
                    emb = emb.cpu()

                    for i in range(emb.size()[0]):
                        embd_dict[images_names[b*bs + i]] = emb[i, :, :, :]
        else:
            images_names = [str(src / x) for x in sorted(os.listdir(src))]

            loader = torch.utils.data.DataLoader(DummyLoader(src, 'crop'), batch_size=bs, shuffle=False,
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
