import argparse
import os

import torch
from pathlib import Path

from tqdm import tqdm


import wandb


from models import check_load_model, SAM2handler, ModelTest
from models.common import Dummy, UnetEncoder, UNet0
from models.Rep_ViT import RepViT, RepViTEncDec, RepViTUnet
from models.detection import RepViTDetYolox, RepViTDetCenterNet
from utils.DL.loaders import load_all
from utils.DL.metrics import Metrics
from utils.DL.logger import Loggers
from utils import random_state, increment_path, my_logger

# setting all random states
random_state(36)

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def main(args):

    if args.MMI:
        raise AttributeError('eh eh eh MMI dataset is not available uuuhhhhhh (do not flag args.MMI)')


    #checking wandb
    if args.wandb == 'None' or args.wandb == 'none' or args.wandb == 'False' or args.wandb == 'false':
        wandb_name = None
    else:
        wandb_name = args.wandb

    # unpacking
    folder = args.folder
    name = args.name

    # creating saving location
    p = Path('runs') / 'test'
    if folder:
        p = p / folder
    os.makedirs(p, exist_ok=True)
    save_path = increment_path(p, name)
    name = save_path.stem

    batch_size = args.batch_size

    if torch.cuda.is_available() and args.device == 'gpu':
        device = 'gpu'
    else:
        device = 'cpu'


    out_classes = args.n_class + 1  # +1 for bkg

    # checking for dataset
    if not args.MMI and not args.Cholect and not args.AtlasDione and not args.Kvasir and not args.prova:
        raise AttributeError('at least one of --MMI, --Cholect, --AtlasDione, --Kvasir arguments must be provided...')
    else:

        # if we get MMI data put them...
        dataset_path = Path(args.data_path)
        data_paths = [dataset_path / p for f, p in zip([args.Cholect, args.AtlasDione, args.Kvasir, args.prova],
                                                       ['Cholect_dataset', 'AtlasDione_dataset', 'kvasir_dataset', 'prova_dataset']) if f]

    assert args.reshape_mode and args.reshape_size, 'both --reshape_size and --reshape_mode are needed together...'

    # loading dataset already as iterable torch loaders (train, val ,(optional) test)
    loader = load_all(data_paths, args.reshape_mode, args.reshape_size, batch_size, test_flag=True,
                       use_label=True, n_workers=args.n_workers, pin_memory=args.pin_memory,
                       store_imgs=args.store_imgs, use_bbox=True)

    # # model (ADJUST)
    # if "." not in args.student:
    #     # means it is not a weight and has to be imported ADJUST => (NEED TO IMPORT IT)
    #     if args.student == "Dummy":
    #         student = Dummy()
    #     elif args.student == 'UnetEncoder':
    #         student = UnetEncoder()
    #         # loading model = bla bla bla
    #     elif args.student == 'RepViT':
    #         student = RepViT(args.arch, args.reshape_size, fuse=True)
    #     elif args.student == 'RepViTEncDec':
    #         student = RepViTEncDec(args.arch, out_classes, fuse=True)
    #     elif args.student == 'Unet':
    #         student = UNet0(out_classes)
    #     elif args.student == 'RepViTUnet':
    #         student = RepViTUnet(args.arch, out_classes, fuse=True)
    #     elif args.student == 'RepViTDetYolox':
    #         student = RepViTDetYolox(args.arch, out_classes, fuse=True)
    #         loss_fn = MmdetLossYolox(reduction='mean', cls_lambda=1., bbox_lambda=1., obj_lambda=1.)
    #     elif args.student == 'RepViTDetCenterNet':
    #         student = RepViTDetCenterNet(args.arch, out_classes, fuse=True)
    #         loss_fn = MmdetLossCenterNet(reduction='mean', cls_lambda=1e-4, bbox_lambda=1.)
    #     else:
    #         raise TypeError("Model name not recognised")
    # else:
    #     # it is a weight
    #     student = args.student

    metrics = Metrics(loss_fn=None, mode='seg', num_classes=out_classes, device=device, top_k=1, thresh=0.5)

    # initializing loggers
    logger = Loggers(metrics=metrics, save_path=save_path, opt=None, test=True, wandb=bool(wandb_name))

    # double-checking whether you parsed weights or model and accounting for transfer learning
    model = check_load_model(args.model_weights, None, my_logger)
    model = ModelTest(model, loader, info_log=my_logger, device=device, metrics=metrics, loggers=logger)

    model.test_loop()

    # initializing wandb
    if wandb_name:
        wandb.login()
        wandb.init(project='MedRobLab_prj', name=name, entity=wandb_name, config=args)

    # finisching wandb
    if wandb_name:
        wandb.finish()

if __name__ == "__main__":

    # list of arguments (ADJUST for student and SAM)
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--model_weights', type=str, required=True, help='path to model weights to test')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')

    # classes (excluding bkg)
    parser.add_argument('--n_class', type=int, default=7, help='the number of classes to segment (excluding bkg)')

    # reshaping BOTH needed (when grid consider 8 elment per batch if size == 256)
    parser.add_argument('--reshape_mode', type=str, default='crop', choices=[None, 'crop', 'pad', 'grid'], help=" how to handle resize")
    parser.add_argument('--reshape_size', type=int, default=1024, help='the finel shape input to model')

    # loggers option
    parser.add_argument('--wandb', type=str, default=None, help='name of wandb profile (if None means not logging) (MedRobLab)')

    parser.add_argument('--folder', type=str, default=None, help='name of folder to which saving results inside runs/train')
    parser.add_argument('--name', type=str, default="exp", help='name of experiment folder inside folder')
    parser.add_argument('--device', type=str, default="gpu", choices=["cpu", "gpu"], help='device to which loading the model')

    # datasets
    parser.add_argument('--data_path', type=str, required=True, help='path to dataset (containing the ones below)')
    # at least one is mandatory
    parser.add_argument('--MMI', action="store_true", help='whether to use MMI dataset')
    parser.add_argument('--Cholect', action="store_true", help='whether to use Cholect dataset')
    parser.add_argument('--AtlasDione', action="store_true", help='whether to use AtlasDione dataset')
    parser.add_argument('--Kvasir', action="store_true", help='whether to use Kvasir dataset')
    parser.add_argument('--prova', action="store_true", help='whether to use the 3 img dataset for debugging')

    # loaders params
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for parallel dataloading ')
    parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory for more efficient passage to gpu')
    parser.add_argument('--store_imgs', action='store_true', help='whether to store all data')

    args = parser.parse_args()

    main(args)


