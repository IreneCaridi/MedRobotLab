import argparse
import os

import torch
from pathlib import Path


import wandb

from models import DistillatioModels, check_load_model, SAM2handler, ModelClass
from models.common import Dummy, UnetEncoder, UNet0
from models.Rep_ViT import RepViT, RepViTEncDec, RepViTUnet
from models.detection import RepViTDetYolox, RepViTDetCenterNet
from utils.DL.callbacks import Callbacks, EarlyStopping, Saver
from utils.DL.loaders import load_all
from utils.DL.optimizers import get_optimizer, scheduler
from utils.DL.losses import SemanticLosses, FullLossKD, MSELoss, MmdetLossYolox, MmdetLossCenterNet
from utils.DL.metrics import Metrics
from utils.DL.logger import Loggers
from utils import random_state, increment_path, json_from_parser, my_logger

# setting all random states
random_state(36)

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
    p = Path('runs') / 'train'
    if folder:
        p = p / folder
    os.makedirs(p, exist_ok=True)
    save_path = increment_path(p, name)
    name = save_path.stem
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    out_classes = args.n_class + 1  # +1 for bkg

    if args.weighted_loss:
        weights = [0.00024, 0.00662, 0.67449, 0.07547, 0.06557, 0.11787, 0.05362, 0.00608]
        weights = torch.tensor(weights, dtype=torch.float32)

        if torch.cuda.is_available() and args.device == 'gpu':
            weights = weights.to('cuda:0')
    else:
        weights = None

    if args.as_encoder:
        enc_flag = True
        use_label_flag = False
        use_box = False
        metrics_mode = 'null'
        loss_fn = MSELoss(reduction="mean")
    elif args.as_predictor:
        enc_flag = False
        use_label_flag = True
        use_box = False
        metrics_mode = 'seg'
        loss_fn = FullLossKD(lamda_dist=0.25, alpha=1, gamma=2, lambdas_focal=(0.5, 0.5), weights=weights)
    elif args.only_supervised:
        enc_flag = False
        use_label_flag = True
        use_box = False
        metrics_mode = 'seg'
        loss_fn = SemanticLosses(alpha=1, gamma=2, lambdas=(0.5, 0.5), weight=weights)  # maybe consider weights...
    elif args.rpn_training:
        enc_flag = False
        use_label_flag = True
        use_box = True
        metrics_mode = 'bbox'
    else:
        raise AttributeError('ok it should not be possible to get there, you broke the parser lol')

    # saving inputs
    json_from_parser(args, save_path)

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
    loaders = load_all(data_paths, args.reshape_mode, args.reshape_size, batch_size, test_flag=False,
                       use_label=use_label_flag, n_workers=args.n_workers, pin_memory=args.pin_memory,
                       store_imgs=args.store_imgs, use_bbox=use_box)

    # model (ADJUST)
    if "." not in args.student:
        # means it is not a weight and has to be imported ADJUST => (NEED TO IMPORT IT)
        if args.student == "Dummy":
            student = Dummy()
        elif args.student == 'UnetEncoder':
            student = UnetEncoder()
            # loading model = bla bla bla
        elif args.student == 'RepViT':
            student = RepViT(args.arch, args.reshape_size, fuse=True)
        elif args.student == 'RepViTEncDec':
            student = RepViTEncDec(args.arch, out_classes, fuse=True)
        elif args.student == 'Unet':
            student = UNet0(out_classes)
        elif args.student == 'RepViTUnet':
            student = RepViTUnet(args.arch, out_classes, fuse=True)
        elif args.student == 'RepViTDetYolox':
            student = RepViTDetYolox(args.arch, out_classes, fuse=True)
            loss_fn = MmdetLossYolox(reduction='mean', cls_lambda=1., bbox_lambda=1., obj_lambda=1.)
        elif args.student == 'RepViTDetCenterNet':
            student = RepViTDetCenterNet(args.arch, out_classes, fuse=True)
            loss_fn = MmdetLossCenterNet(reduction='mean', cls_lambda=1e-4, bbox_lambda=1.)
        else:
            raise TypeError("Model name not recognised")
    else:
        # it is a weight
        student = args.student

    # double-checking whether you parsed weights or model and accounting for transfer learning
    student = check_load_model(student, args.pre_weights, my_logger)

    # initializing callbacks ( could be handled more concisely i guess...)
    stopper = EarlyStopping(patience=args.patience, monitor="val_loss", mode="min")
    saver = Saver(model=student, save_best=True, save_path=save_path, monitor="val_loss", mode='min')
    callbacks = Callbacks([stopper, saver])

    metrics = Metrics(loss_fn=loss_fn, mode=metrics_mode, num_classes=out_classes, device=device, top_k=1, thresh=0.5)

    opt = get_optimizer(student, args.opt, args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)

    # initializing loggers
    logger = Loggers(metrics=metrics, save_path=save_path, opt=opt, test=False, wandb=bool(wandb_name))

    # lr scheduler
    sched = scheduler(opt, args.sched, args.lrf, epochs)

    # building model
    if args.only_supervised or args.rpn_training:
        model = ModelClass(student, loaders, info_log=my_logger, loss_fn=loss_fn, device=device, AMP=args.AMP,
                           optimizer=opt, metrics=metrics, loggers=logger, callbacks=callbacks, sched=sched,
                           freeze=args.freeze_backbone, is_det=use_box)
    else:
        # loading sam as teacher (SAM2_image_predictor instance -- SAM2(nn.module) is in teacher.model)
        teacher = SAM2handler(Path(parent_dir) / args.SAM2_weights, args.SAM2_configs, info_log=my_logger,
                              as_encoder=enc_flag)
        model = DistillatioModels(student, teacher, loaders, info_log=my_logger, loss_fn=loss_fn, device=device, AMP=args.AMP,
                                  optimizer=opt, metrics=metrics, loggers=logger, callbacks=callbacks, sched=sched,
                                  as_encoder=enc_flag)

    # initializing wandb
    if wandb_name:
        wandb.login()
        wandb.init(project='MedRobLab_prj', name=name, entity=wandb_name, config=args)

    # training the model
    model.train_loop(epochs)

    # finisching wandb
    if wandb_name:
        wandb.finish()

if __name__ == "__main__":

    # list of arguments (ADJUST for student and SAM)
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--student', type=str, required=True, help='name of model to train or path to weights to train')
    parser.add_argument('--SAM2_weights', type=str, default= r'sam2\checkpoints\sam2.1_hiera_large.pt', help='path to SAM2 weights (from sam2 repo folder)')
    parser.add_argument('--SAM2_configs', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help='path to SAM2 configs (from sam2 repo folder)')
    parser.add_argument('--pre_weights', type=str, default=None, help='path to backbone weights, if present it ONLY loads weights for it')
    parser.add_argument('--freeze_backbone', action='store_true', help='wether to freeze backbone')

    # RepViT ONLY!!!!!!!!!!!!!!!!
    parser.add_argument('--arch', type=str, default='m1', help='the architecture type of RepViT')

    # classes (excluding bkg)
    parser.add_argument('--n_class', type=int, default=7, help='the number of classes to segment (excluding bkg)')

    # reshaping BOTH needed (when grid consider 8 elment per batch if size == 256)
    parser.add_argument('--reshape_mode', type=str, default='crop', choices=[None, 'crop', 'pad', 'grid'], help=" how to handle resize")
    parser.add_argument('--reshape_size', type=int, default=1024, help='the finel shape input to model')

    # loggers option
    parser.add_argument('--wandb', type=str, default=None, help='name of wandb profile (if None means not logging) (MedRobLab)')

    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--folder', type=str, default=None, help='name of folder to which saving results inside runs/train')
    parser.add_argument('--name', type=str, default="exp", help='name of experiment folder inside folder')
    parser.add_argument('--opt', type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"], help='name of optimizer to use')
    parser.add_argument('--sched', type=str, default=None, choices=["linear", "cos_lr"], help="name of the lr scheduler")
    parser.add_argument('--lr0', type=float, default=0.0004, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.001, help='final learning rate (multiplicative factor)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value (SGD) beta1 (Adam, AdamW)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay value')
    parser.add_argument('--lab_smooth', type=float, default=0, help='label smoothing value')
    parser.add_argument('--patience', type=int, default=30, help='number of epoch to wait for early stopping')
    parser.add_argument('--device', type=str, default="gpu", choices=["cpu", "gpu"], help='device to which loading the model')
    parser.add_argument('--AMP', action="store_true", help='whether to use AMP')
    # probably not userfull
    parser.add_argument('--weighted_loss', action="store_true", help='whether to weight the loss and weight for classes')

    # datasets
    parser.add_argument('--data_path', type=str, required=True, help='path to dataset (containing the ones below)')
    # at least one is mandatory
    parser.add_argument('--MMI', action="store_true", help='whether to use MMI dataset')
    parser.add_argument('--Cholect', action="store_true", help='whether to use Cholect dataset')
    parser.add_argument('--AtlasDione', action="store_true", help='whether to use AtlasDione dataset')
    parser.add_argument('--Kvasir', action="store_true", help='whether to use Kvasir dataset')
    parser.add_argument('--prova', action="store_true", help='whether to use the 3 img dataset for debugging')


    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--as_encoder', action="store_true", help='whether to do encoder KD')
    group1.add_argument('--as_predictor', action="store_true", help='whether to do masks KD')
    group1.add_argument('--only_supervised', action="store_true", help='whether to train with just gt (no KD)')
    group1.add_argument('--rpn_training', action="store_true", help='whether to train an object detector')

    # loaders params
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for parallel dataloading ')
    parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory for more efficient passage to gpu')
    parser.add_argument('--store_imgs', action='store_true', help='wether to store all data')

    args = parser.parse_args()

    main(args)


