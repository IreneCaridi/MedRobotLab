import argparse
import os

import torch.nn as nn
from pathlib import Path

import wandb

from models import DistillatioModels, check_load_model, SAM2handler, ModelClass
from models.common import Dummy, UNetEncoderTrain
from models.Rep_ViT import RepViT
from utils.DL.callbacks import Callbacks, EarlyStopping, Saver
from utils.DL.loaders import load_all
from utils.DL.optimizers import get_optimizer, scheduler
from utils.DL.losses import SemanticLosses, FullLossKD, MSELoss
from utils.DL.metrics import Metrics
from utils.DL.logger import Loggers
from utils import random_state, increment_path, json_from_parser, my_logger

# setting all random states
random_state(36)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def main(args):

    #checking wandb
    if args.wandb == 'None' or args.wandb == 'none' or args.wandb == 'False' or args.wandb == 'false':
        wandb_name = None
    else:
        wandb_name = args.wandb

    # unpacking
    folder = args.folder
    name = args.name

    # creating saving location
    p = Path(folder) / 'train'
    os.makedirs(p, exist_ok=True)
    save_path = increment_path(p, name)
    name = save_path.stem
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    # out_classes = args.n_class + 1  # +1 for bkg

    if args.as_encoder:
        enc_flag = True
        loss_fn = MSELoss(reduction="mean")
    elif args.as_predictor:
        enc_flag = False
        loss_fn = FullLossKD(lamda_dist=0.25, alpha=1, gamma=2, lambdas_focal=(0.5, 0.5), weights=None)
    elif args.only_supervised:
        enc_flag = True
        loss_fn = SemanticLosses(alpha=1, gamma=2, lambdas=(0.5, 0.5), weight=None)  # maybe consider weights...
    else:
        raise AttributeError('ok it should not be possible to get there, you broke the parser lol')

    # saving inputs
    json_from_parser(args, save_path)

    # checking for dataset
    if not args.MMI and not args.Cholect and not args.AtlasDione:
        raise AttributeError('at least one of --MMI, --Cholect, --AtlasDione dataset arguments must be provided...')
    else:
        data_paths = [p for p in [args.MMI, args.Cholect, args.AtlasDione] if p is not None]

    assert args.reshape_mode and args.reshape_size, 'both --reshape_size and --reshape_mode are needed together...'

    # loading dataset already as iterable torch loaders (train, val ,(optional) test)
    loaders = load_all(data_paths, args.reshape_mode, args.reshape_size, batch_size, test_flag=False,
                       use_label=not enc_flag, n_workers=args.n_workers, pin_memory=args.pin_memory)

    # model (ADJUST)
    if "." not in args.student:
        # means it is not a weight and has to be imported ADJUST => (NEED TO IMPORT IT)
        if args.student == "Dummy":
            student = Dummy()
        elif args.student == 'UNetEncoderTrain':
            student = UNetEncoderTrain()
            # loading model = bla bla bla
        elif args.student == 'RepViT':
            student = RepViT('m1', args.reshape_size, fuse=True)
        else:
            raise TypeError("Model name not recognised")
    else:
        # it is a weight
        student = args.student

    # double-checking whether you parsed weights or model and accounting for transfer learning
    student = check_load_model(student, args.backbone)

    # initializing callbacks ( could be handled more concisely i guess...)
    stopper = EarlyStopping(patience=args.patience, monitor="val_loss", mode="min")
    saver = Saver(model=student, save_best=True, save_path=save_path, monitor="val_loss", mode='min')
    callbacks = Callbacks([stopper, saver])

    # for encoder only it is just empty, ADJUST for decoder then
    metrics = Metrics(loss_fn=loss_fn, num_classes=3, device=device, top_k=1, thresh=0.5)

    # if args.weighted_loss:
    # if args.cropped_seq or args.cropped_seq_raw:
    #     weights = torch.tensor([0.62963445, 2.42849968], dtype=torch.float32)  # only m9
    # else:
    #     weights = None

    opt = get_optimizer(student, args.opt, args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)

    # initializing loggers
    logger = Loggers(metrics=metrics, save_path=save_path, opt=opt, test=False, wandb=bool(wandb_name))

    # lr scheduler
    sched = scheduler(opt, args.sched, args.lrf, epochs)

    # building loaders
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padding_x)  # to pad
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad

    # loading sam as teacher (SAM2_image_predictor instance -- SAM2(nn.module) is in teacher.model)
    teacher = SAM2handler(Path(parent_dir) / args.SAM2_weights, args.SAM2_configs, info_log=my_logger,
                          as_encoder=enc_flag)

    # building model
    if args.only_supervised:
        model = ModelClass(student, loaders, info_log=my_logger, loss_fn=loss_fn, device=device, AMP=args.AMP,
                           optimizer=opt, metrics=metrics, loggers=logger, callbacks=callbacks, sched=sched, freeze=None)
    else:
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
    parser.add_argument('--backbone', type=str, default=None, help='path to backbone weights, if present it ONLY loads weights for it')

    # classes (excluding bkg)
    parser.add_argument('--n_class', type=int, default=3, help='the number of classes to segment (excluding bkg)')

    # reshaping BOTH needed
    parser.add_argument('--reshape_mode', type=str, default='crop', choices=[None, 'crop', 'pad'], help=" how to handle resize")
    parser.add_argument('--reshape_size', type=int, default=512, help='the finel shape input to model')

    # loggers option
    parser.add_argument('--wandb', type=str, default='MedRobLab', help='name of wandb profile (if None means not logging)')

    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--folder', type=str, default="runs", help='name of folder to which saving results')
    parser.add_argument('--name', type=str, default="exp", help='name of experiment folder inside folder')
    parser.add_argument('--opt', type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"], help='name of optimizer to use')
    parser.add_argument('--sched', type=str, default=None, choices=["linear", "cos_lr"], help="name of the lr scheduler")
    parser.add_argument('--lr0', type=float, default=0.0004, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.001, help='final learning rate (multiplicative factor)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value (SGD) beta1 (Adam, AdamW)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay value')
    parser.add_argument('--lab_smooth', type=float, default=0, help='label smoothing value')
    parser.add_argument('--patience', type=int, default=30, help='number of epoch to wait for early stopping')
    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"], help='device to which loading the model')
    parser.add_argument('--AMP', action="store_true", help='whether to use AMP')
    # probably not userfull
    parser.add_argument('--weighted_loss', action="store_true", help='whether to weight the loss and weight for classes')

    # datasets (ok they have 3 names but actually its just because I have 3 paths...)
    parser.add_argument('--MMI', type=str, default=None, help='path to MMI dataset')
    parser.add_argument('--Cholect', type=str, default=None, help='path to Cholect dataset')
    parser.add_argument('--AtlasDione', type=str, default=None, help='path to AtlasDione dataset')

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--as_encoder', action="store_true", help='whether to do encoder KD')
    group1.add_argument('--as_predictor', action="store_true", help='whether to do masks KD')
    group1.add_argument('--only_supervised', action="store_true", help='whether to train with just gt (no KD)')

    # loaders params
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for parallel dataloading ')
    parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory for more efficient passage to gpu')

    args = parser.parse_args()

    main(args)


