import argparse
import os

import torch
import torch.nn as nn
from pathlib import Path

from models.DL import ModelClass, check_load_model
from models.DL.common import Dummy, ConvNeXt, ConvNeXtSAM, ResNet1, ResNet2, ResNetTransform, ResNetTransform2, \
                             ResNetTransformerAtt, TransformerEncDec, ResUnet, ResUnetAtt, DarkNetCSP, ResUnetAtt2, \
                             DarkNetCSPBoth, LearnableInitBiLSTM, LearnableInitBiLSTM2, MLPdo, MLPatt, MLPattDo, MLP
from utils.dataloaders import
from utils.DL.callbacks import Callbacks, EarlyStopping, Saver
from utils.DL.loaders import
from utils.DL.optimizers import get_optimizer, scheduler
from utils.DL.collates import keep_unchanged
from utils.DL.metrics import Metrics
from utils.DL.logger import Loggers, LoggersBoth, log_confidence_score, save_predictions
from utils import random_state, increment_path, json_from_parser

# setting all random states
random_state(36)


def main(args):

    # unpacking
    folder = args.folder
    name = args.name
    if not args.test:
        p = Path(folder) / 'train'
    else:
        p = Path(folder) / 'test'
    if not os.path.isdir(p):
        os.mkdir(p)
    save_path = increment_path(p, name)
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    mode = args.mode
    if mode == "binary":
        num_classes = 2
    else:
        num_classes = 3

    bi_head_f = False

    # saving inputs
    json_from_parser(args, save_path)

    # dataset
    # if args.crops:  # crops dataset
    #     crops_data = Crops()
    #     crops_data.split(test_size=0.15)
    #     test_set = CropsDataset(crops_data.test, mode=mode, stratify=False, normalization=args.data_norm,
    #                             sig_mode='single', bi_head=bi_head_f)
    #     val_set = CropsDataset(crops_data.val, mode=mode, stratify=True, normalization=args.data_norm,
    #                            sig_mode='single', bi_head=bi_head_f)
    #     train_set = CropsDataset(crops_data.train, mode=mode, stratify=True, normalization=args.data_norm,
    #                              sig_mode='single', bi_head=bi_head_f)
    # else:
    #     raise ValueError("data format not recognised")

    # model (ADJUST)
    if "." not in args.model:
        # means it is not a weight and has to be imported
        if args.model == "Dummy":
            model = Dummy(num_classes)
            # loading model = bla bla bla
        else:
            raise TypeError("Model name not recognised")
    else:
        # it is a weight
        model = args.model

    # double-checking whether you parsed weights or model and accounting for transfer learning
    mod = check_load_model(model, args.backbone)


    # initializing callbacks ( could be handled more concisely i guess...)
    stopper = EarlyStopping(patience=args.patience, monitor="val_loss", mode="min")
    saver = Saver(model=mod, save_best=True, save_path=save_path, monitor="val_loss",
                  mode='min')
    callbacks = Callbacks([stopper, saver])

    # initializing metrics
    metrics = Metrics(num_classes=num_classes, device=device, top_k=1, thresh=0.5)

    # if args.weighted_loss:
    # if args.cropped_seq or args.cropped_seq_raw:
    #     weights = torch.tensor([0.62963445, 2.42849968], dtype=torch.float32)  # only m9
    # else:
    #     weights = None

    # initializing loss and optimizer (ADJUST)
    loss_fn = nn.CrossEntropyLoss(weight=None, reduction="mean", label_smoothing=args.lab_smooth)
    opt = get_optimizer(mod, args.opt, args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)

    # initializing loggers
    logger = Loggers(metrics=metrics, save_path=save_path, opt=opt, device=device, test=args.test)


    # lr scheduler
    sched = scheduler(opt, args.sched, args.lrf, epochs)

    # building loaders
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padding_x)  # to pad
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding_x)  # to pad

    # building model (ADJUST)
    model = ModelClass(mod, (train_loader, val_loader, test_loader), loss_fn=loss_fn, device=device, AMP=args.AMP,
                       optimizer=opt, metrics=metrics, loggers=logger, callbacks=callbacks, sched=sched,
                       freeze=freeze_list, sequences=seq_flag, bi_head=bi_head_f)

if __name__ == "__main__":

    # list of arguments (ADJUST for student and SAM)
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--model', type=str, required=True, help='name of model to train or path to weights to train')
    parser.add_argument('--backbone', type=str, default=None, help='path to backbone weights, if present it ONLY loads weights for it')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    #ADJUST
    parser.add_argument('--mode', type=str, required=True, choices=["binary", "all", "both"], help="ADJUST")
    parser.add_argument('--data_norm', type=str, default=None, choices=["min_max", "RobustScaler", "Z-score"], help="type of scaler for data")
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

    # ADJUST for selecting the dataset
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('--crops', action="store_true", help='whether to use Crops dataset')
    # group.add_argument('--crops_raw', action="store_true", help='whether to use Crops_raw dataset (extracted from raw signal)')


    args = parser.parse_args()

    main(args)


