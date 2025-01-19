import subprocess
from glob import glob
import os

import argparse

#from utils.active_graphic import get_batch_and_dataset_gui

# sul mio pc...
# FULL: D:\poli_onedrive_backup\dataset_self

# kvasir: D:\poli_onedrive_backup\kvasir_dataset
# Cholect: D:\poli_onedrive_backup\Cholect_dataset
# AtlasDione: D:\poli_onedrive_backup\AtlasDione_dataset

# Collect inputs
#inputs_dict = get_batch_and_dataset_gui()

# DA MODIFICARE
def main(args):
    dataset_path =args.dataset_path
    pre_weights_m1 = args.pre_weights_m1
    pre_weights_m2 = args.pre_weights_m2
    batch_size = args.batch_size


    arguments_list = [
        # transfer learning for m1 model to Unet-encoder
        ["--student", "RepViTUnet", "--epochs", "200", "--batch_size", f'{batch_size}', '--arch', 'm1', '--folder', 'crop/m1',
         "--name", "RepViTUnet_transfer", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "50", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{pre_weights_m1}', '--freeze_backbone'],

        # transfer learning for m2 model to Unet-encoder
        ["--student", "RepViTUnet", "--epochs", "200", "--batch_size", f'{batch_size}', '--arch', 'm2', '--folder', 'crop/m2',
         "--name", "RepViTUnet_transfer", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "50", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{pre_weights_m2}', '--freeze_backbone'],

        # transfer learning for m1 model to RepViT-encoder
        ["--student", "RepViTEncDec", "--epochs", "200", "--batch_size", f'{batch_size}', '--arch', 'm1', '--folder', 'crop/m1',
         "--name", "RepViT_enc_dec_transfer", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "50", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{pre_weights_m1}', '--freeze_backbone'],

        # transfer learning for m2 model to RepViT-encoder
        ["--student", "RepViTEncDec", "--epochs", "200", "--batch_size", f'{batch_size}', '--arch', 'm2', '--folder', 'crop/m2',
         "--name", "RepViT_enc_dec_transfer", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "50", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{pre_weights_m2}', '--freeze_backbone'],

    ## MODELLI SCRATCH
        # RepViT-encoder m1
        ["--student", "RepViTEncDec", "--epochs", "300", "--batch_size", f'{batch_size}', '--arch', 'm1', '--folder', 'crop/m1',
         "--name", "RepViT_enc_dec_scratch", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "80", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}'],

        # RepViT-encoder m2
        ["--student", "RepViTEncDec", "--epochs", "300", "--batch_size", f'{batch_size}', '--arch', 'm2', '--folder', 'crop/m2',
         "--name", "RepViT_enc_dec_scratch", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "80", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}'],

        # RepViT-unet m1
        ["--student", "RepViTUnet", "--epochs", "300", "--batch_size", f'{batch_size}', '--arch', 'm1', '--folder', 'crop/m1',
         "--name", "RepViTUnet_scratch", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "80", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}'],

        # RepViT-unet m2
        ["--student", "RepViTUnet", "--epochs", "300", "--batch_size", f'{batch_size}', '--arch', 'm2', '--folder', 'crop/m2',
         "--name", "RepViTUnet_scratch", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "80", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}'],

        # Unet un po a cazzo
        ["--student", "Unet", "--epochs", "300", "--batch_size", f'{batch_size}', '--folder', 'crop/Unet',
         "--name", "Unet_scratch", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "80", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}'],
    ]


    script_name = "DL_train.py"

    for args in arguments_list:

        cmd = ["python", script_name] + args
        subprocess.run(cmd)


if __name__ == "__main__":

    # list of arguments (ADJUST for student and SAM)
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--pre_weights_m1', type=str, required=True)
    parser.add_argument('--pre_weights_m2', type=str, required=True)

    args = parser.parse_args()

    main(args)
