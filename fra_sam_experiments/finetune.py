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
        # fine-tuning for m1 model to Unet-encoder
        ["--student", "RepViTUnet", "--epochs", "70", "--batch_size", f'{batch_size}', '--arch', 'm1', '--folder', 'crop/m1',
         "--name", "RepViTUnet_finetune", "--sched", "cos_lr", "--lr0", "1e-7", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "20", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{glob(f"{os.getcwd()}/runs/train/crop/m1/RepViTUnet_transfer/weights/best_*.pt")[0]}'],

        # fine-tuning for m2 model to Unet-encoder
        ["--student", "RepViTUnet", "--epochs", "70", "--batch_size", f'{batch_size}', '--arch', 'm2', '--folder', 'crop/m2',
         "--name", "RepViTUnet_finetune", "--sched", "cos_lr", "--lr0", "1e-7", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "20", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{glob(f"{os.getcwd()}/runs/train/crop/m2/RepViTUnet_transfer/weights/best_*.pt")[0]}'],

        # fine-tuning for m1 model to RepViT-encoder
        ["--student", "RepViTEncDec", "--epochs", "70", "--batch_size", f'{batch_size}', '--arch', 'm1', '--folder', 'crop/m1',
         "--name", "RepViT_enc_dec_finetune", "--sched", "cos_lr", "--lr0", "1e-7", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "20", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{glob(f"{os.getcwd()}/runs/train/crop/m1/RepViT_enc_dec_transfer/weights/best_*.pt")[0]}'],

        # fine-tuning for m2 model to RepViT-encoder
        ["--student", "RepViTEncDec", "--epochs", "70", "--batch_size", f'{batch_size}', '--arch', 'm2', '--folder', 'crop/m2',
         "--name", "RepViT_enc_dec_finetune", "--sched", "cos_lr", "--lr0", "1e-7", "--lrf", "1e-3", "--weight_decay", "0.05",
         "--patience", "20", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}',
         '--pre_weights', f'{glob(f"{os.getcwd()}/runs/train/crop/m2/RepViT_enc_dec_transfer/weights/best_*.pt")[0]}'],
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

