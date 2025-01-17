import subprocess
from glob import glob
import os

#from utils.active_graphic import get_batch_and_dataset_gui

# sul mio pc...
# FULL: D:\poli_onedrive_backup\dataset_self

# kvasir: D:\poli_onedrive_backup\kvasir_dataset
# Cholect: D:\poli_onedrive_backup\Cholect_dataset
# AtlasDione: D:\poli_onedrive_backup\AtlasDione_dataset

# Collect inputs
#inputs_dict = get_batch_and_dataset_gui()

# DA MODIFICARE
dataset_path = r"C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\dataset"
pre_weights_m1 = 'data/weights_distill/RepViT_m1/weights/best_40.pt'
pre_weights_m2 = 'data/weights_distill/RepViT_m2/weights/best_54.pt'
batch_size = 32


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

## MODELLI SCRATCH
    # RepViT-encoder m1
    ["--student", "RepViTEncDec", "--epochs", "300", "--batch_size", f'{batch_size}', '--arch', 'm1', '--folder', 'crop/m1',
     "--name", "RepViT_enc_dec_scratch", "--sched", "cos_lr", "--lr0", "1e-4", "--lrf", "1e-3", "--weight_decay", "0.05",
     "--patience", "80", "--device", "gpu", "--AMP", "--only_supervised", "--Cholect", "--data_path", f'{dataset_path}'],

#   # RepViT-encoder m2
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

    #     ["--student", "UnetEncoder", "--reshape_size", "512", "--epochs", "100", "--batch_size", f'{batch_size}',
#      "--name", "Unet_enc_100epcs", "--sched", "cos_lr", "--lr0", "1.25e-2", "--lrf", "5e-5", "--weight_decay", "0.01",
#      "--patience", "10", "--device", "gpu", "--AMP", "--as_encoder", "--Kvasir", "--Cholect", "--AtlasDione",
#      "--data_path", f'{dataset_path}', '--n_workers', '7'],
#
# ["--student", "RepViT", "--reshape_size", "1024", "--epochs", "200", "--batch_size", f'{batch_size}',
#      "--name", "RepViT_enc_200epcs", "--sched", "cos_lr", "--lr0", "1.25e-2", "--lrf", "5e-5", "--weight_decay", "0.01",
#      "--patience", "20", "--device", "gpu", "--AMP", "--as_encoder", "--Kvasir", "--Cholect", "--AtlasDione",
#      "--data_path", f'{dataset_path}', '--n_workers', '7'],
#
#     ["--student", "UnetEncoder", "--reshape_size", "512", "--epochs", "200", "--batch_size", f'{batch_size}',
#      "--name", "Unet_enc_200epcs", "--sched", "cos_lr", "--lr0", "1.25e-2", "--lrf", "5e-5", "--weight_decay", "0.01",
#      "--patience", "20", "--device", "gpu", "--AMP", "--as_encoder", "--Kvasir", "--Cholect", "--AtlasDione",
#      "--data_path", f'{dataset_path}', '--n_workers', '7']
]


script_name = "DL_train.py"

for args in arguments_list:

    cmd = ["python", script_name] + args
    subprocess.run(cmd)


