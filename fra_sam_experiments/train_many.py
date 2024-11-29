import subprocess

from utils.active_graphic import get_batch_and_dataset_gui

# sul mio pc...
# FULL: D:\poli_onedrive_backup\dataset_self

# kvasir: D:\poli_onedrive_backup\kvasir_dataset
# Cholect: D:\poli_onedrive_backup\Cholect_dataset
# AtlasDione: D:\poli_onedrive_backup\AtlasDione_dataset

# Collect inputs
inputs_dict = get_batch_and_dataset_gui()

arguments_list = [
    ["--student", "RepViT", "--reshape_size", "1024", "--epochs", "100", "--batch_size", f"{inputs_dict['batch_size']}",
     "--name", "RepViT_enc_100epcs", "--sched", "cos_lr", "--lr0", "1.25e-2", "--lrf", "5e-5", "--weight_decay", "0.01",
     "--patience", "10", "--device", "gpu", "--AMP", "--as_encoder", "--Kvasir", "--Cholect", "--AtlasDione",
     "--data_path", f"{inputs_dict['path']}"],

    ["--student", "UnetEncoder", "--reshape_size", "512", "--epochs", "100", "--batch_size", f"{inputs_dict['batch_size']}",
     "--name", "Unet_enc_100epcs", "--sched", "cos_lr", "--lr0", "1.25e-2", "--lrf", "5e-5", "--weight_decay", "0.01",
     "--patience", "10", "--device", "gpu", "--AMP", "--as_encoder", "--Kvasir", "--Cholect", "--AtlasDione",
     "--data_path", f"{inputs_dict['path']}"],

["--student", "RepViT", "--reshape_size", "1024", "--epochs", "200", "--batch_size", f"{inputs_dict['batch_size']}",
     "--name", "RepViT_enc_200epcs", "--sched", "cos_lr", "--lr0", "1.25e-2", "--lrf", "5e-5", "--weight_decay", "0.01",
     "--patience", "20", "--device", "gpu", "--AMP", "--as_encoder", "--Kvasir", "--Cholect", "--AtlasDione",
     "--data_path", f"{inputs_dict['path']}"],

    ["--student", "UnetEncoder", "--reshape_size", "512", "--epochs", "200", "--batch_size", f"{inputs_dict['batch_size']}",
     "--name", "Unet_enc_200epcs", "--sched", "cos_lr", "--lr0", "1.25e-2", "--lrf", "5e-5", "--weight_decay", "0.01",
     "--patience", "20", "--device", "gpu", "--AMP", "--as_encoder", "--Kvasir", "--Cholect", "--AtlasDione",
     "--data_path", f"{inputs_dict['path']}"]
]


script_name = "DL_train.py"

for args in arguments_list:

    cmd = ["python", script_name] + args
    subprocess.run(cmd)
    break

