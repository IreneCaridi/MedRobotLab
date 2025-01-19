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

########################################################################################################################
#          !!!!!!!!   DA MODIFICARE    !!!!!!!!
########################################################################################################################

# put your paths here
dataset_path = r"C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\dataset"
pre_weights_m1 = 'data/weights_distill/RepViT_m1/weights/best_40.pt'
pre_weights_m2 = 'data/weights_distill/RepViT_m2/weights/best_54.pt'
batch_size = 32

args = ['--dataset_path', f'{dataset_path}', '--pre_weights_m1', f'{pre_weights_m1}',
        '--pre_weights_m2', f'{pre_weights_m2}', '--batch_size', f'{batch_size}']

script_names = ["transfer_scratch.py", 'finetune.py']

for script_name in script_names:

    cmd = ["python", script_name] + args
    subprocess.run(cmd)


