from functions.metrics import iou_onfoldernpz
import matplotlib.pyplot as plt

# Percorsi delle cartelle
path_target_tip = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\labels\npz_ground_truth_masks\tip"
path_prediction_tip = r"C:\Users\User\Desktop\uni_matteo\quinto_anno\laboratorio_robotics\project\predictions_zeroshot_sam2_video_mmi\merged_masks\tip"

path_target_wrist = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\labels\npz_ground_truth_masks\wrist"
path_prediction_wrist = r"C:\Users\User\Desktop\uni_matteo\quinto_anno\laboratorio_robotics\project\predictions_zeroshot_sam2_video_mmi\merged_masks\wrist"

path_target_shaft = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\labels\npz_ground_truth_masks\shaft"
path_prediction_shaft = r"C:\Users\User\Desktop\uni_matteo\quinto_anno\laboratorio_robotics\project\predictions_zeroshot_sam2_video_mmi\merged_masks\shaft"

iou_tip = iou_onfoldernpz(path_prediction_tip, path_target_tip)
iou_wrist = iou_onfoldernpz(path_prediction_wrist, path_target_wrist)
iou_shaft = iou_onfoldernpz(path_prediction_shaft, path_target_shaft)

data = [iou_tip, iou_wrist, iou_shaft]
labels = ['Tip', 'Wrist', 'Shaft']

colors = ['lightblue', 'lightgreen', 'lightcoral']
box = plt.boxplot(data, labels=labels, patch_artist=True)

# Personalizza i colori delle box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Boxplot delle IoU per Tip, Wrist e Shaft')
plt.ylabel('IoU')
plt.xlabel('Segmento')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
