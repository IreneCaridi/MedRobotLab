from ultralytics import YOLO

weights = r"C:\Users\User\cartelle_matteo\progetto_rob\MedRobotLab\matteo_convnextsam_exp\weights\weights_seg\ConvNext_finetuning_mmi\weights\best.pt"
image_path = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\Desktop\uni Matteo\quarto anno\dopotesi\dataset\images\val\image_1.png"

# Load a model
model = YOLO(weights)

results = model(image_path)

print(results.masks)