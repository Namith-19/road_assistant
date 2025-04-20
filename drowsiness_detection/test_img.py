import cv2
import torch
from ultralytics import YOLO
import os


model_path = "/mnt/c/Users/namit/Desktop/CODE/Road_assistant_new/Drowsiness_detection/DDD_v2/content/runs/detect/drowsiness_yolov8n2/weights/best.pt"   
source_folder = "/mnt/c/Users/namit/Desktop/CODE/Road_assistant_new/Drowsiness_detection/Custom" 
output_folder = "/mnt/c/Users/namit/Desktop/CODE/Road_assistant_new/op"
conf_threshold = 0.3  


if model_path.endswith('.pt'):
    model = YOLO(model_path)
elif model_path.endswith('.onnx'):
    model = YOLO(model_path)  
else:
    raise ValueError("Unsupported model format")


os.makedirs(output_folder, exist_ok=True)


image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(source_folder, img_name)
    img = cv2.imread(img_path)


    results = model(img, conf=conf_threshold)


    res_plotted = results[0].plot() 
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, res_plotted)

    print(f"Inference done → {output_path}")

print("  All done!")
