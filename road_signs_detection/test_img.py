from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "/mnt/c/Users/namit/Desktop/CODE/Road_assistant_new/Road_signs_detection/content/runs/detect/augmented-signs3/weights/best.pt" 
SOURCE = "/mnt/c/Users/namit/Desktop/CODE/Road_assistant_new/Road_signs_detection/test_img" 
SAVE = True  
CONFIDENCE_THRESHOLD = 0.3

model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE, 
    conf=CONFIDENCE_THRESHOLD, 
    save=SAVE
)