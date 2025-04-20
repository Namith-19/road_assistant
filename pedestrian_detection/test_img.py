
from ultralytics import YOLO
import os

model = YOLO("/mnt/c/Users/namit/Desktop/CODE/Road_assistant_new/Pedestrian_detection/yolov8n_person/content/runs/train/yolov8n_person2/weights/best.pt")


results = model.predict(
    source="/mnt/c/Users/namit/Desktop/CODE/Road_assistant_new/Pedestrian_detection/custom",
    conf=0.5,     
    save=True,     
    imgsz=640,     
    show=False     
)


from IPython.display import Image, display
output_dir = model.predictor.save_dir
example_img = os.path.join(output_dir, os.listdir(output_dir)[0])
display(Image(filename=example_img))
