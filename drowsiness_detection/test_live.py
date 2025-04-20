from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/namit/Desktop/CODE/Road_assistant_new/Drowsiness_detection/DDD_v2/content/runs/detect/drowsiness_yolov8n2/weights/best.pt") 
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

 
    results = model(frame)

    annotated_frame = results[0].plot() 


    cv2.imshow("Drowsiness Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()