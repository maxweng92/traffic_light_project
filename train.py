import ultralytics
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model using the specified dataset
model.train(data=r'C:/Users/Hankz/Desktop/yolo/datasets/dataset/data.yaml', epochs=20, imgsz=320)
#model.train(data=r'data.yaml', epochs=30, imgsz=320, augment = True)