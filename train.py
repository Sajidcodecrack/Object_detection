from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' or other variants

# Train the model
model.train(data='datas.yaml', epochs=50, imgsz=640)
