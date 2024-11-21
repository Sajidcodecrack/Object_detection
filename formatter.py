import torch
from ultralytics import YOLO

# Load the model
model = YOLO('D:/Machine learning/mask_cup/best.pt')

# Export the model to ONNX
model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
