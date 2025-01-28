from ultralytics import YOLO
import torch

def convert_yolo_to_onnx():
    # Load the YOLOv8 model
    model = YOLO('models/best.pt')

    # Export the model to ONNX format
    model.export(format='onnx', opset=12, simplify=True, dynamic=False, imgsz=640)

    print("Model converted to ONNX format successfully.")

if __name__ == "__main__":
    convert_yolo_to_onnx()