from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('models/Yolov86thRoundbestWeights.pt')

# Export the model to ONNX format
model.export(format='onnx', dynamic=True, simplify=True)