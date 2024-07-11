import torch
import torch.onnx
from ultralytics import YOLO

# Load YOLOv8 model for inference
model = YOLO('best_8S_TUN2.pt', task='detect')  # Specify task to avoid training configurations

# Ensure the model is in evaluation mode
model.model.eval()

# Create a dummy input with the correct shape (e.g., 1, 3, 640, 640)
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX
torch.onnx.export(model.model, dummy_input, "best_8S_TUN2.onnx", opset_version=12)