
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best_8S_TUN2.pt")

# Export to NCNN format
model.export(format="ncnn")  # creates '/yolov8n_ncnn_model'


# Load the exported NCNN model
ncnn_model = YOLO("best_8S_TUN2_ncnn_model")
