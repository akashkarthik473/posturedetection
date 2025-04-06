from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")


results = model.train(data="webcam_model/config.yaml", epochs=75)

