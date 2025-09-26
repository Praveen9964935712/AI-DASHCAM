from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        # COCO classes for vehicles and persons
        self.allowed_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            if class_id in self.allowed_classes:
                detections.append([x1, y1, x2, y2, conf, class_id])
        return detections
