# src/aimbot_vision/yolo_classifier.py
import cv2
import numpy as np
from ultralytics import YOLO

class TargetClassifier:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.65):
        self.model = YOLO(model_path)
        self.conf_thresh = confidence_threshold
        self.target_classes = {'drone', 'hand', 'sports ball'}

    def detect_targets(self, frame: np.ndarray) -> list:
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]
            
            if class_name in self.target_classes and conf >= self.conf_thresh:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'class': class_name,
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': conf
                })
        return detections