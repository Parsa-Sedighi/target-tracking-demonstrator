"""Fake detector used to build and demo the operator UI without a trained model.

A detector is anything with a detect(frame) method that returns a list of dicts:
    {"class": "drone", "bbox": (x, y, w, h), "confidence": 0.87}
where bbox is the top-left corner plus width and height, in pixels of the frame.

When a real model (YOLO, RF-DETR, ...) is ready, wrap it in a class with the same
detect(frame) method and pass it to MainWindow(detector=...). Nothing else changes.
"""
import math
import time

# The first object disappears for HIDE_SECONDS out of every HIDE_PERIOD seconds,
# so you can see the UI go from TRACKING to SEARCHING and back.
HIDE_PERIOD = 10.0
HIDE_SECONDS = 1.0


class DemoDetector:
    def __init__(self):
        self.t0 = time.monotonic()
        # Each object moves on a smooth looping path. ax/ay = how far it swings
        # (fraction of the frame), wx/wy = how fast, phase = where it starts.
        self.objects = [
            {"cls": "drone", "size": (56, 38), "ax": 0.32, "ay": 0.22,
             "wx": 0.35, "wy": 0.50, "phase": 0.0, "hides": True},
            {"cls": "sports ball", "size": (40, 40), "ax": 0.25, "ay": 0.30,
             "wx": 0.55, "wy": 0.40, "phase": 4.0, "hides": False},
            {"cls": "hand", "size": (70, 70), "ax": 0.36, "ay": 0.18,
             "wx": 0.30, "wy": 0.60, "phase": 2.0, "hides": False},
        ]

    def detect(self, frame):
        h, w = frame.shape[:2]
        t = time.monotonic() - self.t0
        detections = []
        for i, obj in enumerate(self.objects):
            if obj["hides"] and (t % HIDE_PERIOD) < HIDE_SECONDS:
                continue
            bw, bh = obj["size"]
            cx = w / 2 + obj["ax"] * w * math.sin(obj["wx"] * t + obj["phase"])
            cy = h / 2 + obj["ay"] * h * math.sin(obj["wy"] * t + obj["phase"] * 0.5)
            x = int(min(max(cx - bw / 2, 0), w - bw))
            y = int(min(max(cy - bh / 2, 0), h - bh))
            confidence = 0.80 + 0.15 * math.sin(1.3 * t + i)
            detections.append({"class": obj["cls"], "bbox": (x, y, bw, bh),
                               "confidence": confidence})
        return detections
