# src/aimbot_ui/video_canvas.py
import cv2
import numpy as np

class HUDOverlayRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render_hud(self, frame: np.ndarray, detections: list) -> np.ndarray:
        out_frame = frame.copy()
        h, w = out_frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Center boresight marker
        cv2.drawMarker(out_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        
        for det in detections:
            x, y, bw, bh = det['bbox']
            label = f"{det['class']} [{det['confidence']:.2f}]"
            # Draw targeting box and tag
            cv2.rectangle(out_frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(out_frame, label, (x, max(y - 8, 15)), self.font, 0.5, (0, 0, 255), 1)
            
        return out_frame