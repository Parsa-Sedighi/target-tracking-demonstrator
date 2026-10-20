# src/aimbot_vision/ocr_extractor.py
import cv2
import pytesseract
import numpy as np

def extract_target_text(frame: np.ndarray, bbox: tuple) -> str:
    x, y, w, h = bbox
    # Boundary clamping
    H, W = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    roi = frame[y:min(y+h, H), x:min(x+w, W)]
    
    if roi.size == 0:
        return ""
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(thresh, config=config)
    return raw_text.strip()