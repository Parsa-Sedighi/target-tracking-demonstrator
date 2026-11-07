"""OCR of index cards on the locked target (FR-03), Tesseract via pytesseract.

Runs in its own worker thread on demand, so it never touches the coordinate
stream (PRD story 4: <= 200 ms on request without interrupting tracking).
Windows: install Tesseract (UB Mannheim build) and, if it is not on PATH, set
``ocr.tesseract_cmd`` in the config.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OcrWorker:
    def __init__(self, cfg: Optional[dict], on_result: Callable[[dict], None]):
        self.cfg = cfg or {}
        self.on_result = on_result
        self.available = False
        self.reason = ""
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        try:
            import pytesseract

            cmd = self.cfg.get("tesseract_cmd")
            if not cmd and os.name == "nt" and not shutil.which("tesseract") and os.path.exists(_WIN_DEFAULT):
                cmd = _WIN_DEFAULT
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
            self.version = str(pytesseract.get_tesseract_version())
            self.pt = pytesseract
            self.available = True
        except Exception as exc:
            self.reason = f"Tesseract not available: {exc}"
            log.info(self.reason)
        if self.available:
            threading.Thread(target=self._run, name="ocr", daemon=True).start()

    def request(self, image: np.ndarray, bbox, meta: dict) -> bool:
        if not self.available:
            self.on_result({"ok": False, "text": "", "error": self.reason or "OCR unavailable", **meta})
            return False
        H, W = image.shape[:2]
        x1, y1, x2, y2 = bbox
        m = float(self.cfg.get("margin", 0.08))
        w, h = x2 - x1, y2 - y1
        x1, y1 = int(max(0, x1 - m * w)), int(max(0, y1 - m * h))
        x2, y2 = int(min(W, x2 + m * w)), int(min(H, y2 + m * h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            self.on_result({"ok": False, "text": "", "error": "target too small for OCR", **meta})
            return False
        crop = image[y1:y2, x1:x2].copy()
        try:
            self._q.put_nowait((crop, meta, time.perf_counter()))
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        self._stop.set()

    def _prep(self, crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Return binarised candidates (dark text on white, card edge removed, text >= ~30 px tall)."""
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        target_h = int(self.cfg.get("target_height_px", 220))
        s = max(1.0, min(8.0, target_h / max(1, g.shape[0])))
        if s > 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        block = max(15, (min(g.shape[:2]) // 6) | 1)
        adapt = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)
        out = []
        for name, th in (("adaptive", adapt), ("otsu", otsu)):
            if np.mean(th) < 127:          # make text dark on a light background
                th = 255 - th
            inv = 255 - th                 # remove dark blobs touching the crop edge (card border, background)
            h, w = inv.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            for x in range(0, w, max(1, w // 16)):
                for y in (0, h - 1):
                    if inv[y, x] == 255:
                        cv2.floodFill(inv, mask, (x, y), 0)
            for y in range(0, h, max(1, h // 16)):
                for x in (0, w - 1):
                    if inv[y, x] == 255:
                        cv2.floodFill(inv, mask, (x, y), 0)
            clean = cv2.copyMakeBorder(255 - inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            out.append((name, clean))
        return out

    def _run(self) -> None:
        whitelist = self.cfg.get("whitelist", "")
        extra = f" -c tessedit_char_whitelist={whitelist}" if whitelist else ""
        attempts = [("adaptive", int(self.cfg.get("psm", 11))), ("otsu", 6)]
        while not self._stop.is_set():
            try:
                crop, meta, t_req = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            best = {"text": "", "conf": -1.0, "score": -1.0}
            try:
                cands = dict(self._prep(crop))
                for name, psm in attempts:
                    data = self.pt.image_to_data(cands[name], config=f"--oem 3 --psm {psm}{extra}",
                                                 output_type=self.pt.Output.DICT)
                    words = []
                    for w, c in zip(data["text"], data["conf"]):
                        w, c = str(w).strip(), float(c)
                        alnum = sum(ch.isalnum() for ch in w)
                        # drop fiducial squares / specks that OCR reads as "o", "0", "."
                        fiducial = len(w) <= 3 and set(w) <= set("oO0Qe.") and c < 90
                        if c >= 30 and not fiducial and (alnum >= 2 or (alnum == 1 and c >= 80)):
                            words.append((w, c))
                    if not words:
                        continue
                    text = " ".join(w for w, _ in words)
                    conf = sum(c for _, c in words) / len(words)
                    score = conf * sum(ch.isalnum() for ch in text)
                    if score > best["score"]:
                        best = {"text": text, "conf": conf, "score": score}
                    if conf > 85:      # good enough: skip the second attempt (time budget 200 ms)
                        break
                ms = (time.perf_counter() - t_req) * 1000
                min_conf = float(self.cfg.get("min_conf", 60))
                if best["text"] and best["conf"] < min_conf:
                    self.on_result({"ok": False, "text": "", "error": f"no readable text (best guess '{best['text']}', "
                                    f"conf {best['conf']:.0f} < {min_conf:.0f})", "ms": round(ms, 1), **meta})
                    continue
                self.on_result({"ok": bool(best["text"]), "text": best["text"], "conf": round(best["conf"], 1),
                                "ms": round(ms, 1), **meta})
            except Exception as exc:
                self.on_result({"ok": False, "text": "", "error": str(exc), **meta})
