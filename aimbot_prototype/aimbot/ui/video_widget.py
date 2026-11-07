"""Live video canvas with HUD overlays (FR-06) and click / drag target designation (FR-09).

Rendering happens only in the Qt GUI thread from the latest engine snapshot, so it
can never delay the tracking or control threads (PRD: UI isolated from control path).
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

CLASS_COLORS = {
    "drone": QColor(80, 200, 255), "ball": QColor(240, 230, 60), "card": QColor(255, 255, 255),
    "hand": QColor(255, 140, 120), "person": QColor(255, 120, 220), "phone": QColor(160, 255, 160),
    # non-targets the YOLO26 drone model is trained to tell apart from drones
    "bird": QColor(150, 150, 150), "airplane": QColor(170, 170, 200), "helicopter": QColor(200, 170, 140),
    "manual": QColor(255, 180, 0),
}
STATE_COLORS = {
    "IDLE": QColor(120, 130, 140), "SEARCHING": QColor(240, 180, 30), "TRACKING": QColor(40, 210, 90),
    "LOST": QColor(255, 90, 40), "PAUSED": QColor(120, 160, 255), "CALIBRATING": QColor(170, 120, 255),
    "ESTOP": QColor(230, 30, 30), "FAULT": QColor(230, 30, 30),
}


class VideoWidget(QWidget):
    def __init__(self, on_designate: Callable[[float, float, Optional[float], Optional[float]], None],
                 on_set_aim: Callable[[float, float], None], parent=None):
        super().__init__(parent)
        self.on_designate = on_designate
        self.on_set_aim = on_set_aim
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._img: Optional[QImage] = None
        self._arr: Optional[np.ndarray] = None
        self.snap: dict = {}
        self._drag_start: Optional[QPointF] = None
        self._drag_now: Optional[QPointF] = None
        self.aim_mode = False
        self.render_ms = 0.0
        self.display_fps = 0.0
        self._fps_n = 0
        self._fps_t = time.perf_counter()
        self._target = QRectF()

    # ------------------------------------------------------------------ data in
    def set_frame(self, arr: Optional[np.ndarray], snap: dict) -> None:
        self.snap = snap
        if arr is not None:
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            self._arr = arr   # keep alive while QImage references it
            h, w = arr.shape[:2]
            self._img = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_BGR888)
        self.update()

    # ------------------------------------------------------------------ geometry
    def _image_rect(self) -> QRectF:
        if self._img is None:
            return QRectF(0, 0, self.width(), self.height())
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_norm(self, p: QPointF) -> Optional[tuple[float, float]]:
        r = self._target
        if r.width() <= 0 or not r.contains(p):
            return None
        return (p.x() - r.x()) / r.width(), (p.y() - r.y()) / r.height()

    def _from_norm(self, x: float, y: float) -> QPointF:
        r = self._target
        return QPointF(r.x() + x * r.width(), r.y() + y * r.height())

    def _rect_norm(self, b) -> QRectF:
        r = self._target
        return QRectF(r.x() + b[0] * r.width(), r.y() + b[1] * r.height(), (b[2] - b[0]) * r.width(),
                      (b[3] - b[1]) * r.height())

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = e.position()
            self._drag_now = e.position()
        elif e.button() == Qt.MouseButton.RightButton:
            n = self._to_norm(e.position())
            if n:
                self.on_set_aim(*n)

    def mouseMoveEvent(self, e) -> None:
        if self._drag_start is not None:
            self._drag_now = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return
        a, b = self._to_norm(self._drag_start), self._to_norm(e.position())
        self._drag_start = self._drag_now = None
        self.update()
        if a is None or b is None:
            return
        if self.aim_mode:
            self.aim_mode = False
            self.on_set_aim(*b)
            return
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        if dx < 0.01 and dy < 0.01:
            self.on_designate(b[0], b[1], None, None)       # click
        else:
            self.on_designate(min(a[0], b[0]), min(a[1], b[1]), dx, dy)   # drag rectangle

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _e) -> None:
        t0 = time.perf_counter()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(12, 14, 18))
        self._target = self._image_rect()
        if self._img is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(self._target, self._img)
        else:
            p.setPen(QColor(160, 170, 180))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "waiting for video...")
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_hud(p)
        p.end()
        self.render_ms = (time.perf_counter() - t0) * 1000
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self.display_fps = self._fps_n / (now - self._fps_t)
            self._fps_n, self._fps_t = 0, now

    def _label(self, p: QPainter, x: float, y: float, text: str, color: QColor, bg=QColor(0, 0, 0, 160)) -> None:
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 8, fm.height() + 2
        p.fillRect(QRectF(x, y - h, w, h), bg)
        p.setPen(color)
        p.drawText(QPointF(x + 4, y - 4), text)

    def _draw_hud(self, p: QPainter) -> None:
        s = self.snap or {}
        ov = s.get("overlay") or {}
        state = (s.get("state") or {}).get("state", "?")
        f = QFont("Consolas", 9)
        f.setStyleHint(QFont.StyleHint.Monospace)
        p.setFont(f)
        # ROI used by the detector
        if ov.get("roi"):
            pen = QPen(QColor(120, 120, 255, 140), 1, Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawRect(self._rect_norm(ov["roi"]))
        # detections
        lock = ov.get("lock")
        for d in ov.get("detections", []):
            col = CLASS_COLORS.get(d["cls"], QColor(200, 200, 200))
            if lock and d.get("tid") == lock.get("tid"):
                continue
            pen = QPen(col, 1.2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = self._rect_norm(d["bbox"])
            p.drawRect(r)
            self._label(p, r.x(), r.y(), f"{d['cls']} #{d['tid']} {d['conf']:.2f}", col)
        # aim point (boresight / LiDAR axis) + LiDAR beam footprint
        aim = ov.get("aim")
        if aim:
            c = self._from_norm(*aim)
            pen = QPen(QColor(255, 255, 255, 200), 1.2)
            p.setPen(pen)
            L = 14
            p.drawLine(QPointF(c.x() - L, c.y()), QPointF(c.x() - 4, c.y()))
            p.drawLine(QPointF(c.x() + 4, c.y()), QPointF(c.x() + L, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - L), QPointF(c.x(), c.y() - 4))
            p.drawLine(QPointF(c.x(), c.y() + 4), QPointF(c.x(), c.y() + L))
            br = ov.get("beam_r", 0) * self._target.width()
            if br > 3:
                p.setPen(QPen(QColor(255, 80, 80, 170), 1, Qt.PenStyle.DashLine))
                p.drawEllipse(c, br, br)
        # predicted target position (Kalman)
        if ov.get("pred"):
            c = self._from_norm(*ov["pred"])
            p.setPen(QPen(QColor(0, 255, 255), 1.5))
            p.drawLine(QPointF(c.x() - 6, c.y() - 6), QPointF(c.x() + 6, c.y() + 6))
            p.drawLine(QPointF(c.x() - 6, c.y() + 6), QPointF(c.x() + 6, c.y() - 6))
        # locked target: corner brackets
        if lock:
            r = self._rect_norm(lock["bbox"])
            good = lock.get("misses", 0) == 0
            col = QColor(40, 230, 90) if good else QColor(255, 170, 0)
            p.setPen(QPen(col, 2.5))
            k = max(6.0, min(r.width(), r.height()) * 0.25)
            for (x, y, sx, sy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                                   (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
                p.drawLine(QPointF(x, y), QPointF(x + sx * k, y))
                p.drawLine(QPointF(x, y), QPointF(x, y + sy * k))
            name = f"LOCK {lock['cls']}" + (f" #{lock['tid']}" if lock.get("tid", -1) >= 0 else "")
            self._label(p, r.x(), r.y() - 2, f"{name}  {lock.get('score', 0):.2f}", col)
        # drag rectangle
        if self._drag_start is not None and self._drag_now is not None:
            p.setPen(QPen(QColor(255, 200, 0), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        # top-left text block
        tel = s.get("telemetry") or {}
        lat = s.get("latency") or {}
        rates = s.get("rates") or {}
        jit = s.get("jitter") or {}
        lines = [f"{state}",
                 f"cam {rates.get('camera', 0):5.1f}  det {rates.get('macro', 0):4.1f}  ctl {rates.get('control', 0):5.1f}  "
                 f"disp {self.display_fps:4.1f} Hz",
                 f"latency p95 {lat.get('cmd_p95') or 0:5.1f} ms   render {self.render_ms:4.1f} ms"]
        if tel.get("err_az_deg"):
            e = math.hypot(float(tel["err_az_deg"]), float(tel["err_el_deg"]))
            lines.append(f"err {e:5.2f} deg   jitter {jit.get('jitter_dia_in_20ft', float('nan')):4.2f} in @20ft")
        if tel.get("range_m"):
            lines.append(f"range {float(tel['range_m']):6.2f} m {'(on target)' if tel.get('range_on_target') == 1 else ''}")
        x0, y0 = self._target.x() + 8, self._target.y() + 8
        p.fillRect(QRectF(x0 - 4, y0 - 2, 330, 16 * len(lines) + 6), QColor(0, 0, 0, 140))
        for i, ln in enumerate(lines):
            p.setPen(STATE_COLORS.get(state, QColor(255, 255, 255)) if i == 0 else QColor(220, 225, 230))
            p.drawText(QPointF(x0, y0 + 13 + 16 * i), ln)
        # banners
        banner = None
        if state == "LOST":
            banner = ("TARGET LOST", QColor(255, 90, 40))
        elif state == "ESTOP":
            banner = ("EMERGENCY STOP", QColor(230, 30, 30))
        elif state == "FAULT":
            banner = ("FAULT - " + (s.get("state") or {}).get("reason", ""), QColor(230, 30, 30))
        elif state == "SEARCHING" and (s.get("state") or {}).get("reacquire_remaining_s") is not None:
            banner = (f"RE-ACQUIRING  {s['state']['reacquire_remaining_s']:.1f} s", QColor(240, 180, 30))
        elif self.aim_mode:
            banner = ("CLICK THE POINT THE LiDAR / LASER IS AIMED AT", QColor(170, 120, 255))
        if banner:
            bf = QFont("Segoe UI", 16, QFont.Weight.Bold)
            p.setFont(bf)
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(banner[0]) + 30
            r = QRectF(self._target.center().x() - w / 2, self._target.bottom() - 60, w, 40)
            p.fillRect(r, QColor(0, 0, 0, 170))
            p.setPen(QPen(banner[1], 2))
            p.drawRect(r)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, banner[0])
