"""Operator UI mockup for the Target Tracking Demonstrator.

Shows live video (demo, webcam, or a video file) with detection boxes, lets the
operator click a box to select a target, and start/stop tracking. Nothing here
controls real hardware; it is the display side only.
"""
import math
import sys
import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from demo_detector import DemoDetector
from video_canvas import HUDOverlayRenderer  # Parsa's box/crosshair drawing
from camera_source import CameraSource, load_camera_config

DEMO_SIZE = (960, 540)          # width, height of the demo background
LOST_FRAMES_LIMIT = 15          # frames without the target before SEARCHING
MATCH_DISTANCE = 0.18           # how far (fraction of frame width) a target may move per frame
TRACK_KEEP_FRAMES = 45          # keep a vanished object's ID this long (1.5 s at 30 fps)
STATE_COLORS = {"IDLE": "#6b7280", "SEARCHING": "#d97706", "TRACKING": "#16a34a"}


class VideoLabel(QLabel):
    """Shows frames and reports clicks in frame pixel coordinates."""
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background-color: #111; color: #e5e7eb;")
        self._frame_w = 0
        self._frame_h = 0

    def show_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._frame_w, self._frame_h = w, h
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull() or self._frame_w == 0:
            return
        # The picture is centered inside the label, so remove the margins first.
        left = (self.width() - pixmap.width()) / 2
        top = (self.height() - pixmap.height()) / 2
        px = event.position().x() - left
        py = event.position().y() - top
        if 0 <= px < pixmap.width() and 0 <= py < pixmap.height():
            self.clicked.emit(int(px / pixmap.width() * self._frame_w),
                              int(py / pixmap.height() * self._frame_h))


class MainWindow(QMainWindow):
    def __init__(self, detector=None, detector_name=None, camera_config_path=None,
                 camera_capture_factory=None):
        super().__init__()
        self.setWindowTitle("Target Tracking Demonstrator - Operator UI")
        # The real model (e.g. RF-DETR) runs on webcam / video frames. The Demo source has
        # no real drones in it, so it always uses the fake DemoDetector.
        self.model_detector = detector
        self.model_name = detector_name or ("Model" if detector else None)
        self.demo_detector = DemoDetector()
        self.renderer = HUDOverlayRenderer()

        self.camera_config_path = camera_config_path
        self.camera_capture_factory = camera_capture_factory
        self.camera_error = False
        self.capture = None          # CameraSource, video-file capture, or None on Demo
        self.source_name = "Demo"
        self.is_file = False
        self.selected = None         # {"class": str, "center": (x, y)} of the chosen target
        self.tracking = False
        self.lost_frames = 0
        self.state = "IDLE"
        self.last_dets = []
        self.tracks = {}             # id -> {"class", "center", "age"}, used by the ID assigner
        self.next_id = 1
        self.frame_times = deque(maxlen=30)   # recent frame timestamps, for the FPS readout
        self.target_fps = 30
        self.fps_color = None
        self.demo_frame = self._make_demo_background()

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)         # about 30 frames per second
        if camera_config_path is not None:
            self.source_box.setCurrentText("Camera")

    # ---------- layout ----------
    def _build_ui(self):
        self.video = VideoLabel()
        self.video.clicked.connect(self._on_video_clicked)

        self.source_box = QComboBox()
        self.source_box.addItems(["Demo", "Camera", "Video file..."])
        self.source_box.currentTextChanged.connect(self._on_source_changed)
        source_group = QGroupBox("Video source")
        source_layout = QVBoxLayout(source_group)
        source_layout.addWidget(self.source_box)
        self.camera_label = QLabel()
        self.camera_label.setWordWrap(True)
        source_layout.addWidget(self.camera_label)
        self.restart_camera_button = QPushButton("Restart camera")
        self.restart_camera_button.setEnabled(False)
        self.restart_camera_button.clicked.connect(self._restart_camera)
        source_layout.addWidget(self.restart_camera_button)

        self.state_label = QLabel()
        self.state_label.setAlignment(Qt.AlignCenter)
        self._show_state()

        self.start_button = QPushButton("Start tracking")
        self.stop_button = QPushButton("Stop tracking")
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)

        self.target_label = QLabel()
        self.target_label.setStyleSheet("font-family: Menlo, Consolas, monospace;")
        target_group = QGroupBox("Selected target")
        QVBoxLayout(target_group).addWidget(self.target_label)

        self.det_label = QLabel()
        self.det_label.setAlignment(Qt.AlignTop)
        self.det_label.setStyleSheet("font-family: Menlo, Consolas, monospace;")
        det_group = QGroupBox("Detections")
        QVBoxLayout(det_group).addWidget(self.det_label)

        panel = QVBoxLayout()
        panel.addWidget(source_group)
        panel.addWidget(self.state_label)
        panel.addLayout(buttons)
        panel.addWidget(target_group)
        panel.addWidget(det_group, 1)
        self.detector_label = QLabel()
        self.detector_label.setWordWrap(True)
        self.detector_label.setStyleSheet("font-family: Menlo, Consolas, monospace;")
        panel.addWidget(self.detector_label)
        self._show_detector()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-family: Menlo, Consolas, monospace;")
        panel.addWidget(self.fps_label)
        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(290)

        root = QHBoxLayout()
        root.addWidget(self.video, 1)
        root.addWidget(side)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Click a box in the video to select a target.")
        self._refresh_buttons()

    def _make_demo_background(self):
        w, h = DEMO_SIZE
        rows = np.linspace(25, 70, h).astype(np.uint8)[:, None]
        bg = np.zeros((h, w, 3), np.uint8)
        bg[:, :, 0] = rows
        bg[:, :, 1] = rows // 2
        bg[:, :, 2] = rows // 3
        for x in range(0, w, 80):
            cv2.line(bg, (x, 0), (x, h), (70, 60, 50), 1)
        for y in range(0, h, 80):
            cv2.line(bg, (0, y), (w, y), (70, 60, 50), 1)
        cv2.putText(bg, "DEMO SOURCE - fake detections", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        return bg

    # ---------- video source ----------
    def _on_source_changed(self, name):
        path = None
        if name == "Video file...":
            path, _ = QFileDialog.getOpenFileName(
                self, "Open video", "", "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)")
            if not path:
                self._revert_source_box()
                return
        if not self._set_source(name, path):
            self._revert_source_box()

    def _set_source(self, name, path=None):
        """Switch the video source. Returns False if it could not be opened."""
        capture, interval = None, 33
        if name == "Camera":
            if self.capture is not None:
                self.capture.release()
            self.capture = CameraSource(capture_factory=self.camera_capture_factory)
            self.source_name, self.is_file = "Camera", False
            self.timer.setInterval(33)
            self.target_fps = 30
            self.restart_camera_button.setEnabled(True)
            # A failed camera stays selected and shows an error, never fake demo video.
            return self._restart_camera()
        elif name == "Video file...":
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                capture.release()
                self.statusBar().showMessage("Could not open that video file.")
                return False
            fps = capture.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / fps) if fps and fps > 1 else 33
        if self.capture is not None:
            self.capture.release()
        self.capture = capture
        self.is_file = name == "Video file..."
        self.source_name = name
        self.timer.setInterval(interval)
        self.target_fps = round(1000 / interval)
        self.camera_error = False
        self.camera_label.clear()
        self.restart_camera_button.setEnabled(False)
        self._reset_stream()
        self._show_detector()
        self.statusBar().showMessage(f"Source: {name}")
        return True

    def _reset_stream(self):
        self.frame_times.clear()
        self.fps_label.setText("FPS: --")
        self.tracks = {}
        self.next_id = 1
        self.last_dets = []
        self._clear_target()
        self.det_label.setText("(none)")
        self.target_label.setText("No target selected.\nClick a box in the video.")

    def _restart_camera(self):
        """UC-10: release the old sensor, reload configuration, probe the new feed."""
        if self.source_name != "Camera":
            return False
        self.capture.release()
        self._reset_stream()
        self.camera_error = False
        try:
            config = load_camera_config(self.camera_config_path)
            self.camera_label.setText(config.label)
            if not self.capture.restart(config):
                self._camera_failed(self.capture.last_error)
                return False
        except (OSError, ValueError) as exc:
            self.camera_label.setText("Camera configuration error")
            self._camera_failed(str(exc))
            return False
        self._show_detector()
        self.statusBar().showMessage(f"{config.label} ready. Select a target to start tracking.")
        return True

    def _camera_failed(self, message):
        self.camera_error = True
        self._reset_stream()
        self.video.clear()
        self.video.setText("Camera unavailable\nReconnect the camera, then press Restart camera.")
        self.statusBar().showMessage(message)

    def _active_detector(self):
        if self.capture is None or self.model_detector is None:
            return self.demo_detector
        return self.model_detector

    def _show_detector(self):
        if self._active_detector() is self.demo_detector:
            text = "Detector: Demo (fake)"
            if self.model_detector is not None:
                text += f"\n{self.model_name} loaded - pick Camera or a video file to use it"
        else:
            text = f"Detector: {self.model_name}"
        self.detector_label.setText(text)

    def _revert_source_box(self):
        self.source_box.blockSignals(True)
        self.source_box.setCurrentText(self.source_name)
        self.source_box.blockSignals(False)

    def _read_frame(self):
        if self.capture is None:
            return self.demo_frame.copy()
        ok, frame = self.capture.read()
        if not ok and self.is_file:      # loop the video when it ends
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok and self.source_name == "Camera" and not self.camera_error:
            self._camera_failed(self.capture.last_error)
        return frame if ok else None

    # ---------- main loop ----------
    def _tick(self):
        frame = self._read_frame()
        if frame is None:
            return
        self._update_fps()
        detections = self._add_ids(self._active_detector().detect(frame), frame.shape[1])
        self.last_dets = detections
        matched = self._update_target(detections, frame.shape[1])
        self._update_state(matched)

        # Parsa's renderer prints each box's "class" text, so the ID is added to it
        # here (for drawing only; the real class name is unchanged).
        labeled = [dict(d, **{"class": f"{d['class']} #{d['id']}"}) for d in detections]
        out = self.renderer.render_hud(frame, labeled)
        if matched is not None:
            x, y, w, h = matched["bbox"]
            cv2.rectangle(out, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 0), 3)
            cv2.putText(out, "TARGET", (x, y + h + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        self.video.show_frame(out)
        self._update_panel(detections, matched, frame.shape)

    def _update_fps(self):
        """Measure how many frames per second are really being shown."""
        self.frame_times.append(time.perf_counter())
        if len(self.frame_times) < 2:
            return
        span = self.frame_times[-1] - self.frame_times[0]
        fps = (len(self.frame_times) - 1) / span if span > 0 else 0.0
        self.fps_label.setText(f"FPS: {fps:.1f} (target {self.target_fps})")
        color = "#16a34a" if fps >= 0.9 * self.target_fps else "#d97706"
        if color != self.fps_color:      # only restyle when the color changes
            self.fps_color = color
            self.fps_label.setStyleSheet(
                f"font-family: Menlo, Consolas, monospace; color: {color};")

    def _add_ids(self, detections, frame_width):
        """Give every detection an ID that stays the same from frame to frame.

        A real tracker would provide these. If a detection already has an "id",
        it is kept; otherwise a simple nearest-box match to the previous frame is used.
        """
        max_dist = MATCH_DISTANCE * frame_width
        for track in self.tracks.values():
            track["age"] += 1
        taken = set()
        result = []
        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy = x + w / 2, y + h / 2
            if "id" in det:
                result.append(det)
                continue
            best_id, best_dist = None, max_dist
            for tid, track in self.tracks.items():
                if tid in taken or track["class"] != det["class"]:
                    continue
                dist = math.hypot(cx - track["center"][0], cy - track["center"][1])
                if dist < best_dist:
                    best_id, best_dist = tid, dist
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            taken.add(best_id)
            self.tracks[best_id] = {"class": det["class"], "center": (cx, cy), "age": 0}
            result.append(dict(det, id=best_id))
        # forget objects that have been gone for a while
        self.tracks = {t: v for t, v in self.tracks.items() if v["age"] <= TRACK_KEEP_FRAMES}
        return result

    def _update_target(self, detections, frame_width):
        """Find the selected target again in this frame (nearest box of the same class)."""
        if self.selected is None:
            return None
        sx, sy = self.selected["center"]
        best, best_dist = None, MATCH_DISTANCE * frame_width
        for det in detections:
            if det["class"] != self.selected["class"]:
                continue
            x, y, w, h = det["bbox"]
            dist = math.hypot(x + w / 2 - sx, y + h / 2 - sy)
            if dist < best_dist:
                best, best_dist = det, dist
        if best is None:
            self.lost_frames += 1
        else:
            self.lost_frames = 0
            x, y, w, h = best["bbox"]
            self.selected["center"] = (x + w / 2, y + h / 2)
            self.selected["id"] = best["id"]
        return best

    def _update_state(self, matched):
        if not self.tracking:
            new = "IDLE"
        elif matched is not None:
            new = "TRACKING"
        elif self.lost_frames > LOST_FRAMES_LIMIT:
            new = "SEARCHING"
        else:
            new = self.state
        if new != self.state:
            self.state = new
            self._show_state()
            if new == "SEARCHING":
                self.statusBar().showMessage("Target lost - searching...")
            elif new == "TRACKING":
                self.statusBar().showMessage("Target acquired - tracking.")

    # ---------- panel ----------
    def _show_state(self):
        self.state_label.setText(self.state)
        self.state_label.setStyleSheet(
            f"background-color: {STATE_COLORS[self.state]}; color: white; "
            "font-size: 22px; font-weight: bold; padding: 10px; border-radius: 6px;")

    def _update_panel(self, detections, matched, shape):
        lines = [f"#{d['id']:<3}{d['class']:<12}{d['confidence']:.2f}" for d in detections]
        self.det_label.setText("\n".join(lines) if lines else "(none)")

        if self.selected is None:
            text = "No target selected.\nClick a box in the video."
        elif matched is None:
            text = f"{self.selected['class']} #{self.selected['id']}\nNot visible right now."
        else:
            x, y, w, h = matched["bbox"]
            frame_h, frame_w = shape[:2]
            dx = int(x + w / 2 - frame_w / 2)
            dy = int(y + h / 2 - frame_h / 2)
            text = (f"{matched['class']} #{matched['id']}\nConfidence: {matched['confidence']:.2f}\n"
                    f"Offset: dx={dx:+d} dy={dy:+d} px\nRange: -- (no LiDAR)")
        self.target_label.setText(text)

    def _refresh_buttons(self):
        self.start_button.setEnabled(self.selected is not None and not self.tracking)
        self.stop_button.setEnabled(self.tracking)

    # ---------- user actions ----------
    def _on_video_clicked(self, fx, fy):
        hits = [d for d in self.last_dets
                if d["bbox"][0] <= fx <= d["bbox"][0] + d["bbox"][2]
                and d["bbox"][1] <= fy <= d["bbox"][1] + d["bbox"][3]]
        if hits:
            det = min(hits, key=lambda d: d["bbox"][2] * d["bbox"][3])  # smallest box wins
            x, y, w, h = det["bbox"]
            self.selected = {"class": det["class"], "center": (x + w / 2, y + h / 2),
                             "id": det["id"]}
            self.lost_frames = 0
            self.statusBar().showMessage(f"Selected {det['class']}. Press Start tracking.")
        elif not self.tracking:
            self.selected = None
        self._refresh_buttons()

    def _on_start(self):
        if self.selected is not None:
            self.tracking = True
            self.state = "SEARCHING"     # becomes TRACKING on the next frame if the target is visible
            self._show_state()
            self._refresh_buttons()

    def _on_stop(self):
        self.tracking = False
        self.state = "IDLE"
        self._show_state()
        self._refresh_buttons()
        self.statusBar().showMessage("Tracking stopped.")

    def _clear_target(self):
        self.selected = None
        self.tracking = False
        self.lost_frames = 0
        self.state = "IDLE"
        self._show_state()
        self._refresh_buttons()

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        super().closeEvent(event)


def main(detector=None, detector_name=None, camera_config_path=None):
    app = QApplication(sys.argv)
    window = MainWindow(detector, detector_name, camera_config_path=camera_config_path)
    window.resize(1240, 720)
    window.show()
    sys.exit(app.exec())
