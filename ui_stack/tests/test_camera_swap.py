"""UC-10 / PROJECT23-14 preliminary UI integration tests using injected cameras.

Runs the existing DemoDetector, target selection/state logic and HUD renderer.
No physical camera, Jetson, ROS transport, RF-DETR weights or SWIR optics are tested.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ui_stack" / "src" / "display_gui"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "integration_stack" / "sensor_interface"))
from demo_detector import DemoDetector
from main_window import MainWindow


@pytest.fixture(scope="session")
def app(record_testsuite_property):
    record_testsuite_property("jira_issue", "PROJECT23-14")
    record_testsuite_property("test_stage", "PRELIMINARY MOCK TEST")
    record_testsuite_property("real_hardware_tested", "false")
    record_testsuite_property("trained_model_accuracy_tested", "false")
    return QApplication.instance() or QApplication([])


class InjectedCamera:
    def __init__(self, frame):
        self.frame = frame
        self.connected = True
        self.released = False

    def isOpened(self):
        return self.connected and not self.released

    def read(self):
        return (True, self.frame.copy()) if self.isOpened() else (False, None)

    def release(self):
        self.released = True


class CheckedDemoDetector(DemoDetector):
    """Reuse Saam's detector while asserting the input expected by RF-DETR/HUD."""
    def __init__(self):
        super().__init__()
        self.shapes = []

    def detect(self, frame):
        assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
        assert frame.flags.c_contiguous
        self.shapes.append(frame.shape)
        return super().detect(frame)


def write_profile(path, spectrum="visible"):
    path.write_text(json.dumps({
        "backend": "mock", "spectrum": spectrum,
        "pixel_format": "bgr8" if spectrum == "visible" else "mono16",
        "white_level": 4095,
    }))


def select_and_track(window):
    # The existing demo sports ball never disappears, unlike its drone.
    target = next(d for d in window.last_dets if d["class"] == "sports ball")
    x, y, w, h = target["bbox"]
    window._on_video_clicked(x + w // 2, y + h // 2)
    window.start_button.click()
    window._tick()
    assert window.state == "TRACKING"
    assert window.selected is not None
    assert not window.video.pixmap().isNull()


def test_preliminary_visible_disconnect_swir_restart(app, tmp_path):
    profile = tmp_path / "camera.json"
    write_profile(profile)
    visible = InjectedCamera(np.full((480, 640, 3), 90, dtype=np.uint8))
    swir = InjectedCamera(np.full((240, 320), 2048, dtype=np.uint16))
    opened = []

    def factory(config):
        if opened:
            assert opened[-1].released
        capture = visible if config.spectrum == "visible" else swir
        opened.append(capture)
        return capture

    detector = CheckedDemoDetector()
    window = MainWindow(detector, "DemoDetector (preliminary)", profile, factory)
    window.timer.stop()
    try:
        window._tick()
        select_and_track(window)
        assert detector.shapes[-1] == (480, 640, 3)
        count = len(detector.shapes)

        visible.connected = False
        window._tick()
        assert window.camera_error and window.state == "IDLE"
        assert window.selected is None and not window.last_dets and not window.tracks
        assert not window.start_button.isEnabled()
        assert "Camera unavailable" in window.video.text()
        window._tick()
        assert len(detector.shapes) == count  # No stale frames or fake Demo fallback.

        write_profile(profile, "swir")  # Swap the injected sensor by config only.
        window.restart_camera_button.click()
        assert not window.camera_error and visible.released
        assert window.state == "IDLE" and window.selected is None
        window._tick()
        select_and_track(window)
        assert detector.shapes[-1] == (240, 320, 3)
        assert window.model_detector is detector  # Same detector, no rewritten tracking path.
        assert "PRELIMINARY MOCK: SWIR" in window.camera_label.text()
    finally:
        window.close()
    assert swir.released


def test_camera_open_failure_does_not_fall_back_to_fake_demo(app, tmp_path):
    profile = tmp_path / "camera.json"
    write_profile(profile)
    camera = InjectedCamera(np.zeros((10, 10, 3), dtype=np.uint8))
    camera.connected = False
    detector = CheckedDemoDetector()
    window = MainWindow(detector, camera_config_path=profile,
                        camera_capture_factory=lambda config: camera)
    window.timer.stop()
    try:
        window._tick()
        assert window.source_name == window.source_box.currentText() == "Camera"
        assert window.camera_error and camera.released
        assert not detector.shapes and not window.last_dets
        assert window.restart_camera_button.isEnabled()
    finally:
        window.close()


def test_restart_recovers_after_replacement_camera_was_initially_missing(app, tmp_path):
    profile = tmp_path / "camera.json"
    write_profile(profile, "swir")
    missing = InjectedCamera(np.zeros((240, 320), dtype=np.uint16))
    missing.connected = False
    connected = InjectedCamera(np.full((240, 320), 2048, dtype=np.uint16))
    captures = iter([missing, connected])
    window = MainWindow(CheckedDemoDetector(), camera_config_path=profile,
                        camera_capture_factory=lambda config: next(captures))
    window.timer.stop()
    try:
        assert window.camera_error and missing.released
        window.restart_camera_button.click()
        assert not window.camera_error
        window._tick()
        select_and_track(window)
    finally:
        window.close()
    assert connected.released


def test_bad_profile_on_restart_clears_tracking_and_closes_camera(app, tmp_path):
    profile = tmp_path / "camera.json"
    write_profile(profile)
    camera = InjectedCamera(np.zeros((480, 640, 3), dtype=np.uint8))
    window = MainWindow(CheckedDemoDetector(), camera_config_path=profile,
                        camera_capture_factory=lambda config: camera)
    window.timer.stop()
    try:
        window._tick()
        select_and_track(window)
        profile.write_text('{"pixel_format": "packed12"}')
        window.restart_camera_button.click()
        assert camera.released and window.camera_error
        assert window.state == "IDLE" and window.selected is None
        assert not window.last_dets and "pixel_format" in window.statusBar().currentMessage()
    finally:
        window.close()


def test_existing_demo_and_video_paths_still_render(app, monkeypatch):
    import main_window

    video = InjectedCamera(np.zeros((480, 640, 3), dtype=np.uint8))
    video.get = lambda prop: 30.0
    monkeypatch.setattr(main_window.cv2, "VideoCapture", lambda path: video)
    window = MainWindow()
    window.timer.stop()
    try:
        window._tick()
        assert window.source_name == "Demo" and window.last_dets
        assert window._set_source("Video file...", "injected.mp4")
        window._tick()
        select_and_track(window)
        assert not window.restart_camera_button.isEnabled()
        assert window._set_source("Demo") and video.released
        window._tick()
        assert window.last_dets and not window.video.pixmap().isNull()
    finally:
        window.close()


def test_launcher_exposes_camera_profile_without_loading_model():
    result = subprocess.run([sys.executable, str(ROOT / "ui_stack" / "run_ui.py"), "--help"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--camera-config" in result.stdout and "--demo" in result.stdout
