"""PROJECT23-14: camera ingestion shared by the operator UI and future ROS nodes.

The downstream contract is contiguous HxWx3 uint8 BGR, as expected by the
existing RF-DETR detector, DemoDetector and HUDOverlayRenderer. The mock backend
tests this contract only; it does not simulate SWIR optics or validate hardware.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    backend: str = "opencv"
    device: int | str = 0
    spectrum: str = "visible"
    pixel_format: str = "bgr8"
    white_level: int = 65535

    def __post_init__(self):
        if self.backend not in ("opencv", "mock"):
            raise ValueError("backend must be opencv or mock")
        if self.spectrum not in ("visible", "swir"):
            raise ValueError("spectrum must be visible or swir")
        if self.pixel_format not in ("bgr8", "mono8", "mono16"):
            raise ValueError("pixel_format must be bgr8, mono8 or mono16")
        if not ((type(self.device) is int and self.device >= 0)
                or (isinstance(self.device, str) and self.device.strip())):
            raise ValueError("device must be a nonnegative camera index or a device path")
        if type(self.white_level) is not int or not 1 <= self.white_level <= 65535:
            raise ValueError("white_level must be an integer from 1 to 65535")

    @property
    def label(self):
        prefix = "PRELIMINARY MOCK" if self.backend == "mock" else "Camera"
        return f"{prefix}: {self.spectrum.upper()} / {self.pixel_format}"


def load_camera_config(path=None):
    """Read again on restart so camera changes need configuration, not code edits."""
    if path is None:
        return CameraConfig()
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Camera configuration must be a JSON object")
    try:
        return CameraConfig(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid camera configuration: {exc}") from exc


def normalize_frame(frame, config):
    """Convert decoded BGR/monochrome frames without changing their dimensions.

    mono16 assumes unpacked, right-aligned samples. white_level is fixed by the
    sensor profile (4095 for 12-bit, 65535 for 16-bit), not stretched per frame.
    Packed pixels, Bayer data and vendor transports need a driver adapter first.
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Camera returned an empty frame")
    if config.pixel_format == "bgr8":
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("bgr8 requires an HxWx3 uint8 frame")
        return np.ascontiguousarray(frame)

    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    dtype = np.uint16 if config.pixel_format == "mono16" else np.uint8
    if frame.ndim != 2 or frame.dtype != dtype:
        raise ValueError(f"{config.pixel_format} requires a single-channel {np.dtype(dtype)} frame")
    if config.pixel_format == "mono16":
        frame = np.rint(np.clip(frame.astype(np.float32), 0, config.white_level)
                        * (255.0 / config.white_level)).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_GRAY2BGR)


class MockCamera:
    """Synthetic decoded camera frames for preliminary tests, never real sensor data."""
    def __init__(self, config):
        self.config = config
        self.opened = True
        self.sequence = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.opened:
            return False, None
        self.sequence += 1
        gray = np.tile(np.linspace(20, 180, 640, dtype=np.uint8), (480, 1))
        x = 40 + self.sequence % 400
        cv2.rectangle(gray, (x, 180), (x + 60, 220), 240, -1)
        cv2.putText(gray, "PRELIMINARY MOCK - hardware untested", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        if self.config.pixel_format == "mono16":
            frame = np.rint(gray.astype(np.float32) * (self.config.white_level / 255)).astype(np.uint16)
        elif self.config.pixel_format == "mono8":
            frame = gray
        else:
            frame = np.stack((gray, gray // 2, gray // 3), axis=2)
        return True, frame

    def release(self):
        self.opened = False


def open_capture(config):
    if config.backend == "mock":
        return MockCamera(config)
    capture = cv2.VideoCapture(config.device)
    # Request unconverted decoded pixels. Support depends on the camera/backend;
    # normalize_frame rejects incompatible output instead of displaying bad data.
    try:
        if config.pixel_format != "bgr8":
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    except cv2.error:
        capture.release()
        raise
    return capture


class CameraSource:
    """Restartable ingestion with the read/release interface already used by the UI.

    capture_factory permits simulated disconnects in tests without replacing the
    ingestion, detector, tracking or rendering code under test.
    """
    def __init__(self, config=None, capture_factory=None):
        self.config = config or CameraConfig()
        self.capture_factory = capture_factory or open_capture
        self._capture = None
        self._first_frame = None
        self.last_error = "Camera has not been started"

    def restart(self, config=None):
        # Close the previous device before opening its replacement (exclusive handles).
        self.release()
        if config is not None:
            self.config = config
        try:
            self._capture = self.capture_factory(self.config)
            if not self._capture.isOpened():
                raise OSError(f"Could not open camera device {self.config.device}")
            ok, frame = self._capture.read()
            if not ok:
                raise OSError("Camera opened but did not supply a frame")
            self._first_frame = normalize_frame(frame, self.config)
        except (OSError, ValueError, cv2.error) as exc:
            self._fail(str(exc))
            return False
        self.last_error = ""
        return True

    def isOpened(self):
        return self._capture is not None and self._capture.isOpened()

    def read(self):
        if self._first_frame is not None:
            frame, self._first_frame = self._first_frame, None
            return True, frame
        if self._capture is None:
            return False, None
        try:
            ok, frame = self._capture.read()
            if not ok:
                raise OSError("Camera disconnected or no frame received; reconnect and restart camera")
            return True, normalize_frame(frame, self.config)
        except (OSError, ValueError, cv2.error) as exc:
            self._fail(str(exc))
            return False, None

    def _fail(self, message):
        self.release()
        self.last_error = message

    def release(self):
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._first_frame = None
