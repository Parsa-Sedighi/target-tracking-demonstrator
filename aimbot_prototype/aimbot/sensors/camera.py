"""Camera ingestion layer (FR-10 / NFR-01 sensor abstraction).

Every camera backend runs its own capture thread and pushes ``Frame`` objects
into a shared ``FrameBuffer``. Downstream code never knows which sensor is
mounted - it only sees frames + the sensor's intrinsics, so VIS and SWIR (or a
webcam, a video file, the simulator) are interchangeable at runtime.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from typing import Callable, Optional

import cv2
import numpy as np

from .. import clock
from ..perception.geometry import Intrinsics
from ..types import Frame

log = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe 'latest frame' slot plus a short history for re-seeding the tracker."""

    def __init__(self, history: int = 16):
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._hist: deque[Frame] = deque(maxlen=max(2, history))
        self._next_id = 0
        self.listeners: list = []     # called in the camera thread for every frame (e.g. the logger)

    def new_id(self) -> int:
        with self._cond:
            i = self._next_id
            self._next_id += 1
            return i

    def put(self, frame: Frame) -> None:
        with self._cond:
            self._latest = frame
            self._hist.append(frame)
            self._cond.notify_all()
        for cb in self.listeners:
            try:
                cb(frame)
            except Exception:
                log.exception("frame listener failed")

    def latest(self) -> Optional[Frame]:
        return self._latest

    def wait_newer(self, after_id: int, timeout: float = 0.5) -> Optional[Frame]:
        with self._cond:
            ok = self._cond.wait_for(lambda: self._latest is not None and self._latest.frame_id > after_id, timeout)
            return self._latest if ok else None

    def get(self, frame_id: int) -> Optional[Frame]:
        with self._cond:
            for f in reversed(self._hist):
                if f.frame_id == frame_id:
                    return f
        return None

    def frames_between(self, after_id: int, up_to_id: int) -> list[Frame]:
        with self._cond:
            return [f for f in self._hist if after_id < f.frame_id <= up_to_id]

    def clear(self) -> None:
        with self._cond:
            self._hist.clear()
            self._latest = None


class CameraBase:
    """Capture thread skeleton. Subclasses implement open() / grab() / close()."""

    kind = "base"

    def __init__(self, name: str, cfg: dict, framebuf: FrameBuffer):
        self.name = name
        self.cfg = cfg or {}
        self.framebuf = framebuf
        self.intr: Optional[Intrinsics] = None
        self.fps_meter = clock.RateMeter()
        self.errors = 0
        self.last_error = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.opened = False

    # --- to implement
    def open(self) -> None:
        raise NotImplementedError

    def grab(self):
        """Block until a frame is available. Return (image_bgr, t_capture, extras_dict) or None."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    # --- common
    def start(self) -> None:
        self.open()
        self.opened = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.close()
        finally:
            self.opened = False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                res = self.grab()
            except Exception as exc:  # keep the thread alive; report in self-test/UI
                self.errors += 1
                self.last_error = repr(exc)
                log.warning("camera %s grab error: %s", self.name, exc)
                time.sleep(0.05)
                continue
            if res is None:
                continue
            image, t_capture, extras = res
            fr = Frame(frame_id=self.framebuf.new_id(), t_capture=t_capture, image=image, sensor=self.name,
                       t_host=clock.now(), hw_timestamp_ns=extras.get("hw_ts"), raw=extras.get("raw"),
                       meta=extras.get("meta", {}))
            self.framebuf.put(fr)
            self.fps_meter.tick()

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "fps": round(self.fps_meter.value(), 1),
                "errors": self.errors, "last_error": self.last_error,
                "intrinsics": self.intr.as_dict() if self.intr else None}


# ----------------------------------------------------------------------------- simulator
class SimCamera(CameraBase):
    """Renders the SimWorld at the *true* gimbal angles (closed loop)."""

    kind = "sim"

    def __init__(self, name: str, cfg: dict, framebuf: FrameBuffer, world, true_angles: Callable[[float], tuple]):
        super().__init__(name, cfg, framebuf)
        self.world = world
        self.true_angles = true_angles
        self.fps = float(cfg.get("fps", 60))
        self.latency = float(cfg.get("latency_ms", 8.0)) / 1000.0
        self.intr = world.intr
        self._t_next = 0.0

    def open(self) -> None:
        self._t_next = clock.now()

    def grab(self):
        t = self._t_next
        clock.sleep_until(t, self._stop)
        now = clock.now()
        if now - t > 0.25:  # fell badly behind (debugger, sleep) - resync
            t = now
            self._t_next = now
        self._t_next += 1.0 / self.fps
        pan, tilt = self.true_angles(t)
        img, truth = self.world.render(t, pan, tilt)
        clock.sleep_until(t + self.latency, self._stop)  # exposure + USB transfer delay
        return img, t, {"meta": {"truth": truth, "true_pan": pan, "true_tilt": tilt}}


# ----------------------------------------------------------------------------- OpenCV / V4L2 / file
class OpenCVCamera(CameraBase):
    """Webcams, generic USB3 UVC / V4L2 sensors (VIS or SWIR) and recorded video files."""

    kind = "opencv"

    def __init__(self, name: str, cfg: dict, framebuf: FrameBuffer):
        super().__init__(name, cfg, framebuf)
        self.source = cfg.get("source", 0)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_file = False
        self._file_period = 0.0
        self._t_next = 0.0

    def _api(self) -> int:
        api = str(self.cfg.get("api", "auto")).lower()
        table = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2,
                 "gstreamer": cv2.CAP_GSTREAMER, "ffmpeg": cv2.CAP_FFMPEG, "any": cv2.CAP_ANY}
        if api in table:
            return table[api]
        if sys.platform.startswith("win") and isinstance(self.source, int):
            return cv2.CAP_DSHOW  # opens fast on Windows and honours resolution requests
        if sys.platform.startswith("linux") and isinstance(self.source, int):
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    def open(self) -> None:
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.is_file = isinstance(src, str) and not src.startswith(("rtsp:", "http:", "https:", "v4l2src", "nvarguscamerasrc"))
        self.cap = cv2.VideoCapture(src, self._api())
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {self.source!r}")
        if not self.is_file:
            if self.cfg.get("fourcc"):
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(self.cfg["fourcc"])))
            if self.cfg.get("width"):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg["width"]))
            if self.cfg.get("height"):
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg["height"]))
            if self.cfg.get("fps"):
                self.cap.set(cv2.CAP_PROP_FPS, float(self.cfg["fps"]))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # lowest latency where supported
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._file_period = 1.0 / (float(self.cfg.get("fps") or fps))
            self._t_next = clock.now()
        ok, img = self.cap.read()
        if not ok or img is None:
            raise RuntimeError(f"video source {self.source!r} opened but returned no frames")
        if self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # replay from frame 0 (frame-exact for recorded tests)
        img = self._to_bgr(img)
        h, w = img.shape[:2]
        self.intr = Intrinsics.from_config(self.cfg.get("intrinsics", {}), w, h)
        log.info("camera %s opened: %dx%d, HFOV %.1f deg", self.name, w, h, self.intr.hfov_deg)

    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:                  # mono / SWIR
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if self.cfg.get("flip") in ("h", "horizontal", True):
            img = cv2.flip(img, 1)
        return img

    def grab(self):
        assert self.cap is not None
        if self.is_file:
            clock.sleep_until(self._t_next, self._stop)
            self._t_next += self._file_period
        ok, img = self.cap.read()
        t = clock.now()
        if not ok or img is None:
            if self.is_file and self.cfg.get("loop", True):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None
            time.sleep(0.01)
            return None
        # USB transfer + exposure latency: subtract a configurable estimate
        t_capture = t - float(self.cfg.get("latency_ms", 0.0)) / 1000.0
        return self._to_bgr(img), t_capture, {}

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def create_camera(name: str, cfg: dict, framebuf: FrameBuffer, world=None, true_angles=None) -> CameraBase:
    backend = str(cfg.get("backend", "sim")).lower()
    if backend == "sim":
        if world is None:
            raise RuntimeError("sim camera requires the simulator world")
        return SimCamera(name, cfg, framebuf, world, true_angles)
    if backend in ("opencv", "webcam", "v4l2", "file", "uvc"):
        return OpenCVCamera(name, cfg, framebuf)
    if backend in ("spinnaker", "flir", "blackfly"):
        from .spinnaker_camera import SpinnakerCamera

        return SpinnakerCamera(name, cfg, framebuf)
    raise ValueError(f"unknown camera backend {backend!r}")
