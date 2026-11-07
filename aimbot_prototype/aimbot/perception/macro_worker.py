"""Asynchronous macro inference loop (NFR-02): YOLO never blocks the frame-rate tracker.

Takes the newest frame whenever the detector is free (frames in between are
skipped), optionally only a region of interest around the locked target at
native resolution (keeps small/distant targets above YOLO's minimum size), and
hands results to the vision loop through a queue.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

from .. import clock
from ..sensors.camera import FrameBuffer
from ..types import MacroResult
from .detectors import DetectorBase

log = logging.getLogger(__name__)


class MacroWorker:
    def __init__(self, detector: DetectorBase, framebuf: FrameBuffer, max_hz: float = 30.0,
                 roi_provider: Optional[Callable] = None):
        self.detector = detector
        self.framebuf = framebuf
        self.max_hz = max_hz
        self.roi_provider = roi_provider
        self.out: queue.Queue = queue.Queue(maxsize=8)
        self.rate = clock.RateMeter()
        self.enabled = True
        self.infer_ms = 0.0
        self.age_ms = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.errors = 0
        self.last_error = ""

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="macro-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_detector(self, det: DetectorBase) -> None:
        self.detector = det

    def _run(self) -> None:
        last_id = -1
        period = 1.0 / self.max_hz if self.max_hz > 0 else 0.0
        t_next = 0.0
        while not self._stop.is_set():
            if not self.enabled:
                self._stop.wait(0.05)
                continue
            if period:
                clock.sleep_until(t_next, self._stop)
            frame = self.framebuf.wait_newer(last_id, timeout=0.5)
            if frame is None:
                continue
            last_id = frame.frame_id
            t_next = clock.now() + period
            roi = self.roi_provider(frame) if self.roi_provider else None
            try:
                dets = self.detector.detect(frame, roi)
            except Exception as exc:
                self.errors += 1
                self.last_error = repr(exc)
                log.exception("detector failed")
                self._stop.wait(0.2)
                continue
            t_done = clock.now()
            self.infer_ms = self.detector.last_infer_ms
            self.age_ms = (t_done - frame.t_capture) * 1000.0
            res = MacroResult(frame.frame_id, frame.t_capture, dets, self.infer_ms, roi, t_done)
            try:
                self.out.put_nowait(res)
            except queue.Full:
                try:
                    self.out.get_nowait()
                except queue.Empty:
                    pass
                self.out.put_nowait(res)
            self.rate.tick()

    def info(self) -> dict:
        d = self.detector.info()
        d.update({"hz": round(self.rate.value(), 1), "result_age_ms": round(self.age_ms, 1), "errors": self.errors,
                  "enabled": self.enabled})
        return d
