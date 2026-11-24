"""UC-18: Run a sponsor's tracking program through a documented interface (ES-16.1).

Interface contract for an external tracker:

    class MyTracker:
        def process(self, frame: numpy.ndarray, gimbal: GimbalState) -> TargetPosition | None

The class is named in configuration as ``"package.module:ClassName"`` and
loaded at runtime, so plugging in a sponsor algorithm changes no AimBot code.
``ExternalTrackerBridge`` feeds it each frame plus the current gimbal state,
turns the returned pixel position into the normalized pointing error, and
publishes it through the same sink the built-in micro-tracker uses. The gimbal
control path downstream of that sink is untouched.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class GimbalState:
    pan_deg: float
    tilt_deg: float
    stamp_s: float


@dataclass(frozen=True)
class TargetPosition:
    x_px: float
    y_px: float
    confidence: float = 1.0


class ExternalTracker(Protocol):
    def process(self, frame: np.ndarray, gimbal: GimbalState) -> TargetPosition | None: ...


class PointingErrorSink(Protocol):
    """The existing input of the gimbal control path (the /tracking/pointing_error topic)."""
    def publish(self, dx: float, dy: float, capture_stamp_s: float) -> None: ...


def load_external_tracker(spec: str, **kwargs) -> ExternalTracker:
    """Load ``"package.module:ClassName"`` from configuration and instantiate it."""
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError("tracker spec must look like 'package.module:ClassName'")
    cls = getattr(importlib.import_module(module_name), class_name)
    tracker = cls(**kwargs)
    if not callable(getattr(tracker, "process", None)):
        raise TypeError(f"{spec} has no process(frame, gimbal) method")
    return tracker


class ExternalTrackerBridge:
    def __init__(self, tracker: ExternalTracker, sink: PointingErrorSink,
                 min_confidence: float = 0.3, clock: Callable[[], float] = time.monotonic):
        self.tracker = tracker
        self.sink = sink
        self.min_confidence = min_confidence
        self.clock = clock
        self.latencies_s: list[float] = []

    def on_frame(self, frame: np.ndarray, capture_stamp_s: float, gimbal: GimbalState) -> tuple[float, float] | None:
        target = self.tracker.process(frame, gimbal)
        if target is None or target.confidence < self.min_confidence:
            return None
        height, width = frame.shape[:2]
        dx = (target.x_px - width / 2.0) / (width / 2.0)
        dy = (target.y_px - height / 2.0) / (height / 2.0)
        dx, dy = float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))
        self.sink.publish(dx, dy, capture_stamp_s)
        self.latencies_s.append(self.clock() - capture_stamp_s)
        return dx, dy

    def p95_latency_ms(self) -> float:
        if not self.latencies_s:
            return float("nan")
        return float(np.percentile(self.latencies_s, 95) * 1000)


class BrightestPointTracker:
    """Sample external tracker for testing the interface: follows the brightest blob."""

    def process(self, frame: np.ndarray, gimbal: GimbalState) -> TargetPosition | None:
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame
        y, x = np.unravel_index(int(np.argmax(gray)), gray.shape)
        peak = float(gray[y, x])
        if peak <= 0:
            return None
        return TargetPosition(float(x), float(y), confidence=min(1.0, peak / 255.0))
