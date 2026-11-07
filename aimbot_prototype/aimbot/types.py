"""Shared message/data types. All angles are radians, all times are clock.now() seconds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

BBox = tuple  # (x1, y1, x2, y2) in pixels of the full frame


@dataclass
class Frame:
    frame_id: int
    t_capture: float                 # best estimate of exposure time (host clock)
    image: np.ndarray                # BGR uint8 (H, W, 3)
    sensor: str = "vis"
    t_host: float = 0.0              # when the host received it
    hw_timestamp_ns: Optional[int] = None
    raw: Optional[np.ndarray] = None  # e.g. Bayer mosaic straight from the camera (lossless logging)
    meta: dict = field(default_factory=dict)  # sim ground truth etc.

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass
class Detection:
    bbox: BBox
    cls: str                 # logical class used by the operator ("drone", "ball", ...)
    conf: float
    cls_raw: str = ""        # detector's own class name ("sports ball", ...)
    track_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class MacroResult:
    frame_id: int
    t_capture: float
    detections: list
    infer_ms: float
    roi: Optional[BBox] = None
    t_done: float = 0.0


@dataclass
class GimbalState:
    t: float = 0.0           # time the feedback describes (host clock)
    pan: float = 0.0         # rad, + = right (clockwise seen from above)
    tilt: float = 0.0        # rad, + = up
    pan_rate: float = 0.0
    tilt_rate: float = 0.0
    connected: bool = False
    motors_on: bool = False
    mode: str = ""
    fault: str = ""

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "pan_deg": float(np.degrees(self.pan)),
            "tilt_deg": float(np.degrees(self.tilt)),
            "pan_rate_dps": float(np.degrees(self.pan_rate)),
            "tilt_rate_dps": float(np.degrees(self.tilt_rate)),
            "connected": self.connected,
            "motors_on": self.motors_on,
            "mode": self.mode,
            "fault": self.fault,
        }


@dataclass
class GimbalCommand:
    mode: str                # "angle" | "rate" | "hold"
    pan: float = 0.0         # rad (angle mode) or rad/s (rate mode)
    tilt: float = 0.0
    t: float = 0.0


@dataclass
class RangeReading:
    t: float
    distance_m: float
    strength: int
    temp_c: float
    valid: bool
    raw_cm: int = 0


@dataclass
class TargetMeasurement:
    t_capture: float
    frame_id: int
    u: float                 # target centroid (pixels)
    v: float
    bbox: BBox
    az: float                # absolute target direction (rad, gimbal base frame)
    el: float
    err_az: float            # target minus aim point (rad)
    err_el: float
    dx: float                # FR-08 normalised offset, + = right of aim point, [-1, 1]
    dy: float                # + = above aim point, [-1, 1]
    score: float = 0.0
    source: str = "micro"


@dataclass
class TargetEstimate:
    t: float
    az: float
    el: float
    az_rate: float
    el_rate: float
    valid: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
