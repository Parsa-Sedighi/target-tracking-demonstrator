"""UC-15: Manual gimbal jog within travel limits (ES-10.1, ES-04.3, ES-11.2).

The operator can step the pan and tilt axes from the UI without tracking.
Every jog target is clamped to the software travel limits before it is sent,
so no jog command can drive an axis past its limit (ES-11.2).

Default limits: pan +/-325 deg (Gremsy Mio pan range) and tilt -45 to +90 deg
(ES-04.2). Allowed step sizes are fixed so the UI can offer them as buttons.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Axis(str, Enum):
    PAN = "pan"
    TILT = "tilt"


@dataclass(frozen=True)
class TravelLimits:
    pan_min: float = -325.0
    pan_max: float = 325.0
    tilt_min: float = -45.0
    tilt_max: float = 90.0

    def clamp(self, axis: Axis, angle: float) -> float:
        lo, hi = (self.pan_min, self.pan_max) if axis == Axis.PAN else (self.tilt_min, self.tilt_max)
        return min(max(angle, lo), hi)


class GimbalPort(Protocol):
    def position(self) -> tuple[float, float]: ...
    def move_to(self, pan_deg: float, tilt_deg: float) -> None: ...


ALLOWED_STEPS_DEG = (0.1, 0.5, 1.0, 5.0, 10.0)


@dataclass(frozen=True)
class JogResult:
    axis: Axis
    requested_step: float
    commanded: tuple[float, float]
    clamped: bool


class JogController:
    def __init__(self, gimbal: GimbalPort, limits: TravelLimits = TravelLimits()):
        self.gimbal = gimbal
        self.limits = limits

    def jog(self, axis: Axis, direction: int, step_deg: float) -> JogResult:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if step_deg not in ALLOWED_STEPS_DEG:
            raise ValueError(f"step must be one of {ALLOWED_STEPS_DEG}")
        pan, tilt = self.gimbal.position()
        if axis == Axis.PAN:
            wanted = pan + direction * step_deg
            target = (self.limits.clamp(Axis.PAN, wanted), self.limits.clamp(Axis.TILT, tilt))
        else:
            wanted = tilt + direction * step_deg
            target = (self.limits.clamp(Axis.PAN, pan), self.limits.clamp(Axis.TILT, wanted))
        clamped = target[0 if axis == Axis.PAN else 1] != wanted
        self.gimbal.move_to(*target)
        return JogResult(axis, direction * step_deg, target, clamped)
