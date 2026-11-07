"""Reference pointing controller (ECE subsystem placeholder, AIM-102).

angle mode (default): command the gimbal to where the Kalman filter says the target
  will be ``lead_s`` from now. The lead covers command transport + the gimbal's own
  position-loop lag (~1/K). If the command stream ever stops, the gimbal simply holds
  its last angle - inherently fail-safe.
rate mode: omega = ff_gain * target_rate + kp * error + ki * integral(error), with rate
  and acceleration limits - the PID/feedforward form described in the tasking doc.

The ECE team can replace this with their own node: set control.source = external and
publish rate commands on /aimbot/gimbal/cmd_rate_external (ROS 2 bridge).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from ..perception.geometry import wrap_pi
from ..types import GimbalCommand, GimbalState, TargetEstimate


class TrackingController:
    def __init__(self, cfg: Optional[dict] = None, max_rate: float = math.radians(100)):
        c = cfg or {}
        self.mode = str(c.get("mode", "angle"))
        self.lead = float(c.get("lead_ms", 45.0)) / 1000.0
        self.kp = float(c.get("kp", 6.0))
        self.ki = float(c.get("ki", 0.5))
        self.ff = float(c.get("ff_gain", 1.0))
        self.max_rate = min(max_rate, math.radians(float(c.get("max_rate_dps", 100.0))))
        self.max_acc = math.radians(float(c.get("max_acc_dps2", 900.0)))
        self.deadband = math.radians(float(c.get("deadband_deg", 0.0)))
        self.reset()

    def reset(self) -> None:
        self._i = [0.0, 0.0]
        self._prev_t: Optional[float] = None
        self._prev_sp: Optional[tuple[float, float]] = None
        self._prev_rate = (0.0, 0.0)

    def compute(self, t: float, predict: Callable[[float], Optional[TargetEstimate]], g: GimbalState,
                aim_offset: tuple[float, float]) -> Optional[GimbalCommand]:
        est = predict(t + self.lead)
        if est is None or not est.valid:
            return None
        dt = 0.01 if self._prev_t is None else max(1e-4, min(0.1, t - self._prev_t))
        self._prev_t = t
        # gimbal pointing that puts the aim point (boresight/LiDAR axis) on the target
        pan_sp = est.az - aim_offset[0]
        tilt_sp = est.el - aim_offset[1]
        if self.mode == "angle":
            if self._prev_sp is None:
                self._prev_sp = (g.pan, g.tilt)
            step = self.max_rate * dt
            pp, pt = self._prev_sp
            dp = wrap_pi(pan_sp - pp)
            pan_sp = pp + max(-step, min(step, dp))
            tilt_sp = pt + max(-step, min(step, tilt_sp - pt))
            self._prev_sp = (pan_sp, tilt_sp)
            return GimbalCommand("angle", pan_sp, tilt_sp)
        # rate mode: feedback on the current (latency-corrected) gimbal angle
        age = max(0.0, min(0.1, t - g.t))
        cur_pan = g.pan + g.pan_rate * age
        cur_tilt = g.tilt + g.tilt_rate * age
        e = (wrap_pi(pan_sp - cur_pan), tilt_sp - cur_tilt)
        out = []
        for i, (err, rate_ff) in enumerate(zip(e, (est.az_rate, est.el_rate))):
            if abs(err) < self.deadband:
                err = 0.0
            self._i[i] = max(-0.2, min(0.2, self._i[i] + err * dt))
            w = self.ff * rate_ff + self.kp * err + self.ki * self._i[i]
            w = max(-self.max_rate, min(self.max_rate, w))
            dw = max(-self.max_acc * dt, min(self.max_acc * dt, w - self._prev_rate[i]))
            out.append(self._prev_rate[i] + dw)
        self._prev_rate = (out[0], out[1])
        return GimbalCommand("rate", out[0], out[1])

    def info(self) -> dict:
        return {"mode": self.mode, "lead_ms": self.lead * 1000, "kp": self.kp, "ki": self.ki, "ff_gain": self.ff}
