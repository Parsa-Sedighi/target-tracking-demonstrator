"""Kalman filter on the target's absolute direction (az, el) - TR-3 mitigation.

Measurements come from camera frames (timestamped at capture) and are combined
with the gimbal angles at that same instant, so the estimate is independent of
where the gimbal happens to point. ``predict_to(t)`` lets the 100-200 Hz control
loop act on where the target *is now* (plus actuation lead), not where it was
when the frame was exposed - this is what stops the gimbal from lagging or
overshooting a falling ball.

Reference implementation for the ECE team's fusion node (AIM-115); same math
ports directly to C++/Eigen.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..perception.geometry import wrap_pi
from ..types import TargetEstimate


class AxisKF:
    """1-D constant-velocity (cv) or constant-acceleration (ca) filter, white-noise driven."""

    def __init__(self, model: str = "cv", q: float = 5.0):
        self.model = model
        self.n = 2 if model == "cv" else 3
        self.q = q
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

    def F(self, dt: float) -> np.ndarray:
        if self.n == 2:
            return np.array([[1.0, dt], [0.0, 1.0]])
        return np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])

    def Q(self, dt: float) -> np.ndarray:
        q = self.q
        if self.n == 2:
            return q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        return q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                             [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                             [dt ** 3 / 6, dt ** 2 / 2, dt]])

    def init(self, z: float, r: float, rate_var: float, acc_var: float = 50.0) -> None:
        self.x = np.zeros(self.n)
        self.x[0] = z
        self.P = np.diag([r, rate_var] + ([acc_var] if self.n == 3 else []))

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = self.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q(dt)

    def innovation(self, z: float, r: float) -> tuple[float, float]:
        return z - self.x[0], self.P[0, 0] + r

    def update(self, z: float, r: float) -> None:
        y, S = self.innovation(z, r)
        K = self.P[:, 0] / S
        self.x = self.x + K * y
        self.P = self.P - np.outer(K, self.P[0, :])

    def predicted(self, dt: float) -> np.ndarray:
        return self.F(max(0.0, dt)) @ self.x


class TargetKF:
    def __init__(self, cfg: Optional[dict] = None):
        c = cfg or {}
        self.model = str(c.get("model", "cv"))
        self.q = float(c.get("q", 2.0))
        self.gate = float(c.get("gate_chi2", 13.8))          # 2-dof, 99.9 %
        self.max_rejects = int(c.get("max_consecutive_rejects", 6))
        self.rate_var0 = float(c.get("initial_rate_var", 4.0))
        self.max_coast = float(c.get("max_predict_s", 0.6))
        self.reset()

    def reset(self) -> None:
        self.az = AxisKF(self.model, self.q)
        self.el = AxisKF(self.model, self.q)
        self.t: Optional[float] = None
        self.rejects = 0
        self.updates = 0
        self.last_nis = 0.0

    @property
    def initialized(self) -> bool:
        return self.t is not None

    def update(self, t: float, az: float, el: float, r: float) -> bool:
        if self.t is None:
            self.az.init(az, r, self.rate_var0)
            self.el.init(el, r, self.rate_var0)
            self.t = t
            self.updates = 1
            return True
        dt = t - self.t
        if dt < 0:
            return False   # stale (older than the state): ignore
        self.az.predict(dt)
        self.el.predict(dt)
        self.t = t
        z_az = self.az.x[0] + wrap_pi(az - self.az.x[0])
        y1, s1 = self.az.innovation(z_az, r)
        y2, s2 = self.el.innovation(el, r)
        nis = y1 * y1 / s1 + y2 * y2 / s2
        self.last_nis = nis
        if nis > self.gate:
            self.rejects += 1
            if self.rejects <= self.max_rejects:
                return False
            # target really jumped (re-lock / big manoeuvre): re-initialise on it
            self.az.init(az, r, self.rate_var0)
            self.el.init(el, r, self.rate_var0)
            self.rejects = 0
            return True
        self.rejects = 0
        self.az.update(z_az, r)
        self.el.update(el, r)
        self.updates += 1
        return True

    def shift(self, d_az: float, d_el: float) -> None:
        """Apply a position correction (e.g. the micro tracker was re-anchored to a detection).
        Velocity is kept; position uncertainty is not reduced."""
        if self.t is None:
            return
        self.az.x[0] += d_az
        self.el.x[0] += d_el
        self.rejects = 0

    def predict_to(self, t: float) -> Optional[TargetEstimate]:
        if self.t is None:
            return None
        dt = t - self.t
        valid = dt <= self.max_coast
        dt = min(max(dt, 0.0), self.max_coast)
        a = self.az.predicted(dt)
        e = self.el.predicted(dt)
        return TargetEstimate(t, float(a[0]), float(e[0]), float(a[1]), float(e[1]), valid,
                              {"sigma_az": math.sqrt(max(0.0, self.az.P[0, 0])),
                               "sigma_el": math.sqrt(max(0.0, self.el.P[0, 0])), "age": t - self.t})
