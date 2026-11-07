"""Gimbal abstraction: soft limits, calibration zero offsets, feedback history, watchdog.

Conventions (match MAVLink): pan/yaw + = right (clockwise seen from above),
tilt/pitch + = up. Radians everywhere inside the engine.

Backends implement the ``_send_*`` hooks and call ``_on_feedback`` whenever
encoder/attitude data arrives.
"""

from __future__ import annotations

import bisect
import logging
import math
import threading
from collections import deque
from typing import Optional

from .. import clock
from ..types import GimbalCommand, GimbalState

log = logging.getLogger(__name__)


class GimbalBase:
    kind = "base"
    supports_rate = True
    supports_angle = True

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        lim = self.cfg.get("limits_deg", {}) or {}
        self.pan_min = math.radians(float(lim.get("pan_min", -170.0)))
        self.pan_max = math.radians(float(lim.get("pan_max", 170.0)))
        self.tilt_min = math.radians(float(lim.get("tilt_min", -45.0)))
        self.tilt_max = math.radians(float(lim.get("tilt_max", 90.0)))
        self.max_rate = math.radians(float(self.cfg.get("max_rate_dps", 100.0)))
        self.zero_pan = math.radians(float(self.cfg.get("zero_pan_deg", 0.0)))
        self.zero_tilt = math.radians(float(self.cfg.get("zero_tilt_deg", 0.0)))
        self.invert_pan = bool(self.cfg.get("invert_pan", False))
        self.invert_tilt = bool(self.cfg.get("invert_tilt", False))
        self._lock = threading.Lock()
        self._state = GimbalState()
        self._hist_t: deque[float] = deque(maxlen=600)
        self._hist: deque[tuple[float, float, float, float]] = deque(maxlen=600)
        self.fb_rate = clock.RateMeter()
        self.last_cmd: Optional[GimbalCommand] = None
        self.cmd_rate = clock.RateMeter()
        self.estopped = False

    # ------------------------------------------------------------------ backend hooks
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _send_angles(self, pan: float, tilt: float) -> None:
        raise NotImplementedError

    def _send_rates(self, pan_rate: float, tilt_rate: float) -> None:
        raise NotImplementedError

    def _set_motors(self, on: bool) -> None:
        pass

    # ------------------------------------------------------------------ sign/zero mapping
    def _to_raw(self, pan: float, tilt: float) -> tuple[float, float]:
        p = pan + self.zero_pan
        t = tilt + self.zero_tilt
        return (-p if self.invert_pan else p), (-t if self.invert_tilt else t)

    def _from_raw(self, pan_raw: float, tilt_raw: float) -> tuple[float, float]:
        p = -pan_raw if self.invert_pan else pan_raw
        t = -tilt_raw if self.invert_tilt else tilt_raw
        return p - self.zero_pan, t - self.zero_tilt

    def _rate_sign(self, pr: float, tr: float) -> tuple[float, float]:
        return (-pr if self.invert_pan else pr), (-tr if self.invert_tilt else tr)

    # ------------------------------------------------------------------ public API
    def clamp_angles(self, pan: float, tilt: float) -> tuple[float, float]:
        return min(self.pan_max, max(self.pan_min, pan)), min(self.tilt_max, max(self.tilt_min, tilt))

    def command(self, cmd: GimbalCommand) -> None:
        if self.estopped:
            return
        cmd.t = clock.now()
        if cmd.mode == "angle":
            pan, tilt = self.clamp_angles(cmd.pan, cmd.tilt)
            self._send_angles(*self._to_raw(pan, tilt))
            cmd = GimbalCommand("angle", pan, tilt, cmd.t)
        elif cmd.mode == "rate":
            st = self.state()
            pr = max(-self.max_rate, min(self.max_rate, cmd.pan))
            tr = max(-self.max_rate, min(self.max_rate, cmd.tilt))
            # soft limits: never drive further into a limit (ES-11.2)
            if (st.pan >= self.pan_max and pr > 0) or (st.pan <= self.pan_min and pr < 0):
                pr = 0.0
            if (st.tilt >= self.tilt_max and tr > 0) or (st.tilt <= self.tilt_min and tr < 0):
                tr = 0.0
            self._send_rates(*self._rate_sign(pr, tr))
            cmd = GimbalCommand("rate", pr, tr, cmd.t)
        else:  # hold
            self.hold()
            return
        self.last_cmd = cmd
        self.cmd_rate.tick()

    def hold(self) -> None:
        """Stop moving and keep the current pointing."""
        if self.estopped:
            return
        st = self.state()
        if self.supports_angle and self.cfg.get("command_mode", "angle") == "angle":
            self._send_angles(*self._to_raw(*self.clamp_angles(st.pan, st.tilt)))
            self.last_cmd = GimbalCommand("hold", st.pan, st.tilt, clock.now())
        else:
            self._send_rates(0.0, 0.0)
            self.last_cmd = GimbalCommand("hold", 0.0, 0.0, clock.now())

    def home(self) -> None:
        self.command(GimbalCommand("angle", 0.0, 0.0))

    def motors(self, on: bool) -> None:
        self._set_motors(on)

    def estop(self) -> None:
        """Software emergency halt: zero motion, motors off where supported, latch."""
        try:
            if self.supports_rate:
                self._send_rates(0.0, 0.0)
            self._set_motors(False)
        finally:
            self.estopped = True

    def reset_estop(self) -> None:
        self.estopped = False
        self._set_motors(True)

    def zero_here(self) -> tuple[float, float]:
        """Calibration: define the current pointing as (0, 0). Returns the new zero offsets (rad)."""
        st = self.state()
        self.zero_pan += st.pan
        self.zero_tilt += st.tilt
        with self._lock:
            self._hist.clear()
            self._hist_t.clear()
            self._state.pan -= st.pan
            self._state.tilt -= st.tilt
        return self.zero_pan, self.zero_tilt

    def raw_angles(self) -> tuple[float, float]:
        st = self.state()
        return self._to_raw(st.pan, st.tilt)

    # ------------------------------------------------------------------ feedback
    def _on_feedback(self, t: float, pan_raw: float, tilt_raw: float, pan_rate: float = math.nan,
                     tilt_rate: float = math.nan, **extra) -> None:
        pan, tilt = self._from_raw(pan_raw, tilt_raw)
        pr, tr = self._rate_sign(pan_rate, tilt_rate)
        with self._lock:
            prev = self._hist[-1] if self._hist else None
            if (math.isnan(pr) or math.isnan(tr)) and prev is not None and t > prev[0]:
                dt = t - prev[0]
                pr = (pan - prev[1]) / dt
                tr = (tilt - prev[2]) / dt
            pr = 0.0 if math.isnan(pr) else pr
            tr = 0.0 if math.isnan(tr) else tr
            if prev is None or t > prev[0]:
                self._hist.append((t, pan, tilt, pr))
                self._hist_t.append(t)
            s = self._state
            s.t, s.pan, s.tilt, s.pan_rate, s.tilt_rate = t, pan, tilt, pr, tr
            s.connected = True
            for k, v in extra.items():
                setattr(s, k, v)
        self.fb_rate.tick()

    def state(self) -> GimbalState:
        with self._lock:
            s = self._state
            return GimbalState(s.t, s.pan, s.tilt, s.pan_rate, s.tilt_rate, s.connected, s.motors_on, s.mode, s.fault)

    def feedback_age(self) -> float:
        t = self._state.t
        return clock.now() - t if t > 0 else math.inf

    def angles_at(self, t: float) -> tuple[float, float]:
        """Pointing at host time t (interpolated; extrapolated with the latest rate if t is newer)."""
        with self._lock:
            if not self._hist:
                return self._state.pan, self._state.tilt
            ts = self._hist_t
            if t >= ts[-1]:
                t1, p1, q1, _ = self._hist[-1]
                dt = min(t - t1, 0.1)
                return p1 + self._state.pan_rate * dt, q1 + self._state.tilt_rate * dt
            i = bisect.bisect_left(ts, t)
            if i == 0:
                return self._hist[0][1], self._hist[0][2]
            t0, p0, q0, _ = self._hist[i - 1]
            t1, p1, q1, _ = self._hist[i]
            a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return p0 + a * (p1 - p0), q0 + a * (q1 - q0)

    def true_angles(self, t: float) -> tuple[float, float]:
        """Simulator hook (ground-truth pointing used to render frames)."""
        return self.angles_at(t)

    def info(self) -> dict:
        d = self.state().as_dict()
        d.update({"kind": self.kind, "feedback_hz": round(self.fb_rate.value(), 1),
                  "cmd_hz": round(self.cmd_rate.value(), 1), "estopped": self.estopped,
                  "feedback_age_s": round(min(self.feedback_age(), 99.0), 3),
                  "zero_pan_deg": math.degrees(self.zero_pan), "zero_tilt_deg": math.degrees(self.zero_tilt)})
        return d


class FixedGimbal(GimbalBase):
    """No gimbal (webcam on a desk): reports (0, 0), records what *would* be commanded."""

    kind = "none"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._stop = threading.Event()
        self.would_cmd = (0.0, 0.0)

    def start(self) -> None:
        self._stop.clear()

        def run():
            loop = clock.RateLoop(100.0, self._stop)
            while not self._stop.is_set():
                t = loop.wait()
                self._on_feedback(t, self.zero_pan, self.zero_tilt, 0.0, 0.0, motors_on=False, mode="fixed")

        threading.Thread(target=run, name="gimbal-fixed", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _send_angles(self, pan: float, tilt: float) -> None:
        self.would_cmd = (pan, tilt)

    def _send_rates(self, pan_rate: float, tilt_rate: float) -> None:
        self.would_cmd = (pan_rate, tilt_rate)
