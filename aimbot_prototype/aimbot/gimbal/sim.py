"""Simulated pan/tilt gimbal with rate/acceleration limits and transport latency.

Default numbers are representative of the Gremsy Mio (max controlled speed
~100 deg/s per axis) talking over 115200-baud UART. The internal position
loop mimics the gimbal's own controller in angle mode.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque

from .. import clock
from .base import GimbalBase


class SimGimbal(GimbalBase):
    kind = "sim"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = self.cfg
        self.acc_max = math.radians(float(c.get("sim_max_acc_dps2", 900.0)))
        self.pos_gain = float(c.get("sim_pos_gain", 25.0))          # 1/s, internal angle loop
        self.cmd_latency = float(c.get("sim_cmd_latency_ms", 6.0)) / 1000.0
        self.fb_latency = float(c.get("sim_fb_latency_ms", 5.0)) / 1000.0
        self.fb_hz = float(c.get("feedback_hz", 100.0))
        self.enc_noise = math.radians(float(c.get("sim_encoder_noise_deg", 0.005)))
        # mechanical end stops a bit outside the soft limits
        self.hard = (self.pan_min - 0.05, self.pan_max + 0.05, self.tilt_min - 0.05, self.tilt_max + 0.05)
        self._pos = [0.0, 0.0]
        self._vel = [0.0, 0.0]
        self._mode = "hold"
        self._sp = [0.0, 0.0]
        self._rate_sp = [0.0, 0.0]
        self._motors_on = True
        self._t = clock.now()
        self._pending: deque = deque()
        self._dyn_lock = threading.Lock()
        self._stop = threading.Event()
        self._rng = random.Random(11)

    # ------------------------------------------------------------------ dynamics
    def _apply_pending(self, t_upto: float) -> None:
        while self._pending and self._pending[0][0] <= t_upto:
            _, kind, a, b = self._pending.popleft()
            if kind == "angle":
                self._mode, self._sp = "angle", [a, b]
            elif kind == "rate":
                self._mode, self._rate_sp = "rate", [a, b]
            elif kind == "motors":
                self._motors_on = bool(a)

    def _advance(self, t: float) -> None:
        dt_step = 0.001
        while self._t < t:
            dt = min(dt_step, t - self._t)
            self._apply_pending(self._t)
            lo = (self.hard[0], self.hard[2])
            hi = (self.hard[1], self.hard[3])
            for i in range(2):
                if not self._motors_on:
                    want = 0.0
                elif self._mode == "angle":
                    want = self.pos_gain * (self._sp[i] - self._pos[i])
                elif self._mode == "rate":
                    want = self._rate_sp[i]
                else:
                    want = 0.0
                want = max(-self.max_rate, min(self.max_rate, want))
                dv = want - self._vel[i]
                dv = max(-self.acc_max * dt, min(self.acc_max * dt, dv))
                self._vel[i] += dv
                self._pos[i] += self._vel[i] * dt
                if self._pos[i] < lo[i]:
                    self._pos[i], self._vel[i] = lo[i], 0.0
                elif self._pos[i] > hi[i]:
                    self._pos[i], self._vel[i] = hi[i], 0.0
            self._t += dt

    def true_angles(self, t: float) -> tuple[float, float]:
        with self._dyn_lock:
            if t > self._t:
                self._advance(t)
            return self._pos[0], self._pos[1]

    def _true_state(self, t: float):
        with self._dyn_lock:
            if t > self._t:
                self._advance(t)
            return self._pos[0], self._pos[1], self._vel[0], self._vel[1]

    # ------------------------------------------------------------------ hooks
    def _queue(self, kind: str, a: float, b: float) -> None:
        with self._dyn_lock:
            self._pending.append((clock.now() + self.cmd_latency, kind, a, b))

    def _send_angles(self, pan: float, tilt: float) -> None:
        self._queue("angle", pan, tilt)

    def _send_rates(self, pan_rate: float, tilt_rate: float) -> None:
        self._queue("rate", pan_rate, tilt_rate)

    def _set_motors(self, on: bool) -> None:
        self._queue("motors", 1.0 if on else 0.0, 0.0)

    def start(self) -> None:
        self._stop.clear()
        threading.Thread(target=self._feedback_loop, name="gimbal-sim-fb", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _feedback_loop(self) -> None:
        period = 1.0 / self.fb_hz
        t_next = clock.now()
        while not self._stop.is_set():
            clock.sleep_until(t_next, self._stop)
            t_s = t_next
            t_next += period
            p, q, vp, vq = self._true_state(t_s)
            clock.sleep_until(t_s + self.fb_latency, self._stop)  # serial transport delay
            n = self.enc_noise
            self._on_feedback(t_s, p + self._rng.gauss(0, n), q + self._rng.gauss(0, n), vp, vq,
                              motors_on=self._motors_on, mode=self._mode)
            if clock.now() - t_next > 0.5:
                t_next = clock.now()
