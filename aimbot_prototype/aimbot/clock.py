"""Single time base for the whole system.

Every timestamp inside the engine is ``clock.now()`` (seconds, monotonic,
high resolution). Wall-clock time is only used for file names and log headers.
"""

from __future__ import annotations

import sys
import threading
import time

now = time.perf_counter
wall = time.time

_timer_res_enabled = False


def enable_high_res_timer() -> None:
    """On Windows, ask for 1 ms scheduler resolution so 100 Hz loops are stable."""
    global _timer_res_enabled
    if _timer_res_enabled or not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)  # type: ignore[attr-defined]
        _timer_res_enabled = True
    except Exception:
        pass


def sleep_until(t_target: float, stop: threading.Event | None = None) -> None:
    """Sleep until ``t_target`` (clock.now() seconds). Coarse sleep, then short spins."""
    while True:
        remaining = t_target - now()
        if remaining <= 0:
            return
        if stop is not None and stop.is_set():
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)


class RateLoop:
    """Deadline-based fixed-rate loop with jitter statistics.

    Usage::

        loop = RateLoop(100.0)
        while running:
            t = loop.wait()
            ...
    """

    def __init__(self, hz: float, stop: threading.Event | None = None):
        self.period = 1.0 / float(hz)
        self.stop = stop
        self._next = now() + self.period
        self.iterations = 0
        self.overruns = 0
        self._last = None
        self.dt_avg = self.period
        self.dt_max = 0.0
        self._win_max = 0.0
        self._win_start = now()

    def wait(self) -> float:
        sleep_until(self._next, self.stop)
        t = now()
        if t - self._next > 3 * self.period:
            # Fell far behind (debugger, suspend): resynchronise instead of bursting.
            # Smaller hiccups are caught up so no ticks (telemetry rows) are lost.
            self.overruns += 1
            self._next = t + self.period
        else:
            self._next += self.period
        if self._last is not None:
            dt = t - self._last
            self.dt_avg = 0.98 * self.dt_avg + 0.02 * dt
            self._win_max = max(self._win_max, dt)
            if t - self._win_start > 2.0:
                self.dt_max = self._win_max
                self._win_max = 0.0
                self._win_start = t
        self._last = t
        self.iterations += 1
        return t

    @property
    def hz(self) -> float:
        return 1.0 / self.dt_avg if self.dt_avg > 0 else 0.0


class RateMeter:
    """Counts events and reports a smoothed rate (Hz)."""

    def __init__(self, window_s: float = 1.0):
        self.window = window_s
        self._count = 0
        self._t0 = now()
        self.hz = 0.0
        self.total = 0

    def tick(self, n: int = 1) -> None:
        self._count += n
        self.total += n
        t = now()
        dt = t - self._t0
        if dt >= self.window:
            self.hz = self._count / dt
            self._count = 0
            self._t0 = t

    def value(self) -> float:
        # Decay to zero if events stopped arriving.
        if now() - self._t0 > 3 * self.window:
            return 0.0
        return self.hz
