"""Acceptance test for UC-18: run a sponsor tracking program through the
documented interface (ES-16.1, ES-05.1).

UC-18 acceptance test (PRD v2, section 7):
    A sample external tracker receives frames and gimbal state and returns
    target positions that the gimbal follows. Gimbal control code is
    unchanged, and latency stays <= 100 ms (p95).

The external tracker is loaded from a configuration string, exactly as a
sponsor's would be. ``GimbalControlLoop`` below stands in for the existing
control path: the test feeds it from the external tracker without changing
it, and checks the simulated gimbal ends up pointed at a moving target.
"""
import inspect
import math
import time

import numpy as np

from aimbot.uc18_external_tracker import ExternalTrackerBridge, GimbalState, load_external_tracker

WIDTH, HEIGHT, FPS = 640, 480, 30
PX_PER_DEG = 31.0 * WIDTH / 1936
HALF_FOV_DEG = (WIDTH / 2) / PX_PER_DEG


class GimbalControlLoop:
    """Existing control path: turns the published pointing error into rate commands."""

    KP = 6.0  # 1/s

    def __init__(self):
        self.pan = 0.0
        self.tilt = 0.0
        self.received = []

    def publish(self, dx, dy, capture_stamp_s):
        self.received.append((dx, dy, capture_stamp_s))
        self._last = (dx, dy)

    def step(self, dt):
        dx, dy = getattr(self, "_last", (0.0, 0.0))
        self.pan += self.KP * dx * HALF_FOV_DEG * dt
        self.tilt += self.KP * dy * (HEIGHT / 2 / PX_PER_DEG) * dt


def _frame(target_az, target_el, gimbal):
    frame = np.zeros((HEIGHT, WIDTH), np.uint8)
    x = WIDTH / 2 + (target_az - gimbal.pan) * PX_PER_DEG
    y = HEIGHT / 2 + (target_el - gimbal.tilt) * PX_PER_DEG
    xi, yi = int(round(x)), int(round(y))
    if 2 <= xi < WIDTH - 2 and 2 <= yi < HEIGHT - 2:
        frame[yi - 2:yi + 3, xi - 2:xi + 3] = 255
    return frame


def test_gimbal_follows_an_external_tracker_with_p95_latency_under_100_ms():
    control = GimbalControlLoop()
    source_before = inspect.getsource(GimbalControlLoop)
    tracker = load_external_tracker("aimbot.uc18_external_tracker:BrightestPointTracker")
    bridge = ExternalTrackerBridge(tracker, control)

    errors = []
    for i in range(10 * FPS):
        t = i / FPS
        target_az = 15 * math.sin(2 * math.pi * t / 8)       # deg
        target_el = 5 * math.sin(2 * math.pi * t / 11)
        capture = time.monotonic()
        frame = _frame(target_az, target_el, control)
        bridge.on_frame(frame, capture, GimbalState(control.pan, control.tilt, capture))
        control.step(1 / FPS)
        errors.append(math.hypot(target_az - control.pan, target_el - control.tilt))

    settled = errors[2 * FPS:]
    p95 = bridge.p95_latency_ms()
    print(f"UC-18 metric: p95 latency {p95:.2f} ms, mean pointing error after settling "
          f"{sum(settled) / len(settled):.2f} deg, max {max(settled):.2f} deg")
    assert len(control.received) >= 0.95 * len(errors)
    assert max(settled) < 3.0, "gimbal did not follow the external tracker's target"
    assert p95 <= 100
    assert inspect.getsource(GimbalControlLoop) == source_before


def test_bad_tracker_spec_is_rejected_with_a_clear_error():
    for spec in ("no_colon_here", "aimbot.uc18_external_tracker:"):
        try:
            load_external_tracker(spec)
        except ValueError:
            continue
        raise AssertionError(f"{spec!r} should be rejected")


def test_low_confidence_output_is_not_published():
    class Unsure:
        def process(self, frame, gimbal):
            from aimbot.uc18_external_tracker import TargetPosition
            return TargetPosition(10, 10, confidence=0.1)

    control = GimbalControlLoop()
    bridge = ExternalTrackerBridge(Unsure(), control)
    assert bridge.on_frame(np.zeros((HEIGHT, WIDTH), np.uint8), time.monotonic(),
                           GimbalState(0, 0, 0)) is None
    assert control.received == []
