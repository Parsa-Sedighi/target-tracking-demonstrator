"""Acceptance test for UC-11: click-to-designate any object (FR-09, ES-02.1).

UC-11 acceptance test (PRD v2, section 7):
    Clicking an unlabeled object starts tracking on the next frame. Moved at
    <= 1 m/s for 60 s at 10 ft, it is tracked with RMS error <= 1.0 deg.

The scene is synthetic: a cluttered background and a textured object the
detector has no class for, moving at up to 1 m/s at 10 ft and seen at 30 fps
on a 640-px-wide frame (scaled from the Blackfly S's 31.0 px/deg).
"""
import math

import numpy as np

from aimbot.uc11_click_designate import ClickDesignationTracker, px_per_deg_for_width

WIDTH, HEIGHT, FPS = 640, 480, 30
RANGE_M = 10 * 0.3048
OBJ = 24  # object size in pixels


def _scene(seed: int = 11):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (HEIGHT, WIDTH))
    kernel = np.ones(9) / 9.0
    clutter = np.apply_along_axis(lambda r: np.convolve(r, kernel, "same"), 1, noise)
    clutter = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, clutter)
    background = 110 + 40 * clutter / clutter.std()
    obj = rng.integers(0, 255, (OBJ, OBJ)).astype(float)
    sensor_noise = [rng.normal(0, 3, (HEIGHT, WIDTH)) for _ in range(8)]
    return background, obj, sensor_noise


def _render(background, obj, noise, cx, cy):
    frame = background + noise
    x0, y0 = int(round(cx - OBJ / 2)), int(round(cy - OBJ / 2))
    frame[y0:y0 + OBJ, x0:x0 + OBJ] = obj
    return np.clip(frame, 0, 255).astype(np.float32)


def _path(seconds: float):
    """Object center per frame, never faster than 1 m/s at 10 ft."""
    px_per_m = px_per_deg_for_width(WIDTH) * math.degrees(1 / RANGE_M)
    vx_peak, vy_peak = 0.95 * px_per_m, 0.25 * px_per_m  # px/s
    period_x, period_y = 6.0, 9.0
    ax = vx_peak * period_x / (2 * math.pi)
    ay = vy_peak * period_y / (2 * math.pi)
    frames = int(seconds * FPS)
    for i in range(frames + 1):
        t = i / FPS
        yield (WIDTH / 2 + ax * math.sin(2 * math.pi * t / period_x),
               HEIGHT / 2 + ay * math.sin(2 * math.pi * t / period_y))


def test_tracking_starts_on_next_frame_after_click():
    background, obj, noise = _scene()
    path = list(_path(1.0))
    tracker = ClickDesignationTracker(box_size=OBJ)
    tracker.designate(_render(background, obj, noise[0], *path[0]), path[0])

    result = tracker.update(_render(background, obj, noise[1], *path[1]))

    assert not result.lost
    assert result.frame_index == 1
    cx, cy = result.box.center
    assert math.hypot(cx - path[1][0], cy - path[1][1]) <= 2.0


def test_unlabeled_object_tracked_for_60_s_with_rms_error_under_1_deg():
    background, obj, noise = _scene()
    px_per_deg = px_per_deg_for_width(WIDTH)
    path = list(_path(60.0))
    tracker = ClickDesignationTracker(box_size=OBJ)
    tracker.designate(_render(background, obj, noise[0], *path[0]), path[0])

    errors_deg = []
    for i, (tx, ty) in enumerate(path[1:], start=1):
        result = tracker.update(_render(background, obj, noise[i % len(noise)], tx, ty))
        assert not result.lost, f"lock lost at frame {i}"
        cx, cy = result.box.center
        errors_deg.append(math.hypot(cx - tx, cy - ty) / px_per_deg)

    rms = math.sqrt(sum(e * e for e in errors_deg) / len(errors_deg))
    print(f"UC-11 metric: RMS error {rms:.3f} deg, max {max(errors_deg):.3f} deg over {len(errors_deg)} frames")
    assert len(errors_deg) == 60 * FPS
    assert rms <= 1.0


def test_click_near_the_edge_keeps_the_box_inside_the_frame():
    background, obj, noise = _scene()
    tracker = ClickDesignationTracker(box_size=OBJ)
    box = tracker.designate(_render(background, obj, noise[0], WIDTH / 2, HEIGHT / 2), (2, HEIGHT - 1))

    assert box.x >= 0 and box.y >= 0
    assert box.x + box.w <= WIDTH and box.y + box.h <= HEIGHT
