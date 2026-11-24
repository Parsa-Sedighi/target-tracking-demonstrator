"""UC-11: Click-to-designate any object (FR-09).

The operator clicks a point in the live video. The micro-tracker takes the
region around that point as its template and follows it on every following
frame, even if the detector has no class for the object.

Prototype tracker: normalized cross-correlation (NCC) template matching in
NumPy, with a constant-velocity Kalman filter that centers the search window
on the predicted position. On the Jetson the same ``designate``/``update``
interface wraps OpenCV's VitTrack; this NumPy version runs anywhere, so the
acceptance test needs no GPU or camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Camera model from the PRD: Blackfly S + 8 mm lens = 31.0 px/deg at 1936 px.
FULL_RES_WIDTH_PX = 1936
FULL_RES_PX_PER_DEG = 31.0


def px_per_deg_for_width(width_px: int) -> float:
    """Pixels per degree for a frame resized to ``width_px``."""
    return FULL_RES_PX_PER_DEG * width_px / FULL_RES_WIDTH_PX


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass(frozen=True)
class TrackResult:
    frame_index: int
    box: Box | None
    confidence: float
    lost: bool


class ConstantVelocityKalman:
    """4-state Kalman filter (x, y, vx, vy) for an image-plane target center."""

    def __init__(self, x: float, y: float, process_var: float = 4.0, meas_var: float = 1.0):
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var

    def predict(self) -> tuple[float, float]:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.state[0]), float(self.state[1])

    def correct(self, x: float, y: float) -> None:
        z = np.array([x, y])
        innovation = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P


class ClickDesignationTracker:
    """Locks onto whatever the operator clicks and follows it frame to frame."""

    def __init__(self, box_size: int = 24, search_radius: int = 20,
                 min_confidence: float = 0.5, template_update_rate: float = 0.1):
        if box_size < 8 or box_size % 2:
            raise ValueError("box_size must be an even number of pixels >= 8")
        self.box_size = box_size
        self.search_radius = search_radius
        self.min_confidence = min_confidence
        self.template_update_rate = template_update_rate
        self._template: np.ndarray | None = None
        self._kalman: ConstantVelocityKalman | None = None
        self._frame_index = 0
        self.box: Box | None = None

    @property
    def locked(self) -> bool:
        return self._template is not None

    def designate(self, frame: np.ndarray, click_xy: tuple[float, float]) -> Box:
        """Start tracking the region centered on the operator's click."""
        gray = _to_gray(frame)
        height, width = gray.shape
        half = self.box_size // 2
        cx = int(round(min(max(click_xy[0], half), width - half)))
        cy = int(round(min(max(click_xy[1], half), height - half)))
        x0, y0 = cx - half, cy - half
        self._template = gray[y0:y0 + self.box_size, x0:x0 + self.box_size].copy()
        self._kalman = ConstantVelocityKalman(float(cx), float(cy))
        self._frame_index = 0
        self.box = Box(x0, y0, self.box_size, self.box_size)
        return self.box

    def release(self) -> None:
        self._template = None
        self._kalman = None
        self.box = None

    def update(self, frame: np.ndarray) -> TrackResult:
        """Track the designated region in the next frame."""
        if self._template is None or self._kalman is None:
            raise RuntimeError("No target designated; call designate() first")
        self._frame_index += 1
        gray = _to_gray(frame)
        height, width = gray.shape
        half = self.box_size // 2

        pred_x, pred_y = self._kalman.predict()
        r = self.search_radius
        sx0 = int(max(0, round(pred_x) - half - r))
        sy0 = int(max(0, round(pred_y) - half - r))
        sx1 = int(min(width, round(pred_x) + half + r))
        sy1 = int(min(height, round(pred_y) + half + r))
        region = gray[sy0:sy1, sx0:sx1]
        if region.shape[0] < self.box_size or region.shape[1] < self.box_size:
            return TrackResult(self._frame_index, None, 0.0, True)

        scores = _ncc(region, self._template)
        iy, ix = np.unravel_index(int(np.argmax(scores)), scores.shape)
        confidence = float(scores[iy, ix])
        if confidence < self.min_confidence:
            return TrackResult(self._frame_index, None, confidence, True)

        x0, y0 = sx0 + ix, sy0 + iy
        self._kalman.correct(x0 + half, y0 + half)
        patch = gray[y0:y0 + self.box_size, x0:x0 + self.box_size]
        a = self.template_update_rate
        self._template = (1 - a) * self._template + a * patch
        self.box = Box(float(x0), float(y0), float(self.box_size), float(self.box_size))
        return TrackResult(self._frame_index, self.box, confidence, False)


def pointing_error_deg(box: Box, frame_shape: tuple[int, int], px_per_deg: float) -> tuple[float, float]:
    """Angular offset of the box center from the image center (pan, tilt), in degrees."""
    height, width = frame_shape[:2]
    cx, cy = box.center
    return ((cx - width / 2.0) / px_per_deg, (cy - height / 2.0) / px_per_deg)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    return frame


def _ncc(region: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of ``template`` at every position in ``region``."""
    th, tw = template.shape
    windows = sliding_window_view(region, (th, tw))
    t = template - template.mean()
    t_norm = np.sqrt((t * t).sum()) + 1e-6
    w_mean = windows.mean(axis=(2, 3), keepdims=True)
    w = windows - w_mean
    numerator = np.einsum("ijkl,kl->ij", w, t)
    w_norm = np.sqrt((w * w).sum(axis=(2, 3))) + 1e-6
    return numerator / (w_norm * t_norm)
