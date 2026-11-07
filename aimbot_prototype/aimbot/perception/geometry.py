"""Camera model and gimbal pointing geometry.

Frames
------
* Gimbal base ("world") frame: x forward (pan=0, tilt=0 boresight), y right, z up.
* Camera frame (OpenCV): X right, Y down, Z forward.
* Pan (yaw) is a rotation about +z; positive = to the right (clockwise seen from
  above) - same sign as MAVLink yaw. Tilt (pitch) positive = up, same as MAVLink.
* A direction is described by (az, el): az + right, el + up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# camera-frame vector -> gimbal body-frame vector  (body = [Z, X, -Y] of camera)
C_CAM_TO_BODY = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def gimbal_rotation(pan: float, tilt: float) -> np.ndarray:
    """Rotation taking gimbal-body vectors into the base frame."""
    return rot_z(pan) @ rot_y(-tilt)


def azel_to_dir(az: float, el: float) -> np.ndarray:
    ce = math.cos(el)
    return np.array([ce * math.cos(az), ce * math.sin(az), math.sin(el)])


def dir_to_azel(d: np.ndarray) -> tuple[float, float]:
    x, y, z = float(d[0]), float(d[1]), float(d[2])
    return math.atan2(y, x), math.atan2(z, math.hypot(x, y))


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Optional[np.ndarray] = None  # OpenCV distortion coefficients (k1,k2,p1,p2,k3)

    @classmethod
    def from_lens(cls, width: int, height: int, pixel_um: float, focal_mm: float) -> "Intrinsics":
        f_px = focal_mm / (pixel_um * 1e-3)
        return cls(width, height, f_px, f_px, (width - 1) / 2.0, (height - 1) / 2.0)

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        f_px = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, f_px, f_px, (width - 1) / 2.0, (height - 1) / 2.0)

    @classmethod
    def from_config(cls, c: dict, width: int, height: int) -> "Intrinsics":
        """Build from a config block. Priority: explicit fx/fy/cx/cy > lens > hfov."""
        c = c or {}
        if c.get("fx"):
            intr = cls(int(c.get("width", width)), int(c.get("height", height)), float(c["fx"]),
                       float(c.get("fy", c["fx"])), float(c.get("cx", (width - 1) / 2)),
                       float(c.get("cy", (height - 1) / 2)))
            if c.get("dist"):
                intr.dist = np.array(c["dist"], dtype=np.float64)
            return intr.scaled_to(width, height)
        if c.get("focal_mm") and c.get("pixel_um"):
            # pixel_um is the physical pixel pitch at the *delivered* resolution
            return cls.from_lens(width, height, float(c["pixel_um"]), float(c["focal_mm"]))
        return cls.from_hfov(width, height, float(c.get("hfov_deg", 60.0)))

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        if width == self.width and height == self.height:
            return self
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5, self.dist)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / 2.0 / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / 2.0 / self.fy))

    @property
    def px_per_deg(self) -> float:
        return self.fx * math.pi / 180.0

    def normalized(self, u: float, v: float) -> tuple[float, float]:
        """Pixel -> undistorted normalised image coordinates (x/z, y/z)."""
        if self.dist is not None and np.any(self.dist):
            import cv2

            pts = np.array([[[u, v]]], dtype=np.float64)
            k = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
            out = cv2.undistortPoints(pts, k, self.dist)
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy

    def project_normalized(self, x: float, y: float) -> tuple[float, float]:
        # Distortion is not re-applied (fine for the simulator, which is ideal pinhole).
        return self.fx * x + self.cx, self.fy * y + self.cy

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy, "hfov_deg": self.hfov_deg, "vfov_deg": self.vfov_deg}


def pixel_to_dir(intr: Intrinsics, u: float, v: float, pan: float, tilt: float) -> np.ndarray:
    x, y = intr.normalized(u, v)
    ray_cam = np.array([x, y, 1.0])
    d = gimbal_rotation(pan, tilt) @ (C_CAM_TO_BODY @ ray_cam)
    return d / np.linalg.norm(d)


def pixel_to_azel(intr: Intrinsics, u: float, v: float, pan: float, tilt: float) -> tuple[float, float]:
    return dir_to_azel(pixel_to_dir(intr, u, v, pan, tilt))


def world_to_cam(p_world: np.ndarray, pan: float, tilt: float) -> np.ndarray:
    """Base-frame point/vector -> camera frame (X right, Y down, Z forward)."""
    body = gimbal_rotation(pan, tilt).T @ p_world
    return C_CAM_TO_BODY.T @ body


def world_to_pixel(intr: Intrinsics, p_world: np.ndarray, pan: float, tilt: float):
    """Returns (u, v, depth) or None when the point is behind the camera."""
    pc = world_to_cam(np.asarray(p_world, dtype=np.float64), pan, tilt)
    if pc[2] <= 1e-6:
        return None
    u, v = intr.project_normalized(pc[0] / pc[2], pc[1] / pc[2])
    return u, v, float(pc[2])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return math.acos(c)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0
