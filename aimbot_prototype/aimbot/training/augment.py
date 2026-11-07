"""Camera-matched training augmentation: motion blur and sensor noise.

Instead of "cleaning" training images, make the detector robust to what the Blackfly + gimbal
really produce: motion smear while the gimbal slews or the ball falls (2 ms exposure helps but does
not remove it) and read noise at 12 dB gain in a dim lab. Pure OpenCV/NumPy, so it needs no extra
packages (albumentations would pull in opencv-python-headless, which conflicts with
opencv-python's GUI functions on Windows).

install_camera_aug() inserts CameraAug into Ultralytics' training pipeline, just before the final
formatting step (after mosaic, perspective, HSV and flips); validation images are never touched.
"""

from __future__ import annotations

import logging
import random

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraAug:
    """Picklable (module-level) so Windows data-loader worker processes can use it."""

    def __init__(self, p_blur: float = 0.15, p_noise: float = 0.15, max_kernel: int = 9, max_sigma: float = 8.0):
        self.p_blur, self.p_noise = float(p_blur), float(p_noise)
        self.max_kernel, self.max_sigma = int(max_kernel), float(max_sigma)

    def __call__(self, labels: dict) -> dict:
        img = labels.get("img")
        if img is None or not isinstance(img, np.ndarray):
            return labels
        if self.p_blur > 0 and random.random() < self.p_blur:
            k = random.choice([k for k in (3, 5, 7, 9, 11, 13) if k <= self.max_kernel] or [3])
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0
            rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1.0)
            kernel = cv2.warpAffine(kernel, rot, (k, k))
            s = kernel.sum()
            if s > 0:
                img = cv2.filter2D(img, -1, kernel / s)
        if self.p_noise > 0 and random.random() < self.p_noise:
            sigma = random.uniform(1.5, self.max_sigma)
            noise = np.random.normal(0.0, sigma, img.shape[:2] + ((1,) if img.ndim == 3 else ())).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)   # grey noise, like a mono-ish sensor
        labels["img"] = np.ascontiguousarray(img)
        return labels

    def __repr__(self) -> str:
        return f"CameraAug(p_blur={self.p_blur}, p_noise={self.p_noise})"


def install_camera_aug(p_blur: float = 0.15, p_noise: float = 0.15) -> bool:
    """Patch Ultralytics' YOLODataset so augmented (training) pipelines end with CameraAug."""
    try:
        from ultralytics.data import dataset as D
    except Exception as exc:                          # pragma: no cover
        log.warning("camera augmentation not installed: %s", exc)
        return False
    orig = D.YOLODataset.build_transforms
    if getattr(orig, "_aimbot_camera_aug", False):
        return True

    def build_transforms(self, hyp=None):
        t = orig(self, hyp)
        if getattr(self, "augment", False):
            try:
                t.transforms.insert(max(0, len(t.transforms) - 1), CameraAug(p_blur, p_noise))
            except Exception as exc:                  # pragma: no cover - layout changed in a new Ultralytics
                log.warning("camera augmentation skipped: %s", exc)
        return t

    build_transforms._aimbot_camera_aug = True
    D.YOLODataset.build_transforms = build_transforms
    return True
