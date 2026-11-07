"""Micro (frame-rate) trackers: bridge the gaps between asynchronous YOLO detections.

* ``template``: normalised cross-correlation in a motion-predicted search window,
  sub-pixel peak refinement, slow template adaptation. Always available, ~1 ms.
* ``csrt`` / ``kcf``: OpenCV contrib trackers (needs opencv-contrib-python).
* ``mil``: OpenCV main-module tracker (slow, last resort).

All trackers take ``shift`` = predicted image motion caused by the gimbal between
frames (ego-motion), so the search window stays on the target while the camera slews.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


class MicroTracker:
    name = "base"

    def init(self, image: np.ndarray, bbox, t: float) -> bool:
        raise NotImplementedError

    def update(self, image: np.ndarray, t: float, shift=(0.0, 0.0), prior=None):
        """Return (ok, bbox, score)."""
        raise NotImplementedError

    @property
    def bbox(self):
        raise NotImplementedError


def _subpixel_peak(res: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """Parabolic interpolation of a correlation peak."""
    h, w = res.shape
    dx = dy = 0.0
    if 0 < x < w - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]
        den = l - 2 * c + r
        if abs(den) > 1e-9:
            dx = 0.5 * (l - r) / den
    if 0 < y < h - 1:
        u, c, d = res[y - 1, x], res[y, x], res[y + 1, x]
        den = u - 2 * c + d
        if abs(den) > 1e-9:
            dy = 0.5 * (u - d) / den
    return x + max(-1.0, min(1.0, dx)), y + max(-1.0, min(1.0, dy))


class TemplateTracker(MicroTracker):
    name = "template"

    def __init__(self, cfg: Optional[dict] = None):
        c = cfg or {}
        self.max_tpl = int(c.get("max_template_px", 40))
        self.search_factor = float(c.get("search_factor", 2.2))
        self.min_score = float(c.get("min_score", 0.45))
        self.alpha = float(c.get("update_alpha", 0.05))
        self.update_score = float(c.get("update_min_score", 0.72))
        self.min_size = float(c.get("min_size_px", 6))
        # match on the central part of the box: thin targets (drone arms) are mostly background at the edges,
        # and a template that is mostly background slowly drifts off the target.
        self.tpl_frac = float(c.get("template_frac", 0.6))
        self.color = bool(c.get("color", True))             # match on BGR, not grey (better discrimination)
        self.motion_penalty = float(c.get("motion_penalty", 0.12))  # NCC cost per sigma^2 away from prediction
        self.sigma_px = float(c.get("motion_sigma_px", 5.0))
        self.min_texture = float(c.get("min_texture", 4.0))   # grey-level std-dev a template needs
        self.center = (0.0, 0.0)
        self.size = (0.0, 0.0)
        self.vel = (0.0, 0.0)
        self.t = 0.0
        self.scale = 1.0
        self.tpl: Optional[np.ndarray] = None
        self.score = 0.0
        self._tw = self._th = 0.0

    @property
    def bbox(self):
        (cx, cy), (w, h) = self.center, self.size
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def _prep(self, img: np.ndarray) -> np.ndarray:
        if self.color or img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _patch(self, image: np.ndarray, cx: float, cy: float, w: float, h: float) -> Optional[np.ndarray]:
        H, W = image.shape[:2]
        x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2))
        x2, y2 = x1 + int(round(w)), y1 + int(round(h))
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 - x1 < 2 or y2 - y1 < 2:
            return None
        g = self._prep(image[y1:y2, x1:x2])
        tw, th = max(4, int(round(w * self.scale))), max(4, int(round(h * self.scale)))
        return cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)

    def init(self, image: np.ndarray, bbox, t: float, keep_velocity: bool = False) -> bool:
        x1, y1, x2, y2 = bbox
        H, W = image.shape[:2]
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(W - 1.0, x2), min(H - 1.0, y2)
        w, h = x2 - x1, y2 - y1
        if w < self.min_size or h < self.min_size:
            return False
        f = self.tpl_frac if min(w, h) * self.tpl_frac >= self.min_size else 1.0
        self._tw, self._th = w * f, h * f
        self.scale = min(1.0, self.max_tpl / max(self._tw, self._th))
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        tpl = self._patch(image, cx, cy, self._tw, self._th)
        if tpl is None or float(tpl.std()) < self.min_texture:
            return False     # nothing to correlate on (blank wall): refuse rather than drift
        self.tpl, self.center, self.size, self.t = tpl, (cx, cy), (w, h), t
        if not keep_velocity:
            self.vel = (0.0, 0.0)
        self.score = 1.0
        return True

    def update(self, image: np.ndarray, t: float, shift=(0.0, 0.0), prior=None):
        """prior: predicted centre (px) from the Kalman filter; used instead of the tracker's own
        velocity model when available, which keeps the search window on the target."""
        if self.tpl is None:
            return False, self.bbox, 0.0
        H, W = image.shape[:2]
        dt = max(0.0, min(0.2, t - self.t))
        (cx, cy), (w, h) = self.center, self.size
        if prior is not None:
            px, py = prior
            mv = (4.0, 4.0)
        else:
            px = cx + self.vel[0] * dt + shift[0]
            py = cy + self.vel[1] * dt + shift[1]
            mv = (abs(self.vel[0]) * dt * 0.5 + 4, abs(self.vel[1]) * dt * 0.5 + 4)
        sw = w * self.search_factor / 2 + mv[0]
        sh = h * self.search_factor / 2 + mv[1]
        x0, y0 = int(max(0, math.floor(px - sw))), int(max(0, math.floor(py - sh)))
        x1, y1 = int(min(W, math.ceil(px + sw))), int(min(H, math.ceil(py + sh)))
        th, tw = self.tpl.shape[:2]
        rw, rh = int(round((x1 - x0) * self.scale)), int(round((y1 - y0) * self.scale))
        if rw < tw + 2 or rh < th + 2:
            self.center = (px, py)
            self.t = t
            return False, self.bbox, 0.0
        roi = self._prep(image[y0:y1, x0:x1])
        roi = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_AREA).astype(np.float32)
        res = cv2.matchTemplate(roi, self.tpl, cv2.TM_CCOEFF_NORMED)
        if self.motion_penalty > 0:
            # Prefer matches near the predicted position: symmetric targets (a quadcopter's four
            # rotors) and clutter create secondary correlation peaks a few tens of pixels away.
            hh, ww = res.shape
            ex = (px - x0) * self.scale - tw / 2.0
            ey = (py - y0) * self.scale - th / 2.0
            sig = self.sigma_px * self.scale * (1.0 if prior is not None else 2.5) + (0.0 if prior is not None else 0.2 * max(tw, th))
            xs = (np.arange(ww, dtype=np.float32) - ex) / sig
            ys = (np.arange(hh, dtype=np.float32) - ey) / sig
            pen = self.motion_penalty * np.minimum(ys[:, None] ** 2 + xs[None, :] ** 2, 25.0)
            weighted = res - pen
            _, _, _, loc = cv2.minMaxLoc(weighted)
            # the prior picks *which* peak; the position comes from the unweighted peak (no bias)
            lx0, ly0 = max(0, loc[0] - 2), max(0, loc[1] - 2)
            win = res[ly0:loc[1] + 3, lx0:loc[0] + 3]
            _, score, _, l2 = cv2.minMaxLoc(win)
            loc = (lx0 + l2[0], ly0 + l2[1])
        else:
            _, score, _, loc = cv2.minMaxLoc(res)
        self.score = float(score)
        if not np.isfinite(score) or score < self.min_score:
            self.center = (px, py)
            self.t = t
            return False, self.bbox, float(score) if np.isfinite(score) else 0.0
        mx, my = _subpixel_peak(res, loc[0], loc[1])
        sx, sy = (x1 - x0) / rw, (y1 - y0) / rh
        ncx = x0 + (mx + tw / 2.0) * sx
        ncy = y0 + (my + th / 2.0) * sy
        if dt > 1e-4:
            vx = (ncx - cx - shift[0]) / dt
            vy = (ncy - cy - shift[1]) / dt
            self.vel = (0.6 * self.vel[0] + 0.4 * vx, 0.6 * self.vel[1] + 0.4 * vy)
        self.center, self.t = (ncx, ncy), t
        if score > self.update_score:
            patch = self._patch(image, ncx, ncy, self._tw, self._th)
            if patch is not None and patch.shape == self.tpl.shape:
                cv2.addWeighted(self.tpl, 1 - self.alpha, patch, self.alpha, 0, dst=self.tpl)
        return True, self.bbox, float(score)


class CvTracker(MicroTracker):
    def __init__(self, kind: str):
        self.kind = kind
        self.name = kind
        self._factory = self.factory(kind)
        if self._factory is None:
            raise RuntimeError(f"OpenCV tracker {kind!r} not available (install opencv-contrib-python)")
        self.trk = None
        self._bbox = (0, 0, 0, 0)

    @staticmethod
    def factory(kind: str):
        names = {"csrt": "TrackerCSRT_create", "kcf": "TrackerKCF_create", "mil": "TrackerMIL_create"}
        fn = names.get(kind)
        if fn is None:
            return None
        if hasattr(cv2, fn):
            return getattr(cv2, fn)
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, fn):
            return getattr(legacy, fn)
        return None

    @property
    def bbox(self):
        return self._bbox

    def init(self, image, bbox, t, keep_velocity: bool = False) -> bool:
        x1, y1, x2, y2 = bbox
        self.trk = self._factory()
        self.trk.init(image, (int(x1), int(y1), max(2, int(x2 - x1)), max(2, int(y2 - y1))))
        self._bbox = tuple(map(float, bbox))
        return True

    def update(self, image, t, shift=(0.0, 0.0), prior=None):
        ok, r = self.trk.update(image)
        if ok:
            x, y, w, h = r
            self._bbox = (float(x), float(y), float(x + w), float(y + h))
        return bool(ok), self._bbox, 1.0 if ok else 0.0


def create_micro_tracker(cfg: Optional[dict]) -> MicroTracker:
    c = cfg or {}
    kind = str(c.get("backend", "template")).lower()
    if kind == "auto":
        kind = "template"
    if kind in ("csrt", "kcf", "mil"):
        try:
            return CvTracker(kind)
        except RuntimeError:
            pass
    return TemplateTracker(c)


def available_trackers() -> list[str]:
    out = ["template"]
    for k in ("csrt", "kcf", "mil"):
        if CvTracker.factory(k) is not None:
            out.append(k)
    return out
