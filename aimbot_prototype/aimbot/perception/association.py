"""Multi-target association for macro detections (SORT-lite) + target selection logic.

Track IDs persist across detections so the operator's chosen target is not swapped
for another object when paths cross (ES-01.4 / TR-8). Gimbal ego-motion between
detections is compensated before matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..types import Detection
from .geometry import iou


@dataclass
class Track:
    tid: int
    bbox: tuple
    cls: str
    conf: float
    t: float
    frame_id: int
    hits: int = 1
    misses: int = 0
    vel: tuple = (0.0, 0.0)          # px/s, own motion (ego-motion removed)
    history: list = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return x2 - x1, y2 - y1


def _shift_box(b, dx, dy):
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


class Associator:
    def __init__(self, cfg: Optional[dict] = None):
        c = cfg or {}
        self.iou_min = float(c.get("iou_min", 0.05))
        self.dist_gate = float(c.get("center_gate", 1.5))   # in units of box size
        self.max_misses = int(c.get("max_misses", 6))
        self.min_conf = float(c.get("min_conf", 0.2))
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, dets: list[Detection], t: float, frame_id: int, ego_shift=(0.0, 0.0)) -> list[Track]:
        dets = [d for d in dets if d.conf >= self.min_conf]
        # predict
        preds = {}
        for tr in self.tracks.values():
            dt = max(0.0, min(0.5, t - tr.t))
            preds[tr.tid] = _shift_box(tr.bbox, tr.vel[0] * dt + ego_shift[0], tr.vel[1] * dt + ego_shift[1])
        # candidate pairs (same logical class)
        pairs = []
        for di, d in enumerate(dets):
            dcx, dcy = d.center
            dw, dh = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
            for tid, pb in preds.items():
                tr = self.tracks[tid]
                if tr.cls != d.cls:
                    continue
                ov = iou(d.bbox, pb)
                pcx, pcy = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])
                scale = max(8.0, 0.5 * (dw + dh + (pb[2] - pb[0]) + (pb[3] - pb[1])) / 2)
                dist = math.hypot(dcx - pcx, dcy - pcy) / scale
                if ov >= self.iou_min or dist <= self.dist_gate:
                    pairs.append((ov - 0.25 * dist, di, tid))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            d, tr = dets[di], self.tracks[tid]
            dt = t - tr.t
            if dt > 1e-3:
                ocx, ocy = tr.center
                ncx, ncy = d.center
                vx = (ncx - ocx - ego_shift[0]) / dt
                vy = (ncy - ocy - ego_shift[1]) / dt
                a = 0.5 if tr.hits > 1 else 1.0
                tr.vel = ((1 - a) * tr.vel[0] + a * vx, (1 - a) * tr.vel[1] + a * vy)
            tr.bbox, tr.conf, tr.t, tr.frame_id = d.bbox, d.conf, t, frame_id
            tr.hits += 1
            tr.misses = 0
            d.track_id = tid
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tid = self._next
            self._next += 1
            self.tracks[tid] = Track(tid, d.bbox, d.cls, d.conf, t, frame_id)
            d.track_id = tid
        for tid in list(self.tracks):
            if tid not in used_t and self.tracks[tid].frame_id != frame_id:
                tr = self.tracks[tid]
                tr.misses += 1
                tr.bbox = preds.get(tid, tr.bbox)
                if tr.misses > self.max_misses:
                    del self.tracks[tid]
        return [tr for tr in self.tracks.values() if tr.misses == 0]


class TargetSelector:
    """Chooses which detected track to lock (FR-02 autonomous seeking, TR-8 priority list)."""

    def __init__(self, cfg: Optional[dict] = None):
        c = cfg or {}
        self.target_class = str(c.get("target_class", "auto"))
        self.priority = list(c.get("priority", ["drone", "ball", "card"]))
        self.min_conf = float(c.get("min_conf", 0.5))
        self.min_hits = int(c.get("min_hits", 2))
        self.prefer = str(c.get("prefer", "center"))   # center | confidence | size

    def wanted(self) -> list[str]:
        if self.target_class in ("auto", "", None):
            return self.priority
        if self.target_class == "any":
            return []
        return [self.target_class]

    def select(self, tracks: list[Track], frame_size: tuple[int, int], aim_px: tuple[float, float],
               near_px: Optional[tuple[float, float]] = None, near_gate_px: float = 1e9) -> Optional[Track]:
        wanted = self.wanted()
        cands = []
        for tr in tracks:
            if tr.conf < self.min_conf or tr.hits < self.min_hits:
                continue
            if wanted and tr.cls not in wanted:
                continue
            rank = wanted.index(tr.cls) if wanted else 0
            cx, cy = tr.center
            if near_px is not None:   # re-acquisition: closest to the predicted position
                dist = math.hypot(cx - near_px[0], cy - near_px[1])
                if dist > near_gate_px:
                    continue
                key = (rank, dist)
            elif self.prefer == "confidence":
                key = (rank, -tr.conf)
            elif self.prefer == "size":
                key = (rank, -(tr.size[0] * tr.size[1]))
            else:
                key = (rank, math.hypot(cx - aim_px[0], cy - aim_px[1]))
            cands.append((key, tr))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        return cands[0][1]
