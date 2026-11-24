"""UC-12: Keep the operator's chosen target when other targets cross its path
(FR-02, ES-01.4).

Detections arrive from the asynchronous detector (RF-DETR Nano or YOLO26s).
This module associates them to persistent track IDs (a ByteTrack-style
tracking-by-detection step) and remembers which ID the operator selected, so
the system never switches targets on its own when paths cross.

Association uses a constant-velocity prediction per track. Two targets that
cross with different headings stay separable because each track is matched to
the detection nearest its *predicted* position, not its last seen position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    cx: float
    cy: float
    w: float
    h: float
    label: str
    score: float = 1.0


@dataclass
class Track:
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    label: str
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    misses: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    def predicted(self) -> tuple[float, float]:
        return self.cx + self.vx, self.cy + self.vy


class SelectedTargetTracker:
    """Multi-target tracker that keeps one operator-selected track ID."""

    def __init__(self, gate_px: float = 40.0, max_misses: int = 15,
                 velocity_smoothing: float = 0.5):
        self.gate_px = gate_px
        self.max_misses = max_misses
        self.alpha = velocity_smoothing
        self.tracks: dict[int, Track] = {}
        self.selected_id: int | None = None
        self._next_id = 1

    # ----- operator actions -------------------------------------------------
    def select(self, track_id: int) -> None:
        if track_id not in self.tracks:
            raise KeyError(f"no active track {track_id}")
        self.selected_id = track_id

    def select_nearest(self, x: float, y: float) -> int:
        """Select the track nearest an operator click; returns its ID."""
        if not self.tracks:
            raise LookupError("no tracks to select")
        best = min(self.tracks.values(), key=lambda t: math.hypot(t.cx - x, t.cy - y))
        self.selected_id = best.track_id
        return best.track_id

    @property
    def selected(self) -> Track | None:
        return self.tracks.get(self.selected_id) if self.selected_id is not None else None

    # ----- per-detection-frame update ---------------------------------------
    def update(self, detections: list[Detection]) -> dict[int, Track]:
        tracks = list(self.tracks.values())
        assignment = self._assign(tracks, detections)
        matched_dets = set()
        for track, det_index in assignment:
            det = detections[det_index]
            matched_dets.add(det_index)
            new_vx, new_vy = det.cx - track.cx, det.cy - track.cy
            if track.misses:
                new_vx /= track.misses + 1
                new_vy /= track.misses + 1
            track.vx = self.alpha * track.vx + (1 - self.alpha) * new_vx
            track.vy = self.alpha * track.vy + (1 - self.alpha) * new_vy
            track.cx, track.cy, track.w, track.h = det.cx, det.cy, det.w, det.h
            track.hits += 1
            track.misses = 0
            track.history.append((track.cx, track.cy))

        matched_tracks = {t.track_id for t, _ in assignment}
        for track in tracks:
            if track.track_id not in matched_tracks:
                track.misses += 1
                track.cx, track.cy = track.predicted()  # coast on prediction
                if track.misses > self.max_misses:
                    del self.tracks[track.track_id]

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._spawn(det)
        return self.tracks

    def _spawn(self, det: Detection) -> None:
        track = Track(self._next_id, det.cx, det.cy, det.w, det.h, det.label)
        track.history.append((det.cx, det.cy))
        self.tracks[self._next_id] = track
        self._next_id += 1

    def _cost(self, track: Track, det: Detection) -> float:
        px, py = track.predicted()
        distance = math.hypot(det.cx - px, det.cy - py)
        size_penalty = abs(math.log(max(det.w * det.h, 1.0) / max(track.w * track.h, 1.0))) * 10.0
        class_penalty = 0.0 if det.label == track.label else 25.0
        return distance + size_penalty + class_penalty

    def _assign(self, tracks: list[Track], dets: list[Detection]) -> list[tuple[Track, int]]:
        """Minimum-cost one-to-one assignment within the gate.

        Exhaustive search is exact and fast for the handful of targets in a
        test scene; above 6 tracks it falls back to greedy matching.
        """
        if not tracks or not dets:
            return []
        gate = self.gate_px
        costs = {(ti, di): self._cost(t, d) for ti, t in enumerate(tracks) for di, d in enumerate(dets)}
        if len(tracks) <= 6 and len(dets) <= 6:
            best: dict = {"pairs": [], "score": None}

            def search(ti: int, used: frozenset, pairs: list, total: float) -> None:
                if ti == len(tracks):
                    score = (-len(pairs), total)
                    if best["score"] is None or score < best["score"]:
                        best["pairs"], best["score"] = list(pairs), score
                    return
                search(ti + 1, used, pairs, total)  # leave this track unmatched
                for di in range(len(dets)):
                    c = costs[(ti, di)]
                    if di not in used and c <= gate:
                        pairs.append((ti, di))
                        search(ti + 1, used | {di}, pairs, total + c)
                        pairs.pop()

            search(0, frozenset(), [], 0.0)
            return [(tracks[ti], di) for ti, di in best["pairs"]]
        pairs, used_t, used_d = [], set(), set()
        for (ti, di), c in sorted(costs.items(), key=lambda kv: kv[1]):
            if c <= gate and ti not in used_t and di not in used_d:
                pairs.append((tracks[ti], di))
                used_t.add(ti)
                used_d.add(di)
        return pairs
