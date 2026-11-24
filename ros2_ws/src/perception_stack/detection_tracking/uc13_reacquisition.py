"""UC-13: Reacquire the target after a brief blockage (FR-04, ES-03.1, ES-03.2).

When the micro-tracker loses the target, the system holds the gimbal and
searches the field of view. Each detector frame is checked for a candidate
that matches the lost target: same class, similar size, and inside a search
gate centered on the last predicted position that widens the longer the
target has been gone. A match re-initializes the tracker; after the 5 s
search window the state goes to LOST and the UI shows "Target lost".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    TRACKING = "TRACKING"
    SEARCHING = "SEARCHING"
    LOST = "LOST"


@dataclass(frozen=True)
class Candidate:
    cx: float
    cy: float
    w: float
    h: float
    label: str
    score: float


@dataclass(frozen=True)
class TargetSignature:
    cx: float
    cy: float
    vx: float           # px/s
    vy: float           # px/s
    w: float
    h: float
    label: str


@dataclass(frozen=True)
class Decision:
    state: State
    reinit_box: Candidate | None = None
    hold_gimbal: bool = False
    operator_message: str | None = None


class ReacquisitionManager:
    """Hold-and-search logic between a tracker loss and reacquisition."""

    def __init__(self, search_window_s: float = 5.0, base_gate_px: float = 40.0,
                 gate_growth_px_per_s: float = 60.0, max_size_ratio: float = 1.8,
                 min_score: float = 0.4, max_coast_s: float = 1.0):
        self.search_window_s = search_window_s
        self.base_gate_px = base_gate_px
        self.gate_growth = gate_growth_px_per_s
        self.max_size_ratio = max_size_ratio
        self.min_score = min_score
        self.max_coast_s = max_coast_s
        self.state = State.TRACKING
        self._signature: TargetSignature | None = None
        self._lost_at: float | None = None

    def on_track(self, signature: TargetSignature) -> Decision:
        """Tracker still has the target; remember what it looks like."""
        self._signature = signature
        self.state = State.TRACKING
        self._lost_at = None
        return Decision(State.TRACKING)

    def on_loss(self, now_s: float) -> Decision:
        """Tracker reported low confidence: hold the gimbal and start searching."""
        if self.state == State.TRACKING:
            self.state = State.SEARCHING
            self._lost_at = now_s
        return Decision(self.state, hold_gimbal=True)

    def on_detections(self, now_s: float, candidates: list[Candidate]) -> Decision:
        """Check a detector frame for the lost target while SEARCHING."""
        if self.state == State.TRACKING:
            return Decision(State.TRACKING)
        if self.state == State.LOST:
            return Decision(State.LOST, hold_gimbal=True, operator_message="Target lost")

        elapsed = now_s - self._lost_at
        if elapsed > self.search_window_s:
            self.state = State.LOST
            return Decision(State.LOST, hold_gimbal=True, operator_message="Target lost")

        match = self._best_match(elapsed, candidates)
        if match is None:
            return Decision(State.SEARCHING, hold_gimbal=True)
        self.state = State.TRACKING
        self._lost_at = None
        return Decision(State.TRACKING, reinit_box=match)

    def tick(self, now_s: float) -> Decision:
        """Timer check, so LOST is declared even when no detections arrive."""
        if self.state == State.SEARCHING and now_s - self._lost_at > self.search_window_s:
            self.state = State.LOST
        if self.state == State.LOST:
            return Decision(State.LOST, hold_gimbal=True, operator_message="Target lost")
        return Decision(self.state, hold_gimbal=self.state == State.SEARCHING)

    def _best_match(self, elapsed_s: float, candidates: list[Candidate]) -> Candidate | None:
        sig = self._signature
        if sig is None:
            return None
        coast = min(elapsed_s, self.max_coast_s)
        px, py = sig.cx + sig.vx * coast, sig.cy + sig.vy * coast
        gate = self.base_gate_px + self.gate_growth * elapsed_s
        best, best_cost = None, None
        for c in candidates:
            if c.label != sig.label or c.score < self.min_score:
                continue
            ratio = (c.w * c.h) / max(sig.w * sig.h, 1.0)
            if not (1 / self.max_size_ratio <= ratio <= self.max_size_ratio):
                continue
            distance = math.hypot(c.cx - px, c.cy - py)
            if distance > gate:
                continue
            if best_cost is None or distance < best_cost:
                best, best_cost = c, distance
        return best
