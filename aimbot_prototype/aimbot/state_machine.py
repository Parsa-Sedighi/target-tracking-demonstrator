"""System state machine (FR-04 / UC-9 / ES-03.x).

IDLE --start--> SEARCHING --target acquired--> TRACKING
TRACKING --target lost--> SEARCHING (re-acquire window, default 5 s)
TRACKING --release--> SEARCHING (operator released the target; no re-acquire window)
SEARCHING --window expired--> LOST (hold + "TARGET LOST" alert)
LOST --target acquired--> TRACKING (if auto_resume) ; any --stop--> IDLE
TRACKING/SEARCHING --pause--> PAUSED --resume--> SEARCHING
IDLE --calibrate--> CALIBRATING --done--> IDLE
any --estop--> ESTOP (latched) --reset--> IDLE
any active --gimbal link lost--> FAULT --reset--> IDLE
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from . import clock

log = logging.getLogger(__name__)

IDLE, SEARCHING, TRACKING, LOST, PAUSED, CALIBRATING, ESTOP, FAULT = (
    "IDLE", "SEARCHING", "TRACKING", "LOST", "PAUSED", "CALIBRATING", "ESTOP", "FAULT")
ALL_STATES = (IDLE, SEARCHING, TRACKING, LOST, PAUSED, CALIBRATING, ESTOP, FAULT)
ACTIVE = (SEARCHING, TRACKING, LOST, PAUSED)

_ALLOWED = {
    "start": {IDLE: SEARCHING, LOST: SEARCHING, PAUSED: SEARCHING},
    "acquired": {SEARCHING: TRACKING, LOST: TRACKING},
    "lost": {TRACKING: SEARCHING},
    "release": {TRACKING: SEARCHING},          # operator released the target / sensor swapped: seek again
    "reacquire_timeout": {SEARCHING: LOST},
    "pause": {SEARCHING: PAUSED, TRACKING: PAUSED, LOST: PAUSED},
    "resume": {PAUSED: SEARCHING},
    "stop": {SEARCHING: IDLE, TRACKING: IDLE, LOST: IDLE, PAUSED: IDLE, CALIBRATING: IDLE},
    "calibrate": {IDLE: CALIBRATING},
    "calibrated": {CALIBRATING: IDLE},
    "fault": {SEARCHING: FAULT, TRACKING: FAULT, LOST: FAULT, PAUSED: FAULT, CALIBRATING: FAULT, IDLE: FAULT},
    "reset": {ESTOP: IDLE, FAULT: IDLE},
}


class StateMachine:
    def __init__(self, reacquire_window_s: float = 5.0):
        self._lock = threading.RLock()
        self.state = IDLE
        self.since = clock.now()
        self.reason = "startup"
        self.reacquire_window = reacquire_window_s
        self.lost_at: Optional[float] = None       # when the current re-acquire window started
        self.listeners: list[Callable[[str, str, str, float], None]] = []
        self.transitions = 0
        self.last_transition_us = 0.0

    def fire(self, event: str, reason: str = "") -> bool:
        t0 = clock.now()
        with self._lock:
            if event == "estop":
                new = ESTOP
            else:
                new = _ALLOWED.get(event, {}).get(self.state)
            if new is None or (new == self.state and event != "estop"):
                return False
            old = self.state
            self.state = new
            self.since = clock.now()
            self.reason = reason or event
            self.lost_at = self.since if event == "lost" else (None if new != SEARCHING else self.lost_at)
            if event in ("start", "resume"):
                self.lost_at = None
            self.transitions += 1
            listeners = list(self.listeners)
        self.last_transition_us = (clock.now() - t0) * 1e6
        log.info("state %s -> %s (%s)", old, new, self.reason)
        for cb in listeners:
            try:
                cb(old, new, self.reason, self.since)
            except Exception:
                log.exception("state listener failed")
        return True

    def tick(self) -> None:
        """Time-based transitions (called by the 100 Hz control loop)."""
        with self._lock:
            expired = (self.state == SEARCHING and self.lost_at is not None
                       and clock.now() - self.lost_at > self.reacquire_window)
        if expired:
            self.fire("reacquire_timeout", f"not re-acquired within {self.reacquire_window:.1f} s")

    def reacquire_remaining(self) -> Optional[float]:
        if self.state == SEARCHING and self.lost_at is not None:
            return max(0.0, self.reacquire_window - (clock.now() - self.lost_at))
        return None

    def snapshot(self) -> dict:
        return {"state": self.state, "since": self.since, "reason": self.reason,
                "reacquire_remaining_s": self.reacquire_remaining(), "transitions": self.transitions}
