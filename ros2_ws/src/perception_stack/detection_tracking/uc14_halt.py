"""UC-14: Halt all gimbal motion from the UI (FR-05).

Every gimbal rate command passes through ``GimbalCommandGate``. Pressing Halt
sends a zero-rate command immediately, from the UI thread, without waiting
for the tracking loop. Until the operator presses Start again, any command the
tracking loop still sends is replaced with zero. Every command is logged with
a timestamp, so the halt latency can be checked from the command log.

This is the software halt. The hardware E-stop (ES-11.1) cuts actuator power
independently and is tested by ECE.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

SendFn = Callable[[float, float], None]


@dataclass(frozen=True)
class CommandLogEntry:
    t: float
    pan_rate: float
    tilt_rate: float
    source: str
    blocked: bool


class GimbalCommandGate:
    """Thread-safe gate between the tracking loop and the gimbal port."""

    def __init__(self, send: SendFn, clock: Callable[[], float] = time.monotonic):
        self._send = send
        self._clock = clock
        self._lock = threading.Lock()
        self._halted = True  # nothing moves until a session starts
        self.halted_at: float | None = None
        self.log: list[CommandLogEntry] = []

    @property
    def halted(self) -> bool:
        return self._halted

    def start(self) -> None:
        with self._lock:
            self._halted = False
            self.halted_at = None

    def halt(self) -> float:
        """Stop all motion now. Returns the time the zero command was sent."""
        with self._lock:
            self._halted = True
            self.halted_at = self._clock()
            self._send(0.0, 0.0)
            sent_at = self._clock()
            self.log.append(CommandLogEntry(sent_at, 0.0, 0.0, "halt", False))
            return sent_at

    def command(self, pan_rate: float, tilt_rate: float, source: str = "tracker") -> bool:
        """Forward a rate command unless halted. Returns True if it was forwarded."""
        with self._lock:
            if self._halted:
                self._send(0.0, 0.0)
                self.log.append(CommandLogEntry(self._clock(), 0.0, 0.0, source, True))
                return False
            self._send(pan_rate, tilt_rate)
            self.log.append(CommandLogEntry(self._clock(), pan_rate, tilt_rate, source, False))
            return True
