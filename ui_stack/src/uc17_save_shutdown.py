"""UC-17: Save the session and shut down safely (ES-10.1).

``SessionRecorder`` appends raw frames to ``frames.bin`` and one JSON record
per frame to ``records.jsonl``. ``SaveAndShutdown`` stops recording, flushes
and closes both files, re-reads them to confirm every frame has exactly one
record, sends the gimbal to its safe position and waits for it to arrive,
then reports whether shutdown is safe to complete.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class SessionRecorder:
    def __init__(self, session_dir: str | os.PathLike, frame_shape: tuple[int, int]):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frame_shape = tuple(frame_shape)
        self.frame_bytes = int(np.prod(self.frame_shape))
        self._frames = open(self.dir / "frames.bin", "wb")
        self._records = open(self.dir / "records.jsonl", "w", encoding="utf-8")
        self.frames_written = 0
        self.closed = False
        (self.dir / "manifest.json").write_text(json.dumps({
            "frame_shape": list(self.frame_shape), "dtype": "uint8",
            "frames_file": "frames.bin", "records_file": "records.jsonl",
        }))

    def write(self, frame: np.ndarray, record: dict) -> None:
        if self.closed:
            raise RuntimeError("session is closed")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != self.frame_shape:
            raise ValueError(f"frame shape {frame.shape} != {self.frame_shape}")
        self._frames.write(frame.tobytes())
        self._records.write(json.dumps({"frame_index": self.frames_written, **record}) + "\n")
        self.frames_written += 1

    def close(self) -> None:
        if self.closed:
            return
        for f in (self._frames, self._records):
            f.flush()
            os.fsync(f.fileno())
            f.close()
        self.closed = True


class GimbalPort(Protocol):
    def position(self) -> tuple[float, float]: ...
    def move_to(self, pan_deg: float, tilt_deg: float) -> None: ...


@dataclass(frozen=True)
class ShutdownReport:
    frames_on_disk: int
    records_on_disk: int
    files_closed: bool
    gimbal_safe: bool

    @property
    def counts_match(self) -> bool:
        return self.frames_on_disk == self.records_on_disk

    @property
    def ok(self) -> bool:
        return self.files_closed and self.counts_match and self.gimbal_safe


class SaveAndShutdown:
    def __init__(self, recorder: SessionRecorder, gimbal: GimbalPort,
                 safe_position: tuple[float, float] = (0.0, -30.0),
                 tolerance_deg: float = 0.5, timeout_s: float = 10.0, poll_s: float = 0.01):
        self.recorder = recorder
        self.gimbal = gimbal
        self.safe_position = safe_position
        self.tolerance = tolerance_deg
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def run(self) -> ShutdownReport:
        self.recorder.close()
        frames, records = verify_session(self.recorder.dir)
        self.gimbal.move_to(*self.safe_position)
        deadline = time.monotonic() + self.timeout_s
        safe = False
        while time.monotonic() < deadline:
            pan, tilt = self.gimbal.position()
            if abs(pan - self.safe_position[0]) <= self.tolerance and abs(tilt - self.safe_position[1]) <= self.tolerance:
                safe = True
                break
            time.sleep(self.poll_s)
        return ShutdownReport(frames, records, self.recorder.closed, safe)


def verify_session(session_dir: str | os.PathLike) -> tuple[int, int]:
    """Count frames and records on disk; records must be numbered 0..n-1."""
    session = Path(session_dir)
    manifest = json.loads((session / "manifest.json").read_text())
    frame_bytes = int(np.prod(manifest["frame_shape"]))
    size = (session / "frames.bin").stat().st_size
    if size % frame_bytes:
        raise ValueError("frames.bin holds a partial frame")
    frames = size // frame_bytes
    records = 0
    with open(session / "records.jsonl", encoding="utf-8") as f:
        for expected, line in enumerate(f):
            if json.loads(line)["frame_index"] != expected:
                raise ValueError(f"record {expected} is out of order")
            records += 1
    return frames, records
