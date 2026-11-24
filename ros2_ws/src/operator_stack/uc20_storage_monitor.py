"""UC-20: Warn before storage runs out (TR-9).

Raw recording writes about 368 MB/s, so a 1 TB drive fills in about
45 minutes. ``StorageMonitor`` polls free space on the recording drive on its
own thread. Below the warning threshold it raises an operator warning; below
the stop threshold it tells the recorder to finish its current frame and close
its files cleanly. Tracking runs on separate threads and never waits on
storage, so it continues after recording stops.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class StorageLevel(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class StorageStatus:
    free_bytes: int
    level: StorageLevel
    minutes_left: float


RAW_RECORDING_BYTES_PER_S = 368e6


class StorageMonitor:
    def __init__(self, path: str, warn_free_bytes: float, stop_free_bytes: float,
                 on_warning: Callable[[StorageStatus], None],
                 on_stop: Callable[[StorageStatus], None],
                 poll_interval_s: float = 0.25,
                 usage_fn: Callable[[str], object] = shutil.disk_usage,
                 write_rate_bytes_per_s: float = RAW_RECORDING_BYTES_PER_S):
        if stop_free_bytes >= warn_free_bytes:
            raise ValueError("stop threshold must be below the warning threshold")
        if poll_interval_s > 1.0:
            raise ValueError("poll at least once a second so warnings arrive within 1 s")
        self.path = path
        self.warn_free = warn_free_bytes
        self.stop_free = stop_free_bytes
        self.on_warning = on_warning
        self.on_stop = on_stop
        self.poll_interval_s = poll_interval_s
        self.usage_fn = usage_fn
        self.write_rate = write_rate_bytes_per_s
        self.level = StorageLevel.OK
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()

    def poll(self) -> StorageStatus:
        free = int(self.usage_fn(self.path).free)
        if free <= self.stop_free:
            level = StorageLevel.STOPPED
        elif free <= self.warn_free:
            level = StorageLevel.WARNING
        else:
            level = StorageLevel.OK
        status = StorageStatus(free, level, max(0.0, (free - self.stop_free) / self.write_rate / 60))
        if level == StorageLevel.WARNING and self.level == StorageLevel.OK:
            self.on_warning(status)
        if level == StorageLevel.STOPPED and self.level != StorageLevel.STOPPED:
            if self.level == StorageLevel.OK:
                self.on_warning(status)
            self.on_stop(status)
        if self.level != StorageLevel.STOPPED:
            self.level = level
        return status

    def start(self) -> None:
        self._halt.clear()
        self._thread = threading.Thread(target=self._run, name="storage-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        while not self._halt.is_set():
            self.poll()
            self._halt.wait(self.poll_interval_s)


class StoppableRecorder:
    """Recorder that finishes the frame in progress and closes cleanly on request."""

    def __init__(self, path: str):
        self._file = open(path, "wb")
        self._lock = threading.Lock()
        self.recording = True
        self.frames_written = 0
        self.frame_bytes: int | None = None

    def write(self, frame_bytes: bytes) -> bool:
        with self._lock:
            if not self.recording:
                return False
            self.frame_bytes = len(frame_bytes)
            self._file.write(frame_bytes)
            self.frames_written += 1
            return True

    def stop_cleanly(self, _status: StorageStatus | None = None) -> None:
        with self._lock:
            if self.recording:
                self.recording = False
                self._file.flush()
                self._file.close()
