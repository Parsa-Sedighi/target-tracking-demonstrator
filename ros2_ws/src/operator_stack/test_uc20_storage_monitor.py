"""Acceptance test for UC-20: warn before storage runs out (TR-9).

UC-20 acceptance test (PRD v2, section 7):
    With free space set below the threshold, a warning appears within 1 s,
    and recording stops cleanly before the drive is full while tracking
    continues.

A fake disk reports free space that shrinks as frames are written. The
recorder, the tracking loop and the monitor each run on their own thread.
"""
import os
import threading
import time
from types import SimpleNamespace

import pytest

from aimbot.uc20_storage_monitor import StorageLevel, StorageMonitor, StoppableRecorder

FRAME = bytes(64 * 1024)


class FakeDisk:
    def __init__(self, free_bytes):
        self.free = free_bytes
        self.lock = threading.Lock()

    def usage(self, _path):
        with self.lock:
            return SimpleNamespace(free=self.free)

    def consume(self, n):
        with self.lock:
            self.free -= n
            if self.free < 0:
                raise OSError("disk full")


def test_warning_within_1_s_then_clean_stop_while_tracking_continues(tmp_path):
    disk = FakeDisk(free_bytes=40 * len(FRAME))
    recorder = StoppableRecorder(str(tmp_path / "frames.bin"))
    warnings, stops = [], []
    monitor = StorageMonitor(
        str(tmp_path), warn_free_bytes=30 * len(FRAME), stop_free_bytes=10 * len(FRAME),
        on_warning=lambda s: warnings.append((time.monotonic(), s)),
        on_stop=lambda s: (stops.append(s), recorder.stop_cleanly(s)),
        poll_interval_s=0.1, usage_fn=disk.usage,
    )
    crossed_warning_at = {}
    tracking_updates = [0]
    done = threading.Event()

    def record_loop():
        while not done.is_set() and recorder.recording:
            if recorder.write(FRAME):
                disk.consume(len(FRAME))
                if disk.free <= 30 * len(FRAME) and "t" not in crossed_warning_at:
                    crossed_warning_at["t"] = time.monotonic()
            time.sleep(0.02)

    def tracking_loop():
        while not done.is_set():
            tracking_updates[0] += 1
            time.sleep(1 / 130)

    threads = [threading.Thread(target=record_loop), threading.Thread(target=tracking_loop)]
    monitor.start()
    for t in threads:
        t.start()
    time.sleep(1.5)
    updates_at_stop = tracking_updates[0]
    time.sleep(0.3)
    done.set()
    for t in threads:
        t.join()
    monitor.stop()

    warning_delay = warnings[0][0] - crossed_warning_at["t"]
    print(f"UC-20 metric: warning {warning_delay * 1000:.0f} ms after crossing threshold; "
          f"recording stopped with {disk.free // len(FRAME)} frames of space left")
    assert warning_delay <= 1.0
    assert len(stops) == 1 and not recorder.recording
    assert disk.free > 0, "drive filled before recording stopped"
    assert os.path.getsize(tmp_path / "frames.bin") == recorder.frames_written * len(FRAME)
    assert tracking_updates[0] > updates_at_stop, "tracking stopped with the recorder"
    assert monitor.level == StorageLevel.STOPPED


def test_status_reports_minutes_of_recording_left():
    disk = FakeDisk(free_bytes=int(368e6 * 60 * 10 + 50e9))  # 10 min above the 50 GB stop line
    monitor = StorageMonitor(".", warn_free_bytes=300e9, stop_free_bytes=50e9,
                             on_warning=lambda s: None, on_stop=lambda s: None,
                             usage_fn=disk.usage)
    status = monitor.poll()
    assert status.level == StorageLevel.WARNING
    assert status.minutes_left == pytest.approx(10.0, rel=0.01)


def test_thresholds_are_validated():
    with pytest.raises(ValueError):
        StorageMonitor(".", warn_free_bytes=10, stop_free_bytes=20,
                       on_warning=print, on_stop=print)
    with pytest.raises(ValueError):
        StorageMonitor(".", warn_free_bytes=20, stop_free_bytes=10,
                       on_warning=print, on_stop=print, poll_interval_s=2.0)
