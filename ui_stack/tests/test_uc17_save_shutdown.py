"""Acceptance test for UC-17: save the session and shut down safely (ES-10.1).

UC-17 acceptance test (PRD v2, section 7):
    Save and Shutdown closes the session files with record count equal to
    frame count, returns the gimbal to its safe position and exits.

A simulated gimbal slews toward the commanded angle at 100 deg/s (the Gremsy
Mio's rated speed), so the shutdown sequence really has to wait for it.
"""
import threading
import time

import numpy as np

from aimbot.uc17_save_shutdown import SaveAndShutdown, SessionRecorder, verify_session


class SlewingGimbal:
    RATE_DEG_S = 100.0

    def __init__(self, pan, tilt):
        self._pos = [pan, tilt]
        self._target = [pan, tilt]
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def move_to(self, pan, tilt):
        with self._lock:
            self._advance()
            self._target = [pan, tilt]

    def position(self):
        with self._lock:
            self._advance()
            return tuple(self._pos)

    def _advance(self):
        now = time.monotonic()
        step = self.RATE_DEG_S * (now - self._last)
        self._last = now
        for i in range(2):
            delta = self._target[i] - self._pos[i]
            self._pos[i] += max(-step, min(step, delta))


def test_save_and_shutdown_closes_files_matches_counts_and_parks_the_gimbal(tmp_path):
    rng = np.random.default_rng(17)
    recorder = SessionRecorder(tmp_path / "session_001", frame_shape=(120, 160))
    for i in range(300):
        frame = rng.integers(0, 256, (120, 160), dtype=np.uint8)
        recorder.write(frame, {"timestamp_ns": 1_000_000 * i, "track_state": "TRACKING"})

    gimbal = SlewingGimbal(pan=40.0, tilt=25.0)
    t0 = time.monotonic()
    report = SaveAndShutdown(recorder, gimbal).run()
    print(f"UC-17 metric: {report.frames_on_disk} frames / {report.records_on_disk} records, "
          f"gimbal parked in {time.monotonic() - t0:.2f} s")

    assert report.files_closed
    assert report.frames_on_disk == report.records_on_disk == 300
    assert report.gimbal_safe
    assert report.ok


def test_no_frames_can_be_written_after_shutdown(tmp_path):
    recorder = SessionRecorder(tmp_path / "s", frame_shape=(4, 4))
    recorder.write(np.zeros((4, 4), np.uint8), {"timestamp_ns": 0})
    SaveAndShutdown(recorder, SlewingGimbal(0.0, -30.0)).run()
    try:
        recorder.write(np.zeros((4, 4), np.uint8), {"timestamp_ns": 1})
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed session accepted a frame")
    assert verify_session(tmp_path / "s") == (1, 1)


def test_shutdown_reports_failure_if_the_gimbal_never_reaches_safe_position(tmp_path):
    class StuckGimbal:
        def move_to(self, pan, tilt):
            pass

        def position(self):
            return (90.0, 45.0)

    recorder = SessionRecorder(tmp_path / "s", frame_shape=(4, 4))
    report = SaveAndShutdown(recorder, StuckGimbal(), timeout_s=0.2).run()
    assert report.files_closed and not report.gimbal_safe and not report.ok
