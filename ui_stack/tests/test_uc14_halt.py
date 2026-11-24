"""Acceptance test for UC-14: Halt stops all gimbal motion at once (FR-05).

UC-14 acceptance test (PRD v2, section 7):
    During tracking, Halt drops gimbal commands to zero within 100 ms
    (command-log timestamps), and motion resumes only after Start.

A real thread plays the tracking loop, sending rate commands at 130 Hz to a
fake gimbal port, while the test presses Halt and Start from the main thread.
"""
import threading
import time

from aimbot.uc14_halt import GimbalCommandGate


class FakeGimbal:
    def __init__(self):
        self.received = []  # (time, pan_rate, tilt_rate)

    def send(self, pan_rate, tilt_rate):
        self.received.append((time.monotonic(), pan_rate, tilt_rate))


def _tracking_loop(gate, stop_event):
    while not stop_event.is_set():
        gate.command(12.0, -4.0)
        time.sleep(1 / 130)


def test_halt_zeroes_commands_within_100_ms_and_motion_resumes_only_after_start():
    gimbal = FakeGimbal()
    gate = GimbalCommandGate(gimbal.send)
    gate.start()
    stop = threading.Event()
    loop = threading.Thread(target=_tracking_loop, args=(gate, stop))
    loop.start()
    try:
        time.sleep(0.2)
        pressed_at = time.monotonic()
        gate.halt()
        time.sleep(0.3)
        after_halt = [r for r in gimbal.received if r[0] >= pressed_at]
        first_zero = next(r for r in after_halt if r[1] == 0.0 and r[2] == 0.0)
        latency_ms = (first_zero[0] - pressed_at) * 1000
        print(f"UC-14 metric: halt latency {latency_ms:.2f} ms")
        assert latency_ms <= 100
        assert all(r[1] == 0.0 and r[2] == 0.0 for r in after_halt), "motion resumed while halted"

        resumed_at = time.monotonic()
        gate.start()
        time.sleep(0.2)
        moving = [r for r in gimbal.received if r[0] > resumed_at and r[1] != 0.0]
        assert moving, "commands should flow again after Start"
    finally:
        stop.set()
        loop.join()

    halt_entries = [e for e in gate.log if e.source == "halt"]
    assert len(halt_entries) == 1


def test_gate_starts_halted_so_nothing_moves_before_start():
    gimbal = FakeGimbal()
    gate = GimbalCommandGate(gimbal.send)
    assert gate.command(5.0, 5.0) is False
    assert gimbal.received[-1][1:] == (0.0, 0.0)
