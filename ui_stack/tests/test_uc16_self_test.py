"""Acceptance test for UC-16: startup self-test (ES-10.1).

UC-16 acceptance test (PRD v2, section 7):
    On launch, self-test reports pass or fail for the camera, LiDAR, gimbal
    link, storage and compute within 30 s. Any failure blocks Start and names
    the failed check.

Hardware checks are replaced with fakes; the storage check uses the real
disk-usage call on the test machine.
"""
import time

import pytest

from aimbot.uc16_self_test import (
    Check,
    SessionStartGate,
    StartBlocked,
    run_self_test,
    storage_check,
)


def _checks(**overrides):
    defaults = {
        "camera": Check("camera", lambda: (True, "1936x1464 frame received")),
        "lidar": Check("lidar", lambda: (True, "range 3.05 m")),
        "gimbal_link": Check("gimbal_link", lambda: (True, "heartbeat 50 Hz")),
        "storage": storage_check(".", min_free_gb=0.001),
        "compute": Check("compute", lambda: (True, "GPU available")),
    }
    defaults.update(overrides)
    return list(defaults.values())


def test_all_checks_report_and_start_is_allowed_within_30_s():
    report = run_self_test(_checks())
    print(f"UC-16 metric: self-test finished in {report.duration_s:.2f} s")
    assert report.duration_s <= 30
    assert {r.name for r in report.results} == {"camera", "lidar", "gimbal_link", "storage", "compute"}
    assert report.passed
    gate = SessionStartGate()
    gate.record(report)
    gate.start()  # does not raise


def test_a_failed_check_blocks_start_and_is_named():
    report = run_self_test(_checks(lidar=Check("lidar", lambda: (False, "no reply from ESP32"))))
    gate = SessionStartGate()
    gate.record(report)
    with pytest.raises(StartBlocked) as blocked:
        gate.start()
    assert blocked.value.failed == ["lidar"]
    assert "lidar" in str(blocked.value)


def test_a_hung_check_times_out_and_the_run_still_finishes_within_30_s():
    hung = Check("gimbal_link", lambda: time.sleep(60), timeout_s=1.0)
    crashed = Check("camera", lambda: 1 / 0)
    t0 = time.monotonic()
    report = run_self_test(_checks(gimbal_link=hung, camera=crashed))
    assert time.monotonic() - t0 <= 30
    assert sorted(report.failed_names) == ["camera", "gimbal_link"]
    detail = {r.name: r.detail for r in report.results}
    assert "timed out" in detail["gimbal_link"]
    assert "error" in detail["camera"]


def test_start_is_blocked_until_self_test_has_run():
    with pytest.raises(StartBlocked):
        SessionStartGate().start()
