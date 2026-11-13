"""PROJECT23-11 / UC-07 preliminary mock tests. Physical hardware remains untested."""

import json
import math
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton

from aimbot import clock
from aimbot.config import load_config
from aimbot.engine import Engine
from aimbot.sensors.lidar import NoLidar
from aimbot.types import GimbalCommand
from aimbot.ui.main_window import MainWindow

DISTANCE_M = 3.048  # 10 feet; the board face is positioned independently in SimWorld
INCH_M = 0.0254


def wait_for(predicate, timeout=10, app=None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if app:
            app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for simulator / calibration")


@pytest.fixture
def engine(tmp_path):
    cfg = load_config("calibration_mock", [f"calibration.file={tmp_path / 'calibration.json'}",
                                           f"logging.dir={tmp_path / 'logs'}", "safety.home_on_shutdown=false"])
    e = Engine(cfg)
    yield e
    e.shutdown()


@pytest.fixture
def running(engine):
    engine.start()
    wait_for(lambda: engine.framebuf.latest() is not None and engine.gimbal.feedback_age() < 0.1
             and engine.lidar.latest() is not None)
    assert engine.gimbal.kind == engine.lidar.kind == "sim"
    assert engine.camera.world is engine.world
    return engine


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def calibrate(e):
    assert e.command({"cmd": "calibrate", "distance_m": DISTANCE_M})["ok"]
    wait_for(lambda: bool(e.calib_result))
    return e.calib_result


@pytest.mark.slow
def test_gui_calibration_at_ten_feet(running, app, record_property):
    """Click the real GUI, verify fresh feedback/range, select the board with the real mouse path."""
    e = running
    e.gimbal.command(GimbalCommand("angle", math.radians(8), math.radians(-4)))
    wait_for(lambda: abs(math.degrees(e.gimbal.state().pan) - 8) < 0.1)
    e.command({"cmd": "set_boresight", "x": 0.65, "y": 0.35})
    window = MainWindow(e)
    window.show()
    try:
        wait_for(lambda: window.video._target.width() > 0, app=app)
        window.spin_dist.setValue(10.0)
        button = next(b for b in window.findChildren(QPushButton) if b.text().startswith("Calibrate ("))
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        wait_for(lambda: bool(e.calib_result), app=app)
        result = e.calib_result
        assert result["ok"], result
        assert result["preliminary_mock"]
        assert result["known_distance_m"] == pytest.approx(DISTANCE_M)
        assert e.sm.state == "IDLE"
        assert all(s["ok"] for s in result["steps"])
        zero = next(s for s in result["steps"] if s["step"] == "encoder zero")
        assert abs(zero["pan_deg"]) < 0.3 and abs(zero["tilt_deg"]) < 0.3

        # A new window of readings after the fit: do not validate with the calibration samples.
        sample_time = clock.now()
        wait_for(lambda: e.lidar.latest().t > sample_time + 0.3, app=app)
        measured, count = e.lidar.mean_over(0.25)
        error_in = abs(measured - DISTANCE_M) / INCH_M
        assert count >= 5 and error_in <= 2.0
        assert -0.04 < e.lidar.range_offset_m < -0.01  # compensates the injected +2.5 cm bias

        frame = e.framebuf.latest()
        board = next(t for t in frame.meta["truth"] if t["id"] == 7)
        assert board["visible"] == 1.0
        assert board["pos"][0] == DISTANCE_M
        centre = board["center"]
        assert math.dist(e._aim_px(), centre) > 50  # deliberately incorrect old crosshair
        aim_button = next(b for b in window.findChildren(QPushButton) if b.text() == "Set aim point (click)")
        QTest.mouseClick(aim_button, Qt.MouseButton.LeftButton)
        r = window.video._target
        click = QPoint(round(r.x() + centre[0] / frame.width * r.width()),
                       round(r.y() + centre[1] / frame.height * r.height()))
        QTest.mouseClick(window.video, Qt.MouseButton.LeftButton, pos=click)
        wait_for(lambda: math.dist(e._aim_px(), centre) <= 2.0, app=app)
        wait_for(lambda: "PRELIMINARY MOCK" in window.lbl_cal.text(), app=app)
        crosshair_error = math.dist(e._aim_px(), centre)
        wait_for(lambda: math.dist((e.overlay["aim"][0] * frame.width,
                                    e.overlay["aim"][1] * frame.height), centre) <= 2.0, app=app)
        saved = json.loads(e.calib_path.read_text())
        restored = Engine(e.cfg)
        assert restored.calib == saved
        assert restored.gimbal.zero_pan == pytest.approx(e.gimbal.zero_pan)
        assert restored._lidar_cfg["range_offset_m"] == pytest.approx(e.lidar.range_offset_m)
        assert saved["boresight_px"] == pytest.approx(e.calib["boresight_px"])
        evidence = dict(status="PRELIMINARY MOCK - physical hardware untested", known_distance_m=DISTANCE_M,
                        measured_distance_m=measured, range_error_in=error_in, crosshair_error_px=crosshair_error,
                        encoder_pan_deg=zero["pan_deg"], encoder_tilt_deg=zero["tilt_deg"], calibration=result)
        record_property("uc07_mock_measurements", json.dumps(evidence))
        print("\nUC-07 preliminary mock measurements: " + json.dumps(evidence))
    finally:
        window.timer.stop()
        window.close()


@pytest.mark.parametrize("distance", [0, -1, "invalid", None, float("nan"), float("inf"), float("-inf")])
def test_invalid_distance_is_rejected_without_saving(engine, distance):
    assert not engine.command({"cmd": "calibrate", "distance_m": distance})["ok"]
    assert engine.sm.state == "IDLE"
    assert not engine.calib_path.exists()


@pytest.mark.parametrize("event", ["start", "calibrate", "estop"])
def test_calibration_is_rejected_while_busy(engine, event):
    engine.sm.fire(event)
    assert not engine.command({"cmd": "calibrate", "distance_m": DISTANCE_M})["ok"]
    assert not engine.calib_path.exists()


@pytest.mark.slow
@pytest.mark.parametrize("fault, expected", [
    ("missing_lidar", "no LiDAR"),
    ("stale_encoder", "no fresh gimbal"),
    ("encoder_zero", "encoder zero not confirmed"),
    ("no_range", "no valid LiDAR"),
    ("missing_verification", "no fresh LiDAR verification"),
    ("nan_verification", "no fresh LiDAR verification"),
    ("save_failure", "could not save"),
])
def test_failed_calibration_does_not_replace_saved_values(running, monkeypatch, fault, expected):
    e = running
    e.calib["boresight_px"] = [0.5, 0.5]
    assert e._save_calibration()
    saved = e.calib_path.read_bytes()
    old_zero = e.gimbal.zero_pan, e.gimbal.zero_tilt
    old_offset = e.lidar.range_offset_m
    if fault == "missing_lidar":
        e.lidar.stop()
        e.lidar = NoLidar({})
    elif fault == "stale_encoder":
        monkeypatch.setattr(e.gimbal, "feedback_age", lambda: 2.0)
    elif fault == "encoder_zero":
        real_zero = e.gimbal.zero_here
        def bad_zero():
            real_zero()
            e.gimbal.zero_pan += math.radians(5)
            return e.gimbal.zero_pan, e.gimbal.zero_tilt
        monkeypatch.setattr(e.gimbal, "zero_here", bad_zero)
    elif fault == "no_range":
        monkeypatch.setattr(e.lidar, "mean_over", lambda seconds: (None, 0))
    elif fault in ("missing_verification", "nan_verification"):
        value = None if fault == "missing_verification" else math.nan
        monkeypatch.setattr(e.lidar, "mean_over", lambda seconds: (DISTANCE_M + 0.025, 100)
                            if seconds == 1.0 else (value, 0 if value is None else 25))
    elif fault == "save_failure":
        monkeypatch.setattr(e, "_save_calibration", lambda: False)
    result = calibrate(e)
    assert not result["ok"], result
    assert expected in result["error"]
    assert e.sm.state == "IDLE"
    assert e.calib_path.read_bytes() == saved
    assert (e.gimbal.zero_pan, e.gimbal.zero_tilt) == old_zero
    assert e.lidar.range_offset_m == old_offset


@pytest.mark.slow
@pytest.mark.parametrize("residual_in, passes", [(1.99, True), (-1.99, True), (2.0, True), (-2.0, True),
                                                 (2.01, False), (-2.01, False)])
def test_range_tolerance_checks_independent_verification_samples(running, monkeypatch, residual_in, passes):
    e = running
    # Fault injection at the sensor boundary: the verification batch differs from the fit batch.
    monkeypatch.setattr(e.lidar, "mean_over", lambda seconds: (DISTANCE_M + 0.025, 100)
                        if seconds == 1.0 else (DISTANCE_M + residual_in * INCH_M, 25))
    result = calibrate(e)
    assert result["ok"] is passes, result
    assert result["steps"][-1]["residual_in"] == pytest.approx(abs(residual_in))
    assert e.calib_path.exists() is passes
    if not passes:
        assert e.lidar.range_offset_m == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("command", ["estop", "stop"])
def test_interrupt_calibration_preserves_state_and_does_not_save(running, command):
    e = running
    assert e.command({"cmd": "calibrate", "distance_m": DISTANCE_M})["ok"]
    assert e.sm.state == "CALIBRATING"
    assert not e.command({"cmd": "calibrate", "distance_m": DISTANCE_M})["ok"]
    assert not e.command({"cmd": "set_scenario", "name": "card_walk"})["ok"]
    assert not e.command({"cmd": "set_boresight", "x": 0.9, "y": 0.9})["ok"]
    e.command({"cmd": command})
    wait_for(lambda: bool(e.calib_result))
    assert not e.calib_result["ok"]
    assert e.sm.state == ("ESTOP" if command == "estop" else "IDLE")
    assert not e.calib_path.exists()


@pytest.mark.slow
def test_board_is_a_planar_surface_and_aim_can_miss_it(running):
    e = running
    assert e.world.range_along(clock.now(), 0, 0) == pytest.approx(DISTANCE_M)
    # A 10-degree pointing error must see the background, not a hard-coded 10 ft range.
    assert e.world.range_along(clock.now(), math.radians(10), 0) > DISTANCE_M + 1


@pytest.mark.slow
def test_failed_homing_does_not_zero_or_save(running, monkeypatch):
    e = running
    e.gimbal.command(GimbalCommand("angle", math.radians(8), 0))
    wait_for(lambda: abs(math.degrees(e.gimbal.state().pan) - 8) < 0.1)
    monkeypatch.setattr(e.gimbal, "home", lambda: None)  # stuck actuator
    result = calibrate(e)
    assert not result["ok"]
    assert result["steps"][0]["step"] == "home" and not result["steps"][0]["ok"]
    assert not any(s["step"] == "encoder zero" for s in result["steps"])
    assert not e.calib_path.exists()
