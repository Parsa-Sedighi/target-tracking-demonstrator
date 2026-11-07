"""Closed-loop integration tests in the simulator (real threads, real time, ~1 min total).

    python -m pytest tests -q -m "not slow"     # unit tests only
    python -m pytest tests -q                   # everything
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimbot.config import load_config  # noqa: E402
from aimbot.engine import Engine  # noqa: E402

pytestmark = pytest.mark.slow


def _engine(tmp_path, scenario: str, target: str = "auto", extra=()):
    cfg = load_config("default.yaml", [f"calibration.file={tmp_path / 'calib.json'}", f"logging.dir={tmp_path / 'logs'}",
                                       f"sim.scenario={scenario}", f"seek.target_class={target}", *extra])
    e = Engine(cfg)
    e.start()
    return e


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_track_card_meets_latency_and_fov(tmp_path):
    e = _engine(tmp_path, "card_walk", "card")
    try:
        e.command({"cmd": "start"})
        time.sleep(8)
        assert e.sm.state == "TRACKING"
        lat = e.latency_stats()
        assert lat["cmd_p95"] < 100 and lat["offset_p95"] < 100          # ES-05.1 / NFR-04
        s = e.summary()
        assert s["in_fov_ratio"] is not None and s["in_fov_ratio"] > 0.95  # ES-02.2
        assert e.loop.hz > 95                                             # 100 Hz heartbeat
    finally:
        e.shutdown()
    session = sorted((tmp_path / "logs").iterdir())[-1]
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "analyze_run.py"), str(session), "--no-plots"],
                         capture_output=True, text=True, timeout=60)
    assert "FAIL] UC-6" not in out.stdout and "FAIL] FR-07" not in out.stdout, out.stdout


def test_keeps_drone_when_hand_crosses(tmp_path):
    e = _engine(tmp_path, "crossing", "drone")
    try:
        e.command({"cmd": "start"})
        t_end = time.time() + 10
        wrong = 0
        while time.time() < t_end:
            lk = e.lock
            if lk is not None and lk.cls != "drone":
                wrong += 1
            time.sleep(0.05)
        assert wrong == 0                                                 # ES-01.4
        assert e.stats["acquisitions"] >= 1
    finally:
        e.shutdown()


def test_reacquire_after_occlusion(tmp_path):
    e = _engine(tmp_path, "occlusion", "drone")
    try:
        e.command({"cmd": "start"})
        time.sleep(16)
        times = e.stats["reacquire_times_s"]
        assert e.stats["losses"] >= 1, "scenario should hide the drone behind the pillar"
        assert times and max(times) <= 2.0                                # ES-03.1
    finally:
        e.shutdown()


def test_estop_stops_motion_and_latches(tmp_path):
    e = _engine(tmp_path, "drone_lateral", "drone")
    try:
        e.command({"cmd": "start"})
        time.sleep(3.0)
        e.command({"cmd": "estop"})
        time.sleep(0.6)
        g1 = e.gimbal.state()
        time.sleep(0.5)
        g2 = e.gimbal.state()
        assert e.sm.state == "ESTOP"
        assert abs(math.degrees(g2.pan - g1.pan)) < 0.5                   # no motion after E-stop
        assert not e.command({"cmd": "start"})["ok"]
        e.command({"cmd": "reset"})
        assert e.sm.state == "IDLE"
    finally:
        e.shutdown()


def test_calibration_fixes_lidar_bias(tmp_path):
    e = _engine(tmp_path, "card_walk")
    try:
        time.sleep(1.0)
        # the sim LiDAR has a 2.5 cm bias; with nothing in the beam it sees the front wall at 20 ft
        e.command({"cmd": "calibrate", "distance_m": 6.096})
        t_end = time.time() + 12
        while time.time() < t_end and e.sm.state != "IDLE":
            time.sleep(0.1)
        time.sleep(0.2)
        res = e.calib_result
        assert res.get("ok"), res
        step = [s for s in res["steps"] if s["step"] == "lidar range offset"][0]
        assert step["residual_in"] <= 2.0                                 # UC-7: +-2 in
        assert json.loads((tmp_path / "calib.json").read_text())["range_offset_m"] < 0
    finally:
        e.shutdown()


def test_click_to_track_unclassified_region(tmp_path):
    e = _engine(tmp_path, "card_walk", extra=["detector.enabled=false"])
    try:
        time.sleep(1.0)
        f = e.framebuf.latest()
        card = [t for t in f.meta["truth"] if t["cls"] == "card"][0]      # where the (unclassified) card is
        x1, y1, x2, y2 = card["bbox"]
        e.command({"cmd": "designate", "x": x1 / f.width, "y": y1 / f.height,
                   "w": (x2 - x1) / f.width, "h": (y2 - y1) / f.height})    # FR-09 drag-a-box
        time.sleep(3.0)
        assert e.sm.state == "TRACKING" and e.lock is not None and e.lock.mode == "manual"
        e.command({"cmd": "clear_target"})
        time.sleep(0.3)
        assert e.sm.state == "SEARCHING" and e.lock is None                 # released -> seeking again
    finally:
        e.shutdown()


def test_operator_gateway_roundtrip(tmp_path):
    from aimbot.operator.gateway import RemoteEngine

    port = _free_port()
    e = _engine(tmp_path, "card_walk", "card", extra=["gateway.enabled=true", f"gateway.port={port}"])
    r = RemoteEngine("127.0.0.1", port)
    try:
        r.start()
        t_end = time.time() + 5
        while time.time() < t_end and (r.snapshot()["frame"] is None or not r.snapshot().get("telemetry")):
            time.sleep(0.1)
        s = r.snapshot()
        assert s["frame"] is not None and s["frame"].shape[1] <= 960
        assert r.command({"cmd": "start"})["ok"]
        t_end = time.time() + 5
        while time.time() < t_end and e.sm.state != "TRACKING":
            time.sleep(0.1)
        assert e.sm.state == "TRACKING"
        time.sleep(0.5)
        assert r.snapshot()["state"]["state"] == "TRACKING"
    finally:
        r.stop()
        e.shutdown()


def test_mavlink_backend_against_fake_gremsy(tmp_path):
    port = _free_port()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    fake = subprocess.Popen([sys.executable, str(ROOT / "tools" / "fake_gremsy.py"), "--listen", str(port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        e = _engine(tmp_path, "card_walk", "card", extra=["gimbal.backend=gremsy", f"gimbal.port=tcp:127.0.0.1:{port}",
                                                          "gimbal.protocol=v1", "control.lead_ms=60"])
        try:
            assert e.gimbal.kind == "gremsy", e.alerts
            e.command({"cmd": "start"})
            time.sleep(6)
            assert e.sm.state == "TRACKING"
            assert e.gimbal.fb_rate.value() > 20
            assert abs(e.gimbal.state().pan) > 0.01          # the (emulated) gimbal is really moving
        finally:
            e.shutdown()
    finally:
        fake.terminate()
        fake.wait(5)
