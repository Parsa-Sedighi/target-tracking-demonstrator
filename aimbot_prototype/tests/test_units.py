"""Fast unit tests (no threads, no hardware):  python -m pytest tests -q"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest

from aimbot.bus import Bus
from aimbot.config import deep_merge, load_config
from aimbot.control.controller import TrackingController
from aimbot.control.kalman import TargetKF
from aimbot.gimbal.esp32_bridge import E2J_GIMBAL, FrameParser, crc16_ccitt, encode
from aimbot.gimbal.mavlink_gremsy import euler_to_quat, quat_to_euler
from aimbot.perception.association import Associator, TargetSelector
from aimbot.perception.geometry import Intrinsics, iou, pixel_to_azel, world_to_pixel
from aimbot.sensors.lidar import TF02Parser, decode_reading, encode_frame
from aimbot.state_machine import ESTOP, IDLE, LOST, SEARCHING, TRACKING, StateMachine
from aimbot.types import Detection, GimbalState


# ---------------------------------------------------------------- LiDAR (AIM-113/114)
def test_tf02_frame_roundtrip():
    p = TF02Parser()
    out = p.feed(encode_frame(1234, 800, 41.0))
    assert out == [(1234, 800, int(round((41.0 + 256) * 8)))]
    rd = decode_reading(0.0, *out[0])
    assert rd.valid and abs(rd.distance_m - 12.34) < 1e-9 and abs(rd.temp_c - 41.0) < 0.2


def test_tf02_zero_drop_with_checksum_rejection():
    """100 s at 100 Hz, 2 % corrupted frames, random chunking: every good frame decoded, every bad one rejected."""
    rng = random.Random(0)
    p = TF02Parser()
    stream, good = bytearray(), 0
    for i in range(10000):
        f = bytearray(encode_frame(100 + i % 400, 1000, 35))
        if rng.random() < 0.02:
            f[rng.randrange(2, 9)] ^= 0x5A
        else:
            good += 1
        if rng.random() < 0.01:
            stream += b"\x59\x00\x59"  # junk that looks like a header
        stream += f
    out = []
    i = 0
    while i < len(stream):
        n = rng.randint(1, 64)
        out += p.feed(bytes(stream[i:i + n]))
        i += n
    assert len(out) == good
    assert p.checksum_errors > 0


def test_tf02_unreliable_and_saturated():
    assert not decode_reading(0, 4500, 30, 2300).valid      # strength < 60 -> Dist 4500
    assert not decode_reading(0, 0, 65535, 2300).valid      # saturated
    assert decode_reading(0, 300, 61, 2300).valid


# ---------------------------------------------------------------- geometry
@pytest.mark.parametrize("pan,tilt", [(0, 0), (0.4, 0.1), (-1.2, 0.6), (2.5, -0.5)])
def test_pixel_world_roundtrip(pan, tilt):
    intr = Intrinsics.from_lens(1936, 1464, 4.5, 8.0)
    for u, v in ((968, 732), (100, 50), (1800, 1400)):
        az, el = pixel_to_azel(intr, u, v, pan, tilt)
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]) * 5.0
        uu, vv, _ = world_to_pixel(intr, d, pan, tilt)
        assert abs(uu - u) < 1e-6 and abs(vv - v) < 1e-6


def test_blackfly_kowa_fov_meets_sdr():
    intr = Intrinsics.from_lens(1936, 1464, 4.5, 8.0)
    assert intr.hfov_deg > 40                                  # ES-06.3
    assert (0.0762 / 6.096) * intr.fx >= 20                    # ES-06.2: 3 in at 20 ft >= 20 px
    assert abs(intr.hfov_deg - 57.1) < 0.2


def test_sign_conventions():
    intr = Intrinsics.from_hfov(640, 480, 60)
    az, el = pixel_to_azel(intr, 600, intr.cy, 0, 0)    # right of centre
    assert az > 0 and abs(el) < 1e-9
    az, el = pixel_to_azel(intr, 320, 50, 0, 0)     # above centre
    assert el > 0
    az, el = pixel_to_azel(intr, intr.cx, intr.cy, 0.3, 0.2)
    assert abs(az - 0.3) < 1e-9 and abs(el - 0.2) < 1e-9


def test_quaternion_roundtrip():
    for r, p, y in ((0, 0.3, 1.0), (0, -0.7, -2.0), (0, 1.2, 0.1)):
        rr, pp, yy = quat_to_euler(euler_to_quat(r, p, y))
        assert abs(pp - p) < 1e-6 and abs(yy - y) < 1e-6


# ---------------------------------------------------------------- Kalman / control
def test_kalman_tracks_constant_rate_and_predicts():
    kf = TargetKF({"q": 2.0})
    r = (0.001) ** 2
    rng = random.Random(1)
    for i in range(120):
        t = i / 60
        kf.update(t, 0.5 * t + rng.gauss(0, 0.001), -0.2 * t + rng.gauss(0, 0.001), r)
    est = kf.predict_to(2.0 + 0.05)
    # q = 2 favours smooth position over a precise rate: rate within 0.2 rad/s, prediction within 0.6 deg
    assert abs(est.az_rate - 0.5) < 0.2 and abs(est.el_rate + 0.2) < 0.2
    assert abs(est.az - 0.5 * 2.05) < 0.01


def test_kalman_gate_rejects_outlier():
    kf = TargetKF({"q": 2.0})
    for i in range(30):
        kf.update(i / 60, 0.0, 0.0, 1e-6)
    assert not kf.update(31 / 60, 0.3, 0.0, 1e-6)    # 17 deg jump in one frame


def test_controller_angle_mode_leads_target():
    c = TrackingController({"mode": "angle", "lead_ms": 50, "max_rate_dps": 100})
    kf = TargetKF({})
    for i in range(60):
        kf.update(i / 60, 0.2 + 0.3 * i / 60, 0.0, 1e-6)
    g = GimbalState(t=1.0, pan=0.2 + 0.3, tilt=0.0, connected=True)
    cmd = None
    for k in range(5):
        cmd = c.compute(1.0 + k * 0.01, kf.predict_to, g, (0.0, 0.0))
    assert cmd.mode == "angle" and cmd.pan > 0.5      # ahead of the current target position


# ---------------------------------------------------------------- state machine (FR-04, ES-03)
def test_state_machine_reacquire_window_and_estop():
    sm = StateMachine(reacquire_window_s=0.05)
    assert sm.fire("start") and sm.state == SEARCHING
    assert sm.fire("acquired") and sm.state == TRACKING
    assert sm.fire("lost") and sm.state == SEARCHING and sm.reacquire_remaining() is not None
    time.sleep(0.07)
    sm.tick()
    assert sm.state == LOST
    assert sm.fire("acquired") and sm.state == TRACKING
    t0 = time.perf_counter()
    sm.fire("estop")
    assert sm.state == ESTOP and (time.perf_counter() - t0) < 0.010      # PRD: transition <= 10 ms
    assert not sm.fire("start")                                          # latched
    assert sm.fire("reset") and sm.state == IDLE


# ---------------------------------------------------------------- association / selection (ES-01.4)
def test_association_keeps_ids_when_paths_cross():
    a = Associator({})
    ids = {}
    for i in range(40):
        x1 = 100 + i * 10
        x2 = 500 - i * 10
        dets = [Detection((x1, 200, x1 + 40, 240), "drone", 0.9), Detection((x2, 205, x2 + 40, 245), "hand", 0.9)]
        a.update(dets, i / 30, i)
        for d in dets:
            ids.setdefault(d.cls, set()).add(d.track_id)
    assert len(ids["drone"]) == 1 and len(ids["hand"]) == 1


def test_selector_priority_and_class_filter():
    a = Associator({})
    dets = [Detection((0, 0, 10, 10), "hand", 0.95), Detection((50, 50, 60, 60), "drone", 0.7)]
    for i in range(3):
        trs = a.update([Detection(d.bbox, d.cls, d.conf) for d in dets], i / 30, i)
    sel = TargetSelector({"target_class": "auto", "priority": ["drone", "ball"], "min_conf": 0.5, "min_hits": 2})
    assert sel.select(trs, (640, 480), (320, 240)).cls == "drone"
    sel.target_class = "ball"
    assert sel.select(trs, (640, 480), (320, 240)) is None


# ---------------------------------------------------------------- ESP32 protocol
def test_esp32_framing_and_crc():
    import struct

    assert crc16_ccitt(b"123456789") == 0x29B1                 # CRC-16/CCITT-FALSE check value
    msg = encode(E2J_GIMBAL, 7, struct.pack("<IiiiB", 1000, 1234, -500, 0, 1))
    p = FrameParser()
    junk = b"\x00\xA5\x13"
    out = p.feed(junk + msg[:5]) + p.feed(msg[5:] + msg)
    assert len(out) == 2 and out[0][0] == E2J_GIMBAL and out[0][1] == 7
    bad = bytearray(msg)
    bad[8] ^= 0xFF
    assert p.feed(bytes(bad)) == [] and p.crc_errors == 1


# ---------------------------------------------------------------- bus / config
def test_bus_dispatch_is_fast():
    b = Bus()
    got = []
    b.subscribe("/x", lambda t, m: got.append(m))
    t0 = time.perf_counter()
    for i in range(1000):
        b.publish("/x", i)
    assert (time.perf_counter() - t0) < 0.05 and len(got) == 1000   # PRD story 5: 1000 entries < 50 ms


def test_configs_load():
    for name in ("default", "webcam", "jetson_hw", "jetson_esp32", "jetson_headless", "sim_yolo", "pc_blackfly"):
        cfg = load_config(name)
        assert cfg.get_path("control.rate_hz") == 100
        assert cfg.get_path("camera.active") in cfg.sensors
    assert load_config("webcam").sensors.webcam.backend == "opencv"
    assert "vis" not in load_config("webcam").sensors          # _replace works
    assert load_config("jetson_esp32").gimbal.backend == "esp32"


def test_deep_merge_replace():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    assert deep_merge(base, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}, "b": 1}
    assert deep_merge(base, {"a": {"_replace": True, "z": 9}}) == {"a": {"z": 9}, "b": 1}


def test_template_tracker_follows_and_refuses_blank():
    from aimbot.perception.micro_tracker import TemplateTracker

    rng = np.random.default_rng(0)
    bg = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    trk = TemplateTracker({})
    assert not trk.init(np.full((240, 320, 3), 128, np.uint8), (100, 100, 140, 140), 0.0)   # no texture
    assert trk.init(bg, (100, 100, 140, 140), 0.0)
    moved = np.roll(bg, (3, 5), axis=(0, 1))          # scene shifts 5 px right, 3 px down
    ok, bb, score = trk.update(moved, 1 / 60)
    assert ok and score > 0.9
    assert abs((bb[0] + bb[2]) / 2 - 125) < 0.5 and abs((bb[1] + bb[3]) / 2 - 123) < 0.5


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_model_class_names_map_regardless_of_case():
    from aimbot.perception.detectors import DEFAULT_CLASS_MAP, _lower_keys, map_class

    cmap = dict(DEFAULT_CLASS_MAP, quad_x="drone", Quad="drone")      # + detector.class_map entries
    low = _lower_keys(cmap)
    names = ("UAV", "Drone", " drone ", "Sports Ball", "quad_x", "QUAD", "Spaceship")
    assert [map_class(cmap, low, n) for n in names] == \
        ["drone", "drone", "drone", "ball", "drone", "drone", "Spaceship"]
