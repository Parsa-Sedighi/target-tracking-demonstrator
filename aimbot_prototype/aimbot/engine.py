"""AimBot engine: owns every thread and wires the stacks of the PRD architecture together.

    Integration stack : camera(s) -> FrameBuffer, LiDAR, gimbal backend
    Perception stack  : MacroWorker (async YOLO) -> Associator/Selector -> micro tracker -> Kalman
    Operator stack    : state machine, command(), snapshot(), calibration, self-test, OCR
    Data stack        : SessionLogger (telemetry, frames, detections, events)
    Comms             : Bus (ROS-2-shaped topics) + optional ROS 2 bridge + operator gateway

Threads: camera capture, macro detector, vision loop (every frame), control loop
(100 Hz, also the state heartbeat), LiDAR reader, gimbal I/O, logger writers, OCR.
The GUI only ever calls snapshot()/command(), so rendering can never stall control.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import __version__, clock
from .bus import Bus, Topics
from .config import PACKAGE_ROOT, ConfigNode
from .control.controller import TrackingController
from .control.kalman import TargetKF
from .data.logger import SessionLogger
from .gimbal import FixedGimbal, create_gimbal
from .perception.association import Associator, TargetSelector
from .perception.detectors import ColorBlobDetector, create_detector
from .perception.geometry import Intrinsics, azel_to_dir, iou, pixel_to_azel, world_to_pixel, wrap_pi
from .perception.macro_worker import MacroWorker
from .perception.micro_tracker import available_trackers, create_micro_tracker
from .perception.ocr import OcrWorker
from .sensors.camera import FrameBuffer, create_camera
from .sensors.lidar import ExternalLidar, NoLidar, create_lidar
from .state_machine import (ACTIVE, CALIBRATING, ESTOP, FAULT, IDLE, LOST, PAUSED, SEARCHING, TRACKING,
                            StateMachine)
from .types import Frame, GimbalCommand, TargetMeasurement

log = logging.getLogger(__name__)
D = math.degrees
R = math.radians


@dataclass
class Lock:
    mode: str                      # "classified" (from detector) | "manual" (operator click/drag)
    cls: str
    track_id: int
    conf: float
    bbox: tuple
    t_locked: float
    t_confirm: float
    frame_id: int
    misses: int = 0
    score: float = 1.0
    truth_id: Optional[int] = None
    t_good: float = 0.0       # capture time of the last accepted micro measurement
    t_reseed: float = 0.0     # capture time of the detection the tracker was last re-anchored to


def _pct(values, q) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values), q))


class Engine:
    def __init__(self, cfg: ConfigNode):
        self.cfg = cfg
        self.bus = Bus()
        self.sm = StateMachine(float(cfg.get_path("search.reacquire_window_s", 5.0)))
        self.sm.listeners.append(self._on_state_change)
        self.framebuf = FrameBuffer(int(cfg.get_path("vision.history_frames", 24)))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.events: deque = deque(maxlen=200)
        self.alerts: dict[str, str] = {}
        self.started = False

        # calibration (persisted)
        self.calib_path = Path(cfg.get_path("calibration.file", "calibration.json"))
        if not self.calib_path.is_absolute():
            self.calib_path = PACKAGE_ROOT / self.calib_path
        self.calib = self._load_calibration()

        # gimbal
        gcfg = dict(cfg.get("gimbal", {}))
        gcfg.setdefault("command_mode", cfg.get_path("control.mode", "angle"))
        if "zero_pan_deg" in self.calib:
            gcfg["zero_pan_deg"], gcfg["zero_tilt_deg"] = self.calib["zero_pan_deg"], self.calib["zero_tilt_deg"]
        self.gimbal = create_gimbal(gcfg)

        # sensors (FR-10): named camera configs, one active at a time
        self.sensors: dict[str, dict] = dict(cfg.get("sensors", {}))
        self.active_sensor = str(cfg.get_path("camera.active", next(iter(self.sensors), "vis")))
        self.camera = None
        self.world = None
        self.intr: Optional[Intrinsics] = None

        # lidar
        lcfg = dict(cfg.get("lidar", {}))
        if "range_offset_m" in self.calib:
            lcfg["range_offset_m"] = self.calib["range_offset_m"]
        self._lidar_cfg = lcfg
        self.lidar = None

        # perception
        self.detector = None
        self.detector_error = ""
        self.macro: Optional[MacroWorker] = None
        self.assoc = Associator(cfg.get("association", {}))
        self.selector = TargetSelector(cfg.get("seek", {}))
        self.micro_cfg = dict(cfg.get("micro", {}))
        self.micro = create_micro_tracker(self.micro_cfg)
        self.kf = TargetKF(cfg.get("kalman", {}))
        self.kf_lock = threading.Lock()
        self.controller = TrackingController(cfg.get("control", {}), self.gimbal.max_rate)
        self.control_source = str(cfg.get_path("control.source", "internal"))
        self.logger = SessionLogger(cfg.get("logging", {}))
        self.framebuf.listeners.append(self.logger.log_frame)   # every captured frame, independent of vision
        self.ocr = OcrWorker(cfg.get("ocr", {}), self._on_ocr)
        self.ocr_last: dict = {}

        # vision state (written only by the vision thread)
        self.lock: Optional[Lock] = None
        self._micro_t = 0.0
        self._micro_frame_id = -1
        self._micro_hist: OrderedDict[int, Optional[tuple]] = OrderedDict()
        self._last_tracks: list = []
        self._last_macro = None
        self._pending_designation = None
        self._pending_clear = False
        self._ext_targets: deque = deque(maxlen=64)    # external tracker measurements (normalised coords)
        self._last_vision_frame_id = -1
        self.overlay: dict = {}
        self._last_meas: Optional[TargetMeasurement] = None
        self._true_err = (math.nan, math.nan, None)
        self._macro_count = 0

        # control state
        self.telemetry: dict = {}
        self._jog = (0.0, 0.0, 0.0)
        self._jog_sp: Optional[tuple] = None
        self._hold_sent_t = 0.0
        self._homed_after_lost = False
        self._ext_cmd = None
        self._kf_t_seen = None
        self._entered_search_t = 0.0

        # metrics
        self.vision_rate = clock.RateMeter()
        self.control_rate = clock.RateMeter()
        self.vision_ms = 0.0
        self.vision_skipped = 0
        self.lat_offset_ms: deque = deque(maxlen=2000)     # capture -> /target/offset publish (NFR-04)
        self.lat_cmd_ms: deque = deque(maxlen=2000)        # capture -> gimbal command (ES-05.1)
        self._err_window: deque = deque(maxlen=400)        # (t, err_az_deg, err_el_deg)
        self.stats = {"acquisitions": 0, "losses": 0, "reacquired": 0, "reacquire_times_s": [], "lost_alerts": 0,
                      "frames_tracking": 0, "frames_in_fov": 0, "frames_truth": 0, "max_err_deg": 0.0}
        self._t_lost_event: Optional[float] = None
        self.loop = None
        self.gateway = None
        self.ros2 = None
        self.selftest_result: dict = {}
        self.calib_result: dict = {}

    # ================================================================== lifecycle
    def start(self) -> None:
        clock.enable_high_res_timer()
        self._event("system", f"AimBot {__version__} starting ({self.cfg.get('_source', '')})")
        try:
            self.gimbal.start()
        except Exception as exc:
            msg = f"gimbal '{self.gimbal.kind}' failed to start: {exc}"
            log.error(msg)
            self.alerts["gimbal"] = msg
            self._event("fault", msg)
            fb = FixedGimbal(dict(self.cfg.get("gimbal", {})))
            fb.start()
            self.gimbal = fb
        self._start_camera(self.active_sensor)
        self._start_lidar()
        self._create_detector()
        self.macro = MacroWorker(self.detector, self.framebuf, float(self.cfg.get_path("detector.max_hz", 30)),
                                 roi_provider=self._roi_for_macro)
        if self.cfg.get_path("detector.enabled", True):
            self.macro.start()
        self._spawn(self._vision_loop, "vision")
        self._spawn(self._control_loop, "control")
        if self.cfg.get_path("gateway.enabled", False):
            from .operator.gateway import GatewayServer

            try:
                self.gateway = GatewayServer(self, self.cfg.get("gateway", {}))
                self.gateway.start()
            except Exception as exc:
                self.alerts["gateway"] = f"operator gateway failed: {exc}"
        if self.cfg.get_path("ros2.enabled", False):
            from .ros2_bridge import start_ros2_bridge

            self.ros2 = start_ros2_bridge(self, self.cfg.get("ros2", {}))
            if self.ros2 is None:
                self.alerts["ros2"] = "ROS 2 bridge requested but rclpy is not available"
        self.started = True
        if self.cfg.get_path("logging.record_on_start", False):
            self.command({"cmd": "record", "on": True})

    def shutdown(self) -> None:
        if not self.started:
            return
        self._event("system", "shutting down")
        if self.sm.state in ACTIVE:
            self.sm.fire("stop", "shutdown")
        try:
            if self.cfg.get_path("safety.home_on_shutdown", True) and self.gimbal.kind != "none":
                self.gimbal.home()
                time.sleep(0.2)
        except Exception:
            pass
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        for part in (self.macro, self.camera, self.lidar, self.gimbal, self.gateway, self.ros2, self.ocr):
            try:
                if part is not None:
                    part.stop()
            except Exception:
                log.exception("stopping %s", part)
        if self.logger.active:
            self.logger.stop_session(self.summary())
        self.started = False

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # ------------------------------------------------------------------ components
    def _start_camera(self, name: str) -> bool:
        scfg = dict(self.sensors.get(name, {}))
        if not scfg:
            self.alerts["camera"] = f"no sensor named {name!r} in config"
            return False
        try:
            world, true_angles = None, None
            if str(scfg.get("backend", "")).lower() == "sim":
                if self.world is None:
                    from .sim.world import SimWorld

                    w, h = int(scfg.get("width", 968)), int(scfg.get("height", 732))
                    intr = Intrinsics.from_config(scfg.get("intrinsics", {}), w, h)
                    sim = self.cfg.get("sim", {})
                    self.world = SimWorld(intr, str(sim.get("scenario", "mixed")), int(sim.get("seed", 7)),
                                          float(sim.get("noise_sigma", 3.0)), str(sim.get("card_text", "AIM 23")))
                world, true_angles = self.world, self.gimbal.true_angles
            cam = create_camera(name, scfg, self.framebuf, world, true_angles)
            cam.start()
        except Exception as exc:
            msg = f"camera '{name}' failed: {exc}"
            log.error(msg)
            self.alerts["camera"] = msg
            self._event("fault", msg)
            return False
        self.camera = cam
        self.active_sensor = name
        self.intr = cam.intr
        self.alerts.pop("camera", None)
        self._event("sensor", f"camera '{name}' ({cam.kind}) {cam.intr.width}x{cam.intr.height}, "
                              f"HFOV {cam.intr.hfov_deg:.1f} deg")
        return True

    def _start_lidar(self) -> None:
        lcfg = self._lidar_cfg
        try:
            self.lidar = create_lidar(lcfg, self.world, self.gimbal.true_angles)
            if isinstance(self.lidar, ExternalLidar):
                self._wire_esp32_lidar(self.lidar, lcfg)
            self.lidar.start()
        except Exception as exc:
            msg = f"lidar failed: {exc}"
            log.error(msg)
            self.alerts["lidar"] = msg
            self.lidar = NoLidar(lcfg)

    def _wire_esp32_lidar(self, lidar: ExternalLidar, lcfg: dict) -> None:
        import struct

        from .gimbal.esp32_bridge import E2J_LIDAR, Esp32Link

        link = Esp32Link.get(str(lcfg.get("port", self.cfg.get_path("gimbal.port", "/dev/ttyACM0"))),
                             int(lcfg.get("baud", 921600)))

        def on_lidar(t_rx, seq, payload):
            if len(payload) >= 10:
                t_ms, dist, strength, temp = struct.unpack_from("<IHHH", payload)
                lidar.push(link.esp_to_host(t_ms, t_rx), dist, strength, temp)

        link.on(E2J_LIDAR, on_lidar)
        link.start()

    def _create_detector(self) -> None:
        dcfg = dict(self.cfg.get("detector", {}))
        try:
            self.detector = create_detector(dcfg)
            self.detector.warmup((self.intr.height, self.intr.width, 3) if self.intr else (480, 640, 3))
            self.detector_error = ""
            note = getattr(self.detector, "fallback_used", "")
            if note:
                self.alerts["detector"] = note
        except Exception as exc:
            self.detector_error = f"{dcfg.get('backend')} detector unavailable ({exc}); using colour detector"
            log.error(self.detector_error)
            self.alerts["detector"] = self.detector_error
            self.detector = ColorBlobDetector(dcfg.get("color", {}))

    # ================================================================== operator API
    def command(self, c: dict) -> dict:
        """Single entry point for GUI, gateway and ROS 2 commands. Returns {'ok': bool, 'msg': str}."""
        cmd = str(c.get("cmd", ""))
        ok, msg = True, ""
        if cmd not in ("jog",):
            self._event("operator", f"{cmd} {json.dumps({k: v for k, v in c.items() if k != 'cmd'})}")
        if cmd == "start":
            ok = self.sm.fire("start", "operator start") or self.sm.state == SEARCHING
        elif cmd == "stop":
            ok = self.sm.fire("stop", "operator stop")
            self._pending_clear = True
        elif cmd == "pause":
            ok = self.sm.fire("pause", "operator pause")
        elif cmd == "resume":
            ok = self.sm.fire("resume", "operator resume")
        elif cmd == "estop":
            self.sm.fire("estop", c.get("reason", "operator E-STOP"))
        elif cmd == "reset":
            ok = self.sm.fire("reset", "operator reset")
        elif cmd == "designate":
            self._pending_designation = (float(c["x"]), float(c["y"]), c.get("w"), c.get("h"))
        elif cmd == "clear_target":
            self._pending_clear = True
        elif cmd == "set_target_class":
            self.selector.target_class = str(c.get("cls", "auto"))
            msg = f"seeking {self.selector.target_class}"
        elif cmd == "calibrate":
            if self.sm.state != IDLE:
                ok, msg = False, "stop tracking before calibrating"
            else:
                threading.Thread(target=self._calibrate, args=(float(c.get("distance_m", 3.048)),), daemon=True).start()
        elif cmd == "set_boresight":
            self._set_boresight(float(c["x"]), float(c["y"]))
        elif cmd == "reset_boresight":
            self.calib.pop("boresight_px", None)
            self._save_calibration()
        elif cmd == "selftest":
            threading.Thread(target=self._selftest, daemon=True).start()
        elif cmd == "ocr":
            ok, msg = self._request_ocr()
        elif cmd == "jog":
            self._jog = (R(float(c.get("pan", 0.0))), R(float(c.get("tilt", 0.0))), clock.now())
        elif cmd == "home":
            if self.sm.state in (IDLE, PAUSED, LOST, SEARCHING):
                self._jog_sp = None
                self.gimbal.home()
            else:
                ok, msg = False, "home is only allowed when not tracking"
        elif cmd == "record":
            on = bool(c.get("on", not self.logger.active))
            if on and not self.logger.active:
                self._start_logging("manual")
            elif not on and self.logger.active:
                self.logger.stop_session(self.summary())
        elif cmd == "switch_sensor":
            ok, msg = self._switch_sensor(str(c.get("name", "")))
        elif cmd == "set_scenario":
            if self.world is None:
                ok, msg = False, "not running the simulator"
            else:
                self.world.set_scenario(str(c.get("name", "mixed")))
                self._pending_clear = True
                self.assoc.reset()
                msg = f"scenario {self.world.scenario}"
        elif cmd == "external_target":   # ES-16.1: sponsor/external tracker reports the target position
            self._ext_targets.append((int(c.get("frame_id", -1)), float(c["x"]), float(c["y"]),
                                      float(c.get("w", 0.03)), float(c.get("h", 0.03)), str(c.get("cls", "external")),
                                      clock.now()))
        elif cmd == "external_rate":   # from an external (ECE) controller via ROS 2
            self._ext_cmd = (R(float(c.get("pan_dps", 0.0))), R(float(c.get("tilt_dps", 0.0))), clock.now())
        elif cmd == "set_detector_enabled":
            if self.macro:
                self.macro.enabled = bool(c.get("on", True))
        else:
            ok, msg = False, f"unknown command {cmd!r}"
        self.bus.publish(Topics.OPERATOR_CMD, dict(c))
        return {"ok": bool(ok), "msg": msg}

    # ================================================================== vision loop
    def _vision_loop(self) -> None:
        last_id = -1
        while not self._stop.is_set():
            frame = self.framebuf.wait_newer(last_id, 0.5)
            if frame is None:
                continue
            if last_id >= 0 and frame.frame_id > last_id + 1:
                self.vision_skipped += frame.frame_id - last_id - 1
            last_id = frame.frame_id
            t0 = clock.now()
            try:
                self.intr = self.camera.intr if self.camera is not None and self.camera.intr is not None else self.intr
                if self.intr is None or self.intr.width != frame.width or self.intr.height != frame.height:
                    base = self.intr or Intrinsics.from_hfov(frame.width, frame.height, 60.0)
                    self.intr = base.scaled_to(frame.width, frame.height)
                if self._pending_clear:
                    self._pending_clear = False
                    self._drop_lock("cleared by operator", lost=False)
                if self._ext_targets:
                    self._handle_external(frame)
                if self._pending_designation is not None:
                    req, self._pending_designation = self._pending_designation, None
                    self._handle_designation(frame, req)
                self._drain_macro(frame)
                self._step_micro(frame)
                self._sim_truth_metrics(frame)
                self._build_overlay(frame)
            except Exception:
                log.exception("vision loop error")
            self._last_vision_frame_id = frame.frame_id
            self.vision_ms = (clock.now() - t0) * 1000.0
            self.vision_rate.tick()

    # ---- geometry helpers
    def _aim_px(self) -> tuple[float, float]:
        bs = self.calib.get("boresight_px")
        intr = self.intr
        if bs and intr is not None:
            return float(bs[0]) * intr.width, float(bs[1]) * intr.height
        return (intr.cx, intr.cy) if intr is not None else (0.0, 0.0)

    def _aim_offset(self) -> tuple[float, float]:
        """Direction of the aim pixel relative to the gimbal boresight (rad)."""
        if self.intr is None:
            return 0.0, 0.0
        u, v = self._aim_px()
        return pixel_to_azel(self.intr, u, v, 0.0, 0.0)

    def _ego_shift(self, t_a: float, t_b: float) -> tuple[float, float]:
        if t_a <= 0 or self.intr is None:
            return 0.0, 0.0
        pa, ta = self.gimbal.angles_at(t_a)
        pb, tb = self.gimbal.angles_at(t_b)
        return -self.intr.fx * wrap_pi(pb - pa) * math.cos(tb), self.intr.fy * (tb - ta)

    def _measure(self, frame: Frame, bbox, score: float) -> TargetMeasurement:
        intr = self.intr
        u, v = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        pan, tilt = self.gimbal.angles_at(frame.t_capture)
        az, el = pixel_to_azel(intr, u, v, pan, tilt)
        ao = self._aim_offset()
        err_az = wrap_pi(az - (pan + ao[0]))
        err_el = el - (tilt + ao[1])
        bu, bv = self._aim_px()
        dx = max(-1.0, min(1.0, (u - bu) / (intr.width / 2.0)))
        dy = max(-1.0, min(1.0, -(v - bv) / (intr.height / 2.0)))
        return TargetMeasurement(frame.t_capture, frame.frame_id, u, v, tuple(bbox), az, el, err_az, err_el, dx, dy, score)

    # ---- macro results
    def _drain_macro(self, frame: Frame) -> None:
        if self.macro is None:
            return
        while True:
            try:
                res = self.macro.out.get_nowait()
            except Exception:
                break
            self._macro_count += 1
            # ego-motion between the previous detection and this one for the associator
            prev_t = self._last_macro.t_capture if self._last_macro is not None else 0.0
            tracks = self.assoc.update(res.detections, res.t_capture, res.frame_id, self._ego_shift(prev_t, res.t_capture))
            self._last_macro = res
            self._last_tracks = tracks
            self.logger.log_detections(res)
            self.bus.publish(Topics.DETECTIONS, {"frame_id": res.frame_id, "t_capture": res.t_capture,
                                                 "infer_ms": res.infer_ms,
                                                 "detections": [{"track_id": d.track_id, "cls": d.cls, "conf": d.conf,
                                                                 "bbox": d.bbox} for d in res.detections]})
            st = self.sm.state
            if self.lock is None and st in (SEARCHING, LOST):
                self._try_acquire(res, tracks, frame)
            elif self.lock is not None and self.lock.mode == "classified":
                self._confirm(res, tracks, frame)

    def _try_acquire(self, res, tracks, frame: Frame) -> None:
        near, gate = None, 1e9
        rem = self.sm.reacquire_remaining()
        if rem is not None and self.kf.initialized and self.intr is not None:
            elapsed = self.sm.reacquire_window - rem
            if elapsed < float(self.cfg.get_path("search.near_gate_s", 1.5)):
                with self.kf_lock:
                    est = self.kf.predict_to(res.t_capture)
                if est is not None:
                    pan, tilt = self.gimbal.angles_at(res.t_capture)
                    p = world_to_pixel(self.intr, azel_to_dir(est.az, est.el), pan, tilt)
                    if p is not None:
                        near, gate = (p[0], p[1]), max(60.0, 0.3 * self.intr.width)
        tr = self.selector.select(tracks, (self.intr.width, self.intr.height), self._aim_px(), near, gate)
        if tr is None:
            return
        src = self.framebuf.get(res.frame_id) or frame
        if self._init_micro(src, tr.bbox, frame, keep_velocity=False):
            self.lock = Lock("classified", tr.cls, tr.tid, tr.conf, self.micro.bbox, clock.now(), res.t_capture,
                             self._micro_frame_id, truth_id=self._truth_id_for(src, tr.bbox),
                             t_good=res.t_capture, t_reseed=res.t_capture)
            self._on_new_lock(f"{tr.cls} #{tr.tid} (conf {tr.conf:.2f})")

    def _init_micro(self, src: Frame, bbox, current: Frame, keep_velocity: bool) -> bool:
        """Initialise the micro tracker on the detection's own frame, then catch up to the current frame."""
        try:
            ok = self.micro.init(src.image, bbox, src.t_capture, keep_velocity=keep_velocity)
        except TypeError:
            ok = self.micro.init(src.image, bbox, src.t_capture)
        if not ok:
            return False
        t_prev = src.t_capture
        last_id = src.frame_id
        self._micro_hist[src.frame_id] = tuple(bbox)
        for f in self.framebuf.frames_between(src.frame_id, current.frame_id - 1):
            ok, bb, _ = self.micro.update(f.image, f.t_capture, self._ego_shift(t_prev, f.t_capture))
            self._micro_hist[f.frame_id] = bb if ok else None
            t_prev = f.t_capture
            last_id = f.frame_id
            if not ok:
                return False
        self._micro_t = t_prev
        self._micro_frame_id = last_id   # frame the micro tracker state now corresponds to
        return True

    def _confirm(self, res, tracks, frame: Frame) -> None:
        lk = self.lock
        mb = self._micro_hist.get(res.frame_id)
        same = [tr for tr in tracks if tr.cls == lk.cls and tr.frame_id == res.frame_id]
        best, best_iou = None, 0.0
        if mb is not None:
            for tr in same:
                ov = iou(tr.bbox, mb) + (0.1 if tr.tid == lk.track_id else 0.0)
                if ov > best_iou:
                    best, best_iou = tr, ov
        if best is None or best_iou < 0.2:
            # The micro tracker and the detector disagree. Trust the detector if it sees a target
            # of the locked class close to where the Kalman filter expects it (drift recovery).
            if not same or self.intr is None:
                return
            ref = None
            with self.kf_lock:
                est = self.kf.predict_to(res.t_capture) if self.kf.initialized else None
            if est is not None:
                pan, tilt = self.gimbal.angles_at(res.t_capture)
                p = world_to_pixel(self.intr, azel_to_dir(est.az, est.el), pan, tilt)
                ref = (p[0], p[1]) if p is not None else None
            if ref is None and mb is not None:
                ref = (0.5 * (mb[0] + mb[2]), 0.5 * (mb[1] + mb[3]))
            if ref is None:
                return
            size = max(lk.bbox[2] - lk.bbox[0], lk.bbox[3] - lk.bbox[1])
            gate = max(60.0, float(self.cfg.get_path("micro.recover_gate", 3.0)) * size)
            cand = min(same, key=lambda tr: math.hypot(tr.center[0] - ref[0], tr.center[1] - ref[1]))
            if math.hypot(cand.center[0] - ref[0], cand.center[1] - ref[1]) > gate:
                return
            src = self.framebuf.get(res.frame_id)
            before_id, before = self._micro_frame_id, self.micro.bbox
            if src is not None and self._init_micro(src, cand.bbox, frame, keep_velocity=False):
                lk.bbox, lk.frame_id = self.micro.bbox, self._micro_frame_id
                lk.t_confirm, lk.track_id, lk.conf = res.t_capture, cand.tid, cand.conf
                lk.misses, lk.t_reseed = 0, res.t_capture
                # keep the Kalman prediction (it anchors the tracker's search) - just move it onto the detection
                if not self._shift_kf(before_id, before, lk.bbox):
                    with self.kf_lock:
                        self.kf.reset()
                self.stats["drift_recoveries"] = self.stats.get("drift_recoveries", 0) + 1
            return
        lk.t_confirm, lk.track_id, lk.conf = res.t_capture, best.tid, best.conf
        # Anchor the micro tracker to the detector: if it has slid off the detection (template drift on
        # thin targets / changing background) or the scale changed, re-seed it from the detection frame
        # and fast-forward. The Kalman state is shifted by the same correction (velocity is kept).
        bw, bh = best.size
        mw, mh = mb[2] - mb[0], mb[3] - mb[1]
        size_ratio = (bw * bh) / max(1.0, mw * mh)
        off = math.hypot(best.center[0] - 0.5 * (mb[0] + mb[2]), best.center[1] - 0.5 * (mb[1] + mb[3]))
        tol = max(3.0, float(self.cfg.get_path("micro.reseed_offset_frac", 0.08)) * max(bw, bh))
        stale = lk.score < float(self.cfg.get_path("micro.refresh_score", 0.65))   # template no longer fits
        if off > tol or stale or not (0.75 < size_ratio < 1.33):
            src = self.framebuf.get(res.frame_id)
            before_id = self._micro_frame_id
            before = self.micro.bbox
            if src is not None and self._init_micro(src, best.bbox, frame, keep_velocity=True):
                lk.bbox = self.micro.bbox
                lk.frame_id = self._micro_frame_id
                lk.misses, lk.t_reseed = 0, res.t_capture
                self.stats["reseeds"] = self.stats.get("reseeds", 0) + 1
                self._shift_kf(before_id, before, lk.bbox)

    def _shift_kf(self, before_id: int, before, after) -> bool:
        """Move the Kalman position by the micro tracker's re-anchoring correction (same frame)."""
        f_ref = self.framebuf.get(self._micro_frame_id)
        if f_ref is None or before_id != self._micro_frame_id or self.intr is None or self.kf.updates < 5:
            return False
        pan, tilt = self.gimbal.angles_at(f_ref.t_capture)
        a0 = pixel_to_azel(self.intr, 0.5 * (before[0] + before[2]), 0.5 * (before[1] + before[3]), pan, tilt)
        a1 = pixel_to_azel(self.intr, 0.5 * (after[0] + after[2]), 0.5 * (after[1] + after[3]), pan, tilt)
        with self.kf_lock:
            self.kf.shift(wrap_pi(a1[0] - a0[0]), a1[1] - a0[1])
        return True

    # ---- designation (FR-09)
    def _handle_designation(self, frame: Frame, req) -> None:
        x, y, w, h = req
        W, H = frame.width, frame.height
        px, py = x * W, y * H
        bbox, mode, cls, tid, conf = None, "manual", "manual", -1, 1.0
        if w is None or h is None or float(w) * W < 6 or float(h) * H < 6:
            # click: inside a current detection -> lock that (classified) target
            best = None
            for tr in self._last_tracks:
                x1, y1, x2, y2 = tr.bbox
                if x1 <= px <= x2 and y1 <= py <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best[0]:
                        best = (area, tr)
            if best is not None:
                tr = best[1]
                dt = frame.t_capture - tr.t
                sx, sy = self._ego_shift(tr.t, frame.t_capture)
                ox, oy = tr.vel[0] * dt + sx, tr.vel[1] * dt + sy
                bbox = (tr.bbox[0] + ox, tr.bbox[1] + oy, tr.bbox[2] + ox, tr.bbox[3] + oy)
                mode, cls, tid, conf = "classified", tr.cls, tr.tid, tr.conf
            else:
                s = float(self.cfg.get_path("manual.box_frac", 0.07)) * W
                bbox = (px - s / 2, py - s / 2, px + s / 2, py + s / 2)
        else:
            bbox = (px, py, px + float(w) * W, py + float(h) * H)
        if self.sm.state in (IDLE, LOST, PAUSED):
            self.sm.fire("start", "target designated by operator")
        if self.sm.state not in (SEARCHING, TRACKING):
            self._event("operator", f"designation ignored in state {self.sm.state}")
            return
        if self.lock is not None:
            self.lock = None
        ok = self.micro.init(frame.image, bbox, frame.t_capture)
        if not ok:
            self._event("operator", "designated region too small, outside the frame or without texture to track")
            if self.sm.state == SEARCHING and self.lock is None:
                self.alerts["designation"] = "region not trackable - click on an object or drag a box around it"
            return
        self.alerts.pop("designation", None)
        self._micro_t = frame.t_capture
        self._micro_hist[frame.frame_id] = tuple(self.micro.bbox)
        self.lock = Lock(mode, cls, tid, conf, self.micro.bbox, clock.now(), frame.t_capture, frame.frame_id,
                         truth_id=self._truth_id_for(frame, bbox), t_good=frame.t_capture, t_reseed=frame.t_capture)
        self._on_new_lock(f"operator designated {'%s #%d' % (cls, tid) if mode == 'classified' else 'region'}")

    def _on_new_lock(self, desc: str) -> None:
        with self.kf_lock:
            self.kf.reset()
        self.controller.reset()
        self.stats["acquisitions"] += 1
        if self._t_lost_event is not None:
            dt = clock.now() - self._t_lost_event
            self.stats["reacquired"] += 1
            self.stats["reacquire_times_s"].append(round(dt, 3))
            self._t_lost_event = None
        if self.sm.state == TRACKING:   # re-designation while already tracking
            self._event("track", f"new lock: {desc}")
        else:
            self.sm.fire("acquired", desc)

    def _drop_lock(self, reason: str, lost: bool = True) -> None:
        if self.lock is None:
            return
        self.lock = None
        if lost:
            self.stats["losses"] += 1
            self._t_lost_event = clock.now()
            if self.sm.state == TRACKING:
                self.sm.fire("lost", reason)
        else:
            self._event("track", f"lock released: {reason}")
            if self.sm.state == TRACKING:
                self.sm.fire("release", reason)
            with self.kf_lock:
                self.kf.reset()

    # ---- micro tracking + measurement (every frame)
    def _handle_external(self, frame: Frame) -> None:
        """Measurements from an external tracking program (ES-16.1) replace the micro tracker.
        Positions are normalised to the frame the program saw (gateway video is downscaled)."""
        while self._ext_targets:
            fid, x, y, w, h, cls, t_rx = self._ext_targets.popleft()
            src = self.framebuf.get(fid) if fid >= 0 else None
            src = src or frame
            W, H = src.width, src.height
            bbox = ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
            if self.lock is None or self.lock.mode != "external":
                if self.sm.state in (IDLE, LOST, PAUSED):
                    self.sm.fire("start", "external tracker")
                if self.sm.state not in (SEARCHING, TRACKING):
                    return
                self.lock = Lock("external", cls, -1, 1.0, bbox, clock.now(), src.t_capture, src.frame_id,
                                 t_good=src.t_capture, t_reseed=src.t_capture)
                self._on_new_lock(f"external tracker ({cls})")
            lk = self.lock
            if self._publish_measurement(src, bbox, 1.0):
                lk.bbox, lk.t_good, lk.frame_id, lk.misses = bbox, src.t_capture, src.frame_id, 0

    def _step_micro(self, frame: Frame) -> None:
        lk = self.lock
        if lk is None:
            return
        if lk.mode == "external":
            timeout = float(self.cfg.get_path("external.timeout_s", 0.5))
            if frame.t_capture - lk.t_good > timeout:
                self._drop_lock(f"external tracker silent for {timeout:.1f} s", lost=True)
            return
        if lk.frame_id >= frame.frame_id:   # already processed (designation on this frame)
            self._publish_measurement(frame, lk.bbox, lk.score)
            return
        shift = self._ego_shift(self._micro_t, frame.t_capture)
        prior = self._kf_pixel(frame.t_capture) if self.kf.updates >= 3 else None
        ok, bbox, score = self.micro.update(frame.image, frame.t_capture, shift, prior)
        self._micro_t = frame.t_capture
        self._micro_frame_id = frame.frame_id
        lk.frame_id = frame.frame_id
        accepted = False
        if ok:
            accepted = self._publish_measurement(frame, bbox, score)
        self._micro_hist[frame.frame_id] = tuple(bbox) if accepted else None
        while len(self._micro_hist) > 96:
            self._micro_hist.popitem(last=False)
        if accepted:
            lk.bbox, lk.misses, lk.score = tuple(bbox), 0, score
            lk.t_good = frame.t_capture
        else:
            # tracker failed, or its answer jumped away from the Kalman prediction (outlier)
            lk.misses += 1
            lk.score = score
        reason = None
        # time-based so it behaves the same at 30, 60 or 130 fps; a detector re-seed resets it
        max_gap = float(self.cfg.get_path("micro.max_miss_s", 0.12))
        confirm_to = float(self.cfg.get_path("micro.confirm_timeout_s", 1.0))
        if lk.misses > 0 and frame.t_capture - max(lk.t_good, lk.t_reseed) > max_gap:
            reason = f"micro tracker lost target ({lk.misses} frames, score {score:.2f})"
        elif lk.mode == "classified" and confirm_to > 0 and self.macro is not None and self.macro.enabled \
                and frame.t_capture - lk.t_confirm > confirm_to:
            reason = f"detector has not confirmed the target for {confirm_to:.1f} s"
        if reason:
            self._drop_lock(reason, lost=True)

    def _kf_pixel(self, t: float):
        """Kalman-predicted target position in the image at time t (or None)."""
        if self.intr is None:
            return None
        with self.kf_lock:
            est = self.kf.predict_to(t) if self.kf.initialized else None
        if est is None or not est.valid:
            return None
        pan, tilt = self.gimbal.angles_at(t)
        p = world_to_pixel(self.intr, azel_to_dir(est.az, est.el), pan, tilt)
        return (p[0], p[1]) if p is not None else None

    def _publish_measurement(self, frame: Frame, bbox, score: float) -> bool:
        meas = self._measure(frame, bbox, score)
        sigma_px = float(self.cfg.get_path("kalman.meas_sigma_px", 3.0))
        r = (sigma_px / self.intr.fx) ** 2
        with self.kf_lock:
            accepted = self.kf.update(meas.t_capture, meas.az, meas.el, r)
        if not accepted:
            self.stats["kf_rejected"] = self.stats.get("kf_rejected", 0) + 1
            return False
        self._last_meas = meas
        self._err_window.append((meas.t_capture, D(meas.err_az), D(meas.err_el)))
        t_pub = clock.now()
        self.bus.publish(Topics.TARGET_OFFSET, {
            "t_capture": meas.t_capture, "frame_id": meas.frame_id, "dx": meas.dx, "dy": meas.dy,
            "err_az_rad": meas.err_az, "err_el_rad": meas.err_el, "az_rad": meas.az, "el_rad": meas.el,
            "u": meas.u, "v": meas.v, "score": score, "cls": self.lock.cls if self.lock else "",
            "latency_ms": (t_pub - frame.t_capture) * 1000.0})
        self.lat_offset_ms.append((t_pub - frame.t_capture) * 1000.0)
        return True

    # ---- sim ground truth (only in simulation)
    def _truth_id_for(self, frame: Frame, bbox) -> Optional[int]:
        truth = frame.meta.get("truth")
        if not truth:
            return None
        cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        best = None
        for tr in truth:
            x1, y1, x2, y2 = tr["full_bbox"]
            d = math.hypot(cx - tr["center"][0], cy - tr["center"][1])
            if x1 - 5 <= cx <= x2 + 5 and y1 - 5 <= cy <= y2 + 5 and (best is None or d < best[0]):
                best = (d, tr["id"])
        return best[1] if best else None

    def _sim_truth_metrics(self, frame: Frame) -> None:
        truth = frame.meta.get("truth")
        if truth is None or self.world is None:
            return
        lk = self.lock
        tid = lk.truth_id if lk is not None else getattr(self, "_last_truth_id", None)
        if lk is not None:
            self._last_truth_id = lk.truth_id
        st = self.sm.state
        if tid is None or st not in (TRACKING, SEARCHING, LOST):
            self._true_err = (math.nan, math.nan, None)
            return
        # true error between target and the true aim direction at exposure time
        tp, tt = frame.meta.get("true_pan", 0.0), frame.meta.get("true_tilt", 0.0)
        target = None
        for tr in self.world.target_truth(frame.t_capture):
            if tr["id"] == tid:
                target = tr
        if target is None:     # e.g. the ball has been caught: target no longer exists
            self._true_err = (math.nan, math.nan, None)
            return
        p = np.asarray(target["pos"], float)
        az, el = math.atan2(p[1], p[0]), math.atan2(p[2], math.hypot(p[0], p[1]))
        ao = self._aim_offset()
        # true pointing = raw (physical) gimbal angles used to render the frame
        e_az = D(wrap_pi(az - (tp + ao[0])))
        e_el = D(el - (tt + ao[1]))
        proj = world_to_pixel(self.intr, p, tp, tt)
        in_fov = proj is not None and 0 <= proj[0] < frame.width and 0 <= proj[1] < frame.height
        self._true_err = (e_az, e_el, in_fov)
        if st == TRACKING:
            self.stats["frames_truth"] += 1
            self.stats["frames_in_fov"] += int(in_fov)

    def _build_overlay(self, frame: Frame) -> None:
        W, H = float(frame.width), float(frame.height)

        def nb(b):
            return (b[0] / W, b[1] / H, b[2] / W, b[3] / H)

        dets = []
        # every target the associator currently knows (not just the last - possibly ROI-only - detection)
        for tr in list(self.assoc.tracks.values()):
            if frame.t_capture - tr.t < 0.4 and tr.hits >= 2:
                dets.append({"bbox": nb(tr.bbox), "cls": tr.cls, "conf": round(tr.conf, 2), "tid": tr.tid})
        lk = self.lock
        lock = None
        if lk is not None:
            lock = {"bbox": nb(lk.bbox), "cls": lk.cls, "tid": lk.track_id, "mode": lk.mode, "score": round(lk.score, 2),
                    "misses": lk.misses}
        pred = None
        with self.kf_lock:
            est = self.kf.predict_to(frame.t_capture) if self.kf.initialized else None
        if est is not None and est.valid and self.intr is not None:
            pan, tilt = self.gimbal.angles_at(frame.t_capture)
            p = world_to_pixel(self.intr, azel_to_dir(est.az, est.el), pan, tilt)
            if p is not None:
                pred = (p[0] / W, p[1] / H)
        au, av = self._aim_px()
        beam = math.radians(float(self.cfg.get_path("lidar.beam_deg", 3.0)) / 2.0)
        self.overlay = {"frame_id": frame.frame_id, "w": frame.width, "h": frame.height, "sensor": frame.sensor,
                        "detections": dets, "lock": lock, "pred": pred, "aim": (au / W, av / H),
                        "beam_r": (math.tan(beam) * (self.intr.fx if self.intr else 1000.0)) / W,
                        "roi": nb(self._last_macro.roi) if (self._last_macro and self._last_macro.roi) else None}

    def _roi_for_macro(self, frame: Frame):
        roi = self._macro_roi(frame)
        lk = self.lock
        # with several YOLO models (trained drone model + COCO assist) only the one that can see the
        # locked class runs on the ROI
        self.detector.focus = lk.cls if (roi is not None and lk is not None and lk.mode == "classified") else None
        return roi

    def _macro_roi(self, frame: Frame):
        if not self.cfg.get_path("detector.roi_when_tracking", True):
            return None
        lk = self.lock
        if lk is None or self.sm.state != TRACKING:
            return None
        every = int(self.cfg.get_path("detector.full_frame_every", 5))
        if every > 0 and self._macro_count % every == 0:
            return None
        x1, y1, x2, y2 = lk.bbox
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        s = max(float(self.cfg.get_path("detector.roi_min_px", 320)), 3.0 * max(x2 - x1, y2 - y1))
        if s >= 0.8 * min(frame.width, frame.height):
            return None
        return (max(0.0, cx - s / 2), max(0.0, cy - s / 2), min(frame.width, cx + s / 2), min(frame.height, cy + s / 2))

    # ================================================================== control loop (100 Hz)
    def _control_loop(self) -> None:
        hz = float(self.cfg.get_path("control.rate_hz", 100.0))
        self.loop = loop = clock.RateLoop(hz, self._stop)
        timeout = float(self.cfg.get_path("safety.gimbal_timeout_s", 0.5))
        coast = float(self.cfg.get_path("search.coast_s", 0.4))
        coast_max = float(self.cfg.get_path("search.coast_max_deg", 6.0))
        while not self._stop.is_set():
            t = loop.wait()
            try:
                self.sm.tick()
                st = self.sm.state
                g = self.gimbal.state()
                cmd = None
                if st in ACTIVE and self.gimbal.feedback_age() > timeout and self.gimbal.kind != "none":
                    self.sm.fire("fault", f"gimbal feedback lost for > {timeout:.1f} s (ES-11.3)")
                    self.gimbal.hold()
                    st = self.sm.state
                if st == TRACKING:
                    if self.control_source == "external":
                        ec = self._ext_cmd
                        if ec is not None and t - ec[2] < 0.2:
                            cmd = GimbalCommand("rate", ec[0], ec[1])
                            self.gimbal.command(cmd)
                        else:
                            self._hold(t)
                    else:
                        with self.kf_lock:
                            cmd = self.controller.compute(t, self.kf.predict_to, g, self._aim_offset())
                        if cmd is not None:
                            self.gimbal.command(cmd)
                        else:
                            self._hold(t)
                elif st == SEARCHING:
                    lost_at = self.sm.lost_at
                    cmd = None
                    if lost_at is not None and t - lost_at < coast and self.kf.initialized and self.control_source != "external":
                        # brief occlusion: keep following the prediction, but never chase a target that has
                        # vanished (caught ball) further than coast_max_deg from where it was last seen
                        with self.kf_lock:
                            est_now = self.kf.predict_to(t)
                            last = self.kf.predict_to(self.kf.t) if self.kf.t is not None else None
                        if est_now is not None and last is not None and \
                                D(math.hypot(wrap_pi(est_now.az - last.az), est_now.el - last.el)) < coast_max:
                            with self.kf_lock:
                                cmd = self.controller.compute(t, self.kf.predict_to, g, self._aim_offset())
                    if cmd is not None:
                        self.gimbal.command(cmd)
                    else:
                        self._hold(t)
                elif st == LOST:
                    if self.cfg.get_path("search.on_lost", "hold") == "home" and not self._homed_after_lost:
                        self.gimbal.home()
                        self._homed_after_lost = True
                        cmd = GimbalCommand("angle", 0.0, 0.0)
                    elif not self._homed_after_lost:
                        self._hold(t)
                elif st in (IDLE, PAUSED):
                    cmd = self._apply_jog(t, g)
                # frame -> command latency (age of the newest measurement used)
                if cmd is not None and self.kf.t is not None and st == TRACKING:
                    if self._kf_t_seen != self.kf.t:
                        self._kf_t_seen = self.kf.t
                        self.lat_cmd_ms.append((t - self.kf.t) * 1000.0)
                if self.ros2 is None and self.gimbal.kind == "esp32":
                    self.gimbal.set_jetson_state(st)
                self._publish_heartbeat(t, st, g, cmd, loop)
            except Exception:
                log.exception("control loop error")
            self.control_rate.tick()

    def _hold(self, t: float) -> None:
        if t - self._hold_sent_t > 0.5:
            self.gimbal.hold()
            self._hold_sent_t = t

    def _apply_jog(self, t: float, g) -> Optional[GimbalCommand]:
        jp, jt, jts = self._jog
        if t - jts > 0.3 or (abs(jp) < 1e-6 and abs(jt) < 1e-6):
            if self._jog_sp is not None:
                self._jog_sp = None
                self.gimbal.hold()
            return None
        if self.gimbal.cfg.get("command_mode", "angle") == "angle":
            if self._jog_sp is None:
                self._jog_sp = (g.pan, g.tilt)
            dt = self.loop.period if self.loop else 0.01
            self._jog_sp = self.gimbal.clamp_angles(self._jog_sp[0] + jp * dt, self._jog_sp[1] + jt * dt)
            cmd = GimbalCommand("angle", *self._jog_sp)
        else:
            cmd = GimbalCommand("rate", jp, jt)
        self.gimbal.command(cmd)
        return cmd

    def _publish_heartbeat(self, t: float, st: str, g, cmd, loop) -> None:
        lk = self.lock
        meas = self._last_meas if lk is not None else None
        rng = self.lidar.latest() if self.lidar is not None else None
        with self.kf_lock:
            est = self.kf.predict_to(t) if (self.kf.initialized and st in (TRACKING, SEARCHING)) else None
        on_target = False
        xyz = (math.nan, math.nan, math.nan)
        if rng is not None and rng.valid and meas is not None and t - rng.t < 0.2:
            beam = R(float(self.cfg.get_path("lidar.beam_deg", 3.0)) / 2.0)
            if math.hypot(meas.err_az, meas.err_el) < beam:
                on_target = True
                d = azel_to_dir(meas.az, meas.el) * rng.distance_m
                xyz = (float(d[0]), float(d[1]), float(d[2]))
        te = self._true_err
        row = {
            "t_mono": f"{t:.6f}", "t_wall": f"{time.time():.6f}", "state": st,
            "frame_id": meas.frame_id if meas else "", "frame_t_capture": f"{meas.t_capture:.6f}" if meas else "",
            "target_id": lk.track_id if lk else "", "target_class": lk.cls if lk else "",
            "target_conf": f"{lk.conf:.3f}" if lk else "",
            "bbox_x1": f"{lk.bbox[0]:.1f}" if lk else "", "bbox_y1": f"{lk.bbox[1]:.1f}" if lk else "",
            "bbox_x2": f"{lk.bbox[2]:.1f}" if lk else "", "bbox_y2": f"{lk.bbox[3]:.1f}" if lk else "",
            "u": f"{meas.u:.2f}" if meas else "", "v": f"{meas.v:.2f}" if meas else "",
            "dx": f"{meas.dx:.5f}" if meas else "", "dy": f"{meas.dy:.5f}" if meas else "",
            "err_az_deg": f"{D(meas.err_az):.4f}" if meas else "", "err_el_deg": f"{D(meas.err_el):.4f}" if meas else "",
            "kf_az_deg": f"{D(est.az):.4f}" if est else "", "kf_el_deg": f"{D(est.el):.4f}" if est else "",
            "kf_az_rate_dps": f"{D(est.az_rate):.3f}" if est else "", "kf_el_rate_dps": f"{D(est.el_rate):.3f}" if est else "",
            "pan_deg": f"{D(g.pan):.4f}", "tilt_deg": f"{D(g.tilt):.4f}", "pan_rate_dps": f"{D(g.pan_rate):.3f}",
            "tilt_rate_dps": f"{D(g.tilt_rate):.3f}",
            "cmd_mode": cmd.mode if cmd else "", "cmd_pan": f"{D(cmd.pan):.4f}" if cmd else "",
            "cmd_tilt": f"{D(cmd.tilt):.4f}" if cmd else "",
            "range_m": f"{rng.distance_m:.3f}" if rng else "", "range_valid": int(rng.valid) if rng else "",
            "range_on_target": int(on_target), "target_x_m": f"{xyz[0]:.3f}" if on_target else "",
            "target_y_m": f"{xyz[1]:.3f}" if on_target else "", "target_z_m": f"{xyz[2]:.3f}" if on_target else "",
            "latency_ms": f"{self.lat_cmd_ms[-1]:.2f}" if (self.lat_cmd_ms and st == TRACKING) else "",
            "loop_dt_ms": f"{loop.dt_avg * 1000:.3f}",
            "faults": ";".join(f"{k}:{v}" for k, v in self.alerts.items()),
            "true_err_az_deg": "" if math.isnan(te[0]) else f"{te[0]:.4f}",
            "true_err_el_deg": "" if math.isnan(te[1]) else f"{te[1]:.4f}",
            "true_in_fov": "" if te[2] is None else int(te[2]),
        }
        self.logger.log_telemetry(row)
        self.telemetry = row
        if st == TRACKING and meas is not None:
            self.stats["max_err_deg"] = max(self.stats["max_err_deg"], D(math.hypot(meas.err_az, meas.err_el)))
        self.bus.publish(Topics.STATE, {"t": t, "state": st, "reason": self.sm.reason,
                                        "reacquire_remaining_s": self.sm.reacquire_remaining()})
        self.bus.publish(Topics.GIMBAL_STATE, g)
        if cmd is not None:
            self.bus.publish(Topics.GIMBAL_CMD, cmd)
        if rng is not None:
            self.bus.publish(Topics.LIDAR, rng)
        if est is not None:
            self.bus.publish(Topics.TARGET_ESTIMATE, {"t": t, "az_rad": est.az, "el_rad": est.el, "az_rate": est.az_rate,
                                                      "el_rate": est.el_rate, "range_m": rng.distance_m if on_target else None,
                                                      "xyz": xyz if on_target else None})

    # ================================================================== state changes
    def _on_state_change(self, old: str, new: str, reason: str, t: float) -> None:
        self._event("state", f"{old} -> {new}: {reason}")
        self.bus.publish(Topics.EVENTS, {"type": "state", "old": old, "new": new, "reason": reason, "t": t})
        if new == ESTOP:
            self.gimbal.estop()
            self._pending_clear = True
        if old == ESTOP and new == IDLE:
            self.gimbal.reset_estop()
        if new == LOST:
            self.stats["lost_alerts"] += 1
            self._homed_after_lost = False
        if new == SEARCHING:
            self._entered_search_t = t
            self._hold_sent_t = 0.0
        if new == IDLE:
            self._pending_clear = True
            self._jog_sp = None
            if self.logger.active and self.cfg.get_path("logging.auto_record", True) and old in ACTIVE:
                self.logger.stop_session(self.summary())
                self._event("log", "recording stopped")
        if old == IDLE and new == SEARCHING and self.cfg.get_path("logging.auto_record", True) and not self.logger.active:
            self._start_logging("track")
        self.logger.log_event("state", {"old": old, "new": new, "reason": reason})

    def _start_logging(self, label: str) -> None:
        meta = {"version": __version__, "platform": platform.platform(), "python": platform.python_version(),
                "config_source": self.cfg.get("_source", ""), "config": self.cfg.to_dict(),
                "intrinsics": self.intr.as_dict() if self.intr else None, "calibration": self.calib,
                "sensor": self.active_sensor, "gimbal": self.gimbal.kind,
                "detector": self.detector.info() if self.detector else None,
                "scenario": self.world.scenario if self.world else None,
                "telemetry_note": "angles in degrees; pan + right, tilt + up; dx + right, dy + up of aim point"}
        label = f"{label}_{self.world.scenario}" if self.world else f"{label}_{self.active_sensor}"
        d = self.logger.start_session(label, meta)
        self._event("log", f"recording to {d}")

    # ================================================================== helpers
    def _event(self, kind: str, text: str) -> None:
        self.events.append((time.time(), kind, text))
        if self.logger.active:
            self.logger.log_event(kind, {"text": text})

    def _on_ocr(self, result: dict) -> None:
        self.ocr_last = dict(result, t=time.time())
        txt = result.get("text") or result.get("error", "")
        self._event("ocr", f"OCR: '{txt}' ({result.get('ms', '?')} ms)")
        self.bus.publish(Topics.OCR, self.ocr_last)

    def _request_ocr(self) -> tuple[bool, str]:
        lk = self.lock
        frame = self.framebuf.latest()
        if lk is None or frame is None:
            return False, "no locked target to read"
        ok = self.ocr.request(frame.image, lk.bbox, {"frame_id": frame.frame_id, "target_id": lk.track_id})
        return ok, "" if ok else (self.ocr.reason or "OCR busy")

    def _switch_sensor(self, name: str) -> tuple[bool, str]:
        if name not in self.sensors:
            return False, f"unknown sensor {name!r}"
        if name == self.active_sensor and self.camera is not None:
            return True, "already active"
        t0 = clock.now()
        self._pending_clear = True
        old = self.camera
        if old is not None:
            old.stop()
        self.camera = None
        if not self._start_camera(name):
            if old is not None:
                self._start_camera(old.name)
            return False, f"could not start {name}; reverted"
        dt = clock.now() - t0
        self._event("sensor", f"switched to {name} in {dt:.2f} s (ES-07.2: <= 2 s)")
        return True, f"{name} active in {dt:.2f} s"

    def _set_boresight(self, x: float, y: float) -> None:
        self.calib["boresight_px"] = [x, y]
        self._save_calibration()
        self._event("calibration", f"aim point set to ({x:.3f}, {y:.3f}) of frame")

    def _load_calibration(self) -> dict:
        try:
            if self.calib_path.exists():
                return json.loads(self.calib_path.read_text())
        except Exception as exc:
            log.warning("could not read calibration: %s", exc)
        return {}

    def _save_calibration(self) -> None:
        try:
            self.calib_path.write_text(json.dumps(self.calib, indent=2))
        except Exception as exc:
            log.warning("could not save calibration: %s", exc)

    # ---- calibration routine (UC-7)
    def _calibrate(self, known_m: float) -> None:
        if not self.sm.fire("calibrate", f"calibration at {known_m:.3f} m"):
            return
        res = {"known_distance_m": known_m, "steps": []}
        try:
            self.gimbal.home()
            t_end = clock.now() + 6.0
            while clock.now() < t_end:
                g = self.gimbal.state()
                if abs(D(g.pan)) < 0.3 and abs(D(g.tilt)) < 0.3 and abs(D(g.pan_rate)) < 1 and abs(D(g.tilt_rate)) < 1:
                    break
                time.sleep(0.05)
            time.sleep(0.4)
            g = self.gimbal.state()
            res["steps"].append({"step": "home", "pan_deg": round(D(g.pan), 3), "tilt_deg": round(D(g.tilt), 3),
                                 "ok": abs(D(g.pan)) < 1 and abs(D(g.tilt)) < 1})
            zp, zt = self.gimbal.zero_here()
            self.calib["zero_pan_deg"], self.calib["zero_tilt_deg"] = D(zp), D(zt)
            res["steps"].append({"step": "encoder zero", "zero_pan_deg": round(D(zp), 4), "zero_tilt_deg": round(D(zt), 4),
                                 "ok": True})
            if self.lidar is not None and not isinstance(self.lidar, NoLidar):
                time.sleep(1.0)
                mean, n = self.lidar.mean_over(1.0)
                if mean is not None and n >= 5:
                    raw = mean - self.lidar.range_offset_m
                    self.lidar.range_offset_m = known_m - raw
                    self.calib["range_offset_m"] = self.lidar.range_offset_m
                    time.sleep(0.3)
                    after, _ = self.lidar.mean_over(0.25)
                    err_in = abs((after or mean) - known_m) / 0.0254
                    res["steps"].append({"step": "lidar range offset", "raw_mean_m": round(raw, 4), "n": n,
                                         "offset_m": round(self.lidar.range_offset_m, 4), "residual_in": round(err_in, 2),
                                         "ok": err_in <= 2.0})
                else:
                    res["steps"].append({"step": "lidar range offset", "ok": False, "error": "no valid LiDAR readings"})
            self.calib["calibrated_wall"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_calibration()
            res["ok"] = all(s.get("ok") for s in res["steps"])
        except Exception as exc:
            res["ok"] = False
            res["error"] = str(exc)
        self.calib_result = res
        self._event("calibration", json.dumps(res))
        self.sm.fire("calibrated", "calibration " + ("passed" if res.get("ok") else "FAILED"))

    # ---- self test (ES-10.1)
    def _selftest(self) -> None:
        checks = []

        def add(name, ok, detail):
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        cam = self.camera
        fps = cam.fps_meter.value() if cam else 0
        add("camera streaming", fps >= float(self.cfg.get_path("selftest.min_fps", 30)), f"{fps:.1f} fps ({self.active_sensor})")
        if self.intr:
            add("camera field of view", self.intr.hfov_deg >= 40.0, f"HFOV {self.intr.hfov_deg:.1f} deg, "
                f"{self.intr.px_per_deg:.1f} px/deg (ES-06.2 needs >= 28 at 20 ft for a 3 in target)")
        mi = self.macro.info() if self.macro else {}
        add("detector running", mi.get("hz", 0) > 0, f"{mi.get('name')} {mi.get('hz', 0)} Hz, {mi.get('infer_ms', 0)} ms")
        age = self.gimbal.feedback_age()
        add("gimbal feedback", age < 0.5, f"{self.gimbal.kind}: age {min(age, 99):.3f} s, {self.gimbal.fb_rate.value():.0f} Hz")
        if self.cfg.get_path("selftest.move_gimbal", True) and self.sm.state == IDLE and self.gimbal.kind != "none":
            g0 = self.gimbal.state()
            self.gimbal.command(GimbalCommand("angle", g0.pan + R(3.0), g0.tilt))
            time.sleep(1.0)
            g1 = self.gimbal.state()
            moved = D(g1.pan - g0.pan)
            self.gimbal.command(GimbalCommand("angle", g0.pan, g0.tilt))
            add("gimbal motion (+3 deg pan)", 2.0 < moved < 4.0, f"moved {moved:.2f} deg")
        if self.lidar is not None and not isinstance(self.lidar, NoLidar):
            li = self.lidar.info()
            errs = li.get("checksum_errors", 0)
            tot = max(1, li.get("frames_ok", 0) + errs)
            add("LiDAR", li.get("rate_hz", 0) >= 10 and errs / tot < 0.02,
                f"{li.get('rate_hz')} Hz, {li.get('distance_m', '?')} m, checksum errors {errs}/{tot}")
        lg = self.logger.info()
        add("log storage", (lg.get("free_gb") or 0) > self.logger.min_free_gb, f"{lg.get('free_gb')} GB free in {self.logger.root}")
        add("OCR engine", self.ocr.available, getattr(self.ocr, "version", self.ocr.reason))
        lp = self.loop
        add("control loop rate", lp is not None and lp.hz > 0.9 / lp.period,
            f"{lp.hz:.1f} Hz (target {1 / lp.period:.0f}), overruns {lp.overruns}" if lp else "not running")
        if self.ros2 is not None or self.cfg.get_path("ros2.enabled", False):
            add("ROS 2 bridge", self.ros2 is not None, "active" if self.ros2 else "rclpy not available")
        self.selftest_result = {"t": time.time(), "ok": all(c["ok"] for c in checks), "checks": checks}
        self._event("selftest", ("PASS" if self.selftest_result["ok"] else "FAIL") + " - " +
                    ", ".join(f"{c['name']}:{'ok' if c['ok'] else 'FAIL'}" for c in checks))

    # ================================================================== snapshot for UI / gateway
    def latency_stats(self) -> dict:
        lo, lc = list(self.lat_offset_ms), list(self.lat_cmd_ms)
        return {"offset_p50": _pct(lo, 50), "offset_p95": _pct(lo, 95), "cmd_p50": _pct(lc, 50),
                "cmd_p95": _pct(lc, 95), "cmd_max": max(lc) if lc else None}

    def jitter_stats(self, window_s: float = 1.0) -> dict:
        t_now = clock.now()
        pts = [(a, e) for (t, a, e) in list(self._err_window) if t_now - t <= window_s]
        if len(pts) < 5:
            return {}
        arr = np.asarray(pts)
        rms = float(np.sqrt(np.mean(arr[:, 0] ** 2 + arr[:, 1] ** 2)))
        sd = float(np.sqrt(np.var(arr[:, 0]) + np.var(arr[:, 1])))
        k = math.radians(1.0) * 240.0  # inches at 20 ft per degree
        return {"rms_deg": rms, "jitter_sd_deg": sd, "rms_in_20ft": rms * k, "jitter_dia_in_20ft": 2 * sd * k, "n": len(pts)}

    def summary(self) -> dict:
        s = dict(self.stats)
        s["reacquire_times_s"] = list(s["reacquire_times_s"])
        s["latency"] = self.latency_stats()
        s["in_fov_ratio"] = (s["frames_in_fov"] / s["frames_truth"]) if s["frames_truth"] else None
        s["vision_skipped_frames"] = self.vision_skipped
        if self.loop:
            s["control_hz"] = round(self.loop.hz, 2)
            s["control_overruns"] = self.loop.overruns
        return s

    def snapshot(self, with_frame: bool = True) -> dict[str, Any]:
        frame = self.framebuf.latest()
        ls = self.latency_stats()
        snap = {
            "mode": "local",
            "version": __version__,
            "config": Path(str(self.cfg.get("_source", ""))).name,
            "frame": frame.image if (with_frame and frame is not None) else None,
            "frame_id": frame.frame_id if frame is not None else -1,
            "overlay": self.overlay,
            "state": self.sm.snapshot(),
            "telemetry": self.telemetry,
            "latency": ls,
            "jitter": self.jitter_stats(),
            "rates": {"camera": round(self.camera.fps_meter.value(), 1) if self.camera else 0.0,
                      "vision": round(self.vision_rate.value(), 1), "vision_ms": round(self.vision_ms, 2),
                      "control": round(self.control_rate.value(), 1),
                      "macro": round(self.macro.rate.value(), 1) if self.macro else 0.0},
            "gimbal": self.gimbal.info(),
            "lidar": self.lidar.info() if self.lidar else {},
            "camera": self.camera.info() if self.camera else {},
            "detector": self.macro.info() if self.macro else {},
            "micro": {"backend": getattr(self.micro, "name", "?"), "available": available_trackers()},
            "controller": self.controller.info(),
            "logger": self.logger.info(),
            "ocr": self.ocr_last,
            "ocr_available": self.ocr.available,
            "selftest": self.selftest_result,
            "calibration": self.calib,
            "calibration_result": self.calib_result,
            "events": [{"t": t, "kind": k, "text": x} for (t, k, x) in list(self.events)[-40:]],
            "alerts": dict(self.alerts),
            "seek_class": self.selector.target_class,
            "seek_classes": list(self.cfg.get_path("seek.ui_classes", ["auto", "any", "drone", "ball", "card", "hand", "person"])),
            "sensors": list(self.sensors.keys()),
            "active_sensor": self.active_sensor,
            "scenario": self.world.scenario if self.world else None,
            "summary": self.summary(),
        }
        if self.world is not None:
            from .sim.world import SCENARIOS

            snap["scenarios"] = list(SCENARIOS)
        return snap
