"""Gremsy Mio over MAVLink (pymavlink), mirroring what Gremsy's gSDK does.

Two protocol flavours, both implemented in gSDK (github.com/Gremsy/gSDK):

* ``v1`` (gSDK's example default, and what its Mio log shows):
    - move:  COMMAND_LONG MAV_CMD_DO_MOUNT_CONTROL
             p1=pitch, p2=roll, p3=yaw [deg or deg/s], p6=input mode (1 angle, 2 speed),
             p7=MAV_MOUNT_MODE_MAVLINK_TARGETING
    - mode:  COMMAND_LONG MAV_CMD_USER_2, p7 = 1 lock / 2 follow
    - motor: COMMAND_LONG MAV_CMD_USER_1, p7 = 1 on / 0 off
    - feedback: MOUNT_ORIENTATION (deg; ``yaw`` is relative to the base)
* ``v2`` (MAVLink gimbal protocol v2):
    - move:  GIMBAL_DEVICE_SET_ATTITUDE (quaternion for angles, NaN quaternion + angular
             velocity for rates), flags ROLL_LOCK|PITCH_LOCK (+YAW_LOCK in lock mode)
    - feedback: GIMBAL_DEVICE_ATTITUDE_STATUS

Defaults: 115200 8N1, our sysid 1 / compid 191 (MAV_COMP_ID_ONBOARD_COMPUTER); the
gimbal announces itself with a HEARTBEAT from compid 154 (MAV_COMP_ID_GIMBAL).

``follow`` mode is the default because the base is stationary: yaw is then measured
relative to the base by the encoders instead of an IMU heading that drifts.

``port`` can be a serial device (COM5, /dev/ttyUSB0, /dev/ttyTHS1) or any pymavlink
URL (tcp:127.0.0.1:5760 for tools/fake_gremsy.py).

Not yet run against a physical Mio. tools/gimbal_test.py exercises each step
one at a time for the first bench session.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Optional

# gSDK talks MAVLink 2 (message ids > 255 such as MOUNT_ORIENTATION=265 and GIMBAL_DEVICE_*=284/285
# do not exist in MAVLink 1). pymavlink defaults to v1 unless this is set before it is imported.
os.environ.setdefault("MAVLINK20", "1")

from .. import clock
from .base import GimbalBase

log = logging.getLogger(__name__)

GIMBAL_COMP_IDS = {154, 171, 172, 173, 174, 175}
MAV_CMD_USER_1 = 31010
MAV_CMD_USER_2 = 31011
MODE_LOCK = 1
MODE_FOLLOW = 2


def euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def quat_to_euler(q) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class MavlinkGremsyGimbal(GimbalBase):
    kind = "gremsy"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = self.cfg
        self.port = str(c.get("port", "/dev/ttyUSB0"))
        self.baud = int(c.get("baud", 115200))
        self.protocol = str(c.get("protocol", "v1")).lower()
        self.gimbal_mode = str(c.get("gimbal_mode", "follow")).lower()
        self.sysid = int(c.get("sysid", 1))
        self.compid = int(c.get("compid", 191))
        self.fb_hz = float(c.get("feedback_hz", 50))
        self.send_hz = float(c.get("max_send_hz", 50))
        self.fb_latency = float(c.get("feedback_latency_ms", 5.0)) / 1000.0
        self.connect_timeout = float(c.get("connect_timeout_s", 5.0))
        self.conn = None
        self.mav = None
        self.target_sys: Optional[int] = None
        self.target_comp: Optional[int] = None
        self._stop = threading.Event()
        self._tx_lock = threading.Lock()
        self._pending = None            # latest (kind, a, b) awaiting the sender thread
        self._pending_lock = threading.Lock()
        self._hb_event = threading.Event()
        self.acks: dict[int, dict] = {}
        self._ack_waiters: dict[int, threading.Event] = {}
        self.last_hb = 0.0
        self.sys_errors = 0
        self.model_name = ""
        self.motors_on = False

    # ------------------------------------------------------------------ connection
    def start(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError("pymavlink is not installed (pip install pymavlink)") from exc
        self.mavutil = mavutil
        kwargs = dict(source_system=self.sysid, source_component=self.compid)
        if not any(self.port.startswith(p) for p in ("tcp", "udp")):
            kwargs["baud"] = self.baud
        log.info("Gremsy: opening %s (%s, protocol %s, %s mode)", self.port, self.baud, self.protocol, self.gimbal_mode)
        self.conn = mavutil.mavlink_connection(self.port, **kwargs)
        self.mav = self.conn.mav
        try:
            self.mav.srcSystem, self.mav.srcComponent = self.sysid, self.compid
        except Exception:
            pass
        self._stop.clear()
        threading.Thread(target=self._rx_loop, name="gremsy-rx", daemon=True).start()
        threading.Thread(target=self._hb_loop, name="gremsy-hb", daemon=True).start()
        threading.Thread(target=self._tx_loop, name="gremsy-tx", daemon=True).start()
        if not self._hb_event.wait(self.connect_timeout):
            raise RuntimeError(f"no heartbeat from a Gremsy gimbal on {self.port} within {self.connect_timeout:.0f} s "
                               "(check wiring TX->RX, 3.3 V logic, baud, gimbal power)")
        log.info("Gremsy: gimbal found sysid=%s compid=%s", self.target_sys, self.target_comp)
        self._configure()

    def _configure(self) -> None:
        mav = self.mavutil.mavlink
        fb_msg = 265 if self.protocol == "v1" else 285    # MOUNT_ORIENTATION / GIMBAL_DEVICE_ATTITUDE_STATUS
        interval_us = 1e6 / max(1.0, self.fb_hz)
        r = self.command_long_sync(mav.MAV_CMD_SET_MESSAGE_INTERVAL, [fb_msg, interval_us, 0, 0, 0, 0, 0])
        log.info("Gremsy: set feedback interval -> %s", r)
        if self.protocol == "v1":
            mode = MODE_FOLLOW if self.gimbal_mode == "follow" else MODE_LOCK
            log.info("Gremsy: set mode %s -> %s", self.gimbal_mode,
                     self.command_long_sync(MAV_CMD_USER_2, [0, 0, 0, 0, 0, 0, mode]))
        self._set_motors(True)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ threads
    def _hb_loop(self) -> None:
        mav = None
        while not self._stop.is_set():
            try:
                mav = self.mavutil.mavlink
                with self._tx_lock:
                    self.mav.heartbeat_send(mav.MAV_TYPE_ONBOARD_CONTROLLER, mav.MAV_AUTOPILOT_INVALID, 0, 0,
                                            mav.MAV_STATE_ACTIVE)
            except Exception as exc:
                log.debug("heartbeat send failed: %s", exc)
            self._stop.wait(1.0)

    def _tx_loop(self) -> None:
        period = 1.0 / max(1.0, self.send_hz)
        t_next = clock.now()
        while not self._stop.is_set():
            clock.sleep_until(t_next, self._stop)
            t_next = max(t_next + period, clock.now())
            with self._pending_lock:
                item, self._pending = self._pending, None
            if item is None or self.target_sys is None:
                continue
            try:
                kind, a, b = item
                if kind == "angle":
                    self._tx_angles(a, b)
                elif kind == "rate":
                    self._tx_rates(a, b)
            except Exception as exc:
                log.warning("Gremsy tx failed: %s", exc)

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.2)
            except Exception as exc:
                log.debug("rx error: %s", exc)
                self._stop.wait(0.05)
                continue
            if msg is None:
                continue
            t_rx = clock.now()
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            src_comp = msg.get_srcComponent()
            if mtype == "HEARTBEAT":
                if src_comp in GIMBAL_COMP_IDS:
                    self.target_sys, self.target_comp = msg.get_srcSystem(), src_comp
                    self.last_hb = t_rx
                    self._hb_event.set()
            elif mtype == "MOUNT_ORIENTATION":
                self._on_feedback(t_rx - self.fb_latency, math.radians(msg.yaw), math.radians(msg.pitch),
                                  motors_on=self.motors_on, mode=f"v1-{self.gimbal_mode}")
            elif mtype == "GIMBAL_DEVICE_ATTITUDE_STATUS":
                roll, pitch, yaw = quat_to_euler(msg.q)
                dy = getattr(msg, "delta_yaw", float("nan"))
                if self.gimbal_mode == "lock" and dy is not None and not math.isnan(dy):
                    yaw = dy
                vz, vy = msg.angular_velocity_z, msg.angular_velocity_y
                self._on_feedback(t_rx - self.fb_latency, yaw, pitch, vz, vy, motors_on=not (msg.flags & 1),
                                  mode=f"v2-flags{msg.flags}")
            elif mtype == "COMMAND_ACK":
                self.acks[msg.command] = {"result": msg.result, "t": t_rx}
                ev = self._ack_waiters.get(msg.command)
                if ev is not None:
                    ev.set()
            elif mtype == "SYS_STATUS":
                self.sys_errors = getattr(msg, "errors_count2", 0)
                if self.sys_errors:
                    with self._lock:
                        self._state.fault = f"gimbal error code {self.sys_errors}"
            elif mtype == "GIMBAL_DEVICE_INFORMATION":
                self.model_name = str(getattr(msg, "model_name", ""))

    # ------------------------------------------------------------------ low level
    def command_long(self, command: int, params: list[float]) -> None:
        if self.target_sys is None:
            return
        p = (list(params) + [0.0] * 7)[:7]
        with self._tx_lock:
            self.mav.command_long_send(self.target_sys, self.target_comp, command, 0, *p)

    def command_long_sync(self, command: int, params: list[float], timeout: float = 1.0) -> str:
        ev = threading.Event()
        self._ack_waiters[command] = ev
        self.acks.pop(command, None)
        self.command_long(command, params)
        ok = ev.wait(timeout)
        self._ack_waiters.pop(command, None)
        if not ok:
            return "TIMEOUT"
        res = self.acks.get(command, {}).get("result")
        names = {0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED", 3: "UNSUPPORTED", 4: "FAILED", 5: "IN_PROGRESS"}
        return names.get(res, str(res))

    def _flags(self) -> int:
        mav = self.mavutil.mavlink
        f = mav.GIMBAL_DEVICE_FLAGS_ROLL_LOCK | mav.GIMBAL_DEVICE_FLAGS_PITCH_LOCK
        if self.gimbal_mode == "lock":
            f |= mav.GIMBAL_DEVICE_FLAGS_YAW_LOCK
        return f

    def _tx_angles(self, pan: float, tilt: float) -> None:
        mav = self.mavutil.mavlink
        if self.protocol == "v1":
            self.command_long(mav.MAV_CMD_DO_MOUNT_CONTROL,
                              [math.degrees(tilt), 0.0, math.degrees(pan), 0, 0, 1, mav.MAV_MOUNT_MODE_MAVLINK_TARGETING])
        else:
            q = euler_to_quat(0.0, tilt, pan)
            nan = float("nan")
            with self._tx_lock:
                self.mav.gimbal_device_set_attitude_send(self.target_sys, self.target_comp, self._flags(), q, nan, nan, nan)

    def _tx_rates(self, pan_rate: float, tilt_rate: float) -> None:
        mav = self.mavutil.mavlink
        if self.protocol == "v1":
            self.command_long(mav.MAV_CMD_DO_MOUNT_CONTROL,
                              [math.degrees(tilt_rate), 0.0, math.degrees(pan_rate), 0, 0, 2,
                               mav.MAV_MOUNT_MODE_MAVLINK_TARGETING])
        else:
            nan = float("nan")
            with self._tx_lock:
                self.mav.gimbal_device_set_attitude_send(self.target_sys, self.target_comp, self._flags(),
                                                         [nan, nan, nan, nan], 0.0, tilt_rate, pan_rate)

    # ------------------------------------------------------------------ GimbalBase hooks
    def _send_angles(self, pan: float, tilt: float) -> None:
        with self._pending_lock:
            self._pending = ("angle", pan, tilt)

    def _send_rates(self, pan_rate: float, tilt_rate: float) -> None:
        with self._pending_lock:
            self._pending = ("rate", pan_rate, tilt_rate)

    def _set_motors(self, on: bool) -> None:
        if self.target_sys is None:
            return
        if self.protocol == "v1":
            def _send_and_log():
                r = self.command_long_sync(MAV_CMD_USER_1, [0, 0, 0, 0, 0, 0, 1 if on else 0])
                log.info("Gremsy: motors %s -> %s", "ON" if on else "OFF", r)

            # never block the caller (E-STOP is pressed from the GUI thread)
            threading.Thread(target=_send_and_log, name="gremsy-motors", daemon=True).start()
        else:
            mav = self.mavutil.mavlink
            flags = mav.GIMBAL_DEVICE_FLAGS_NEUTRAL if on else mav.GIMBAL_DEVICE_FLAGS_RETRACT
            nan = float("nan")
            with self._tx_lock:
                self.mav.gimbal_device_set_attitude_send(self.target_sys, self.target_comp, flags,
                                                         [nan, nan, nan, nan], 0.0, 0.0, 0.0)
        self.motors_on = on

    def estop(self) -> None:
        # bypass the rate-limited sender: stop now, then motors off
        try:
            if self.target_sys is not None:
                self._tx_rates(0.0, 0.0)
        except Exception:
            pass
        super().estop()

    def info(self) -> dict:
        d = super().info()
        d.update({"port": self.port, "protocol": self.protocol, "gimbal_mode": self.gimbal_mode,
                  "target": f"{self.target_sys}/{self.target_comp}", "heartbeat_age_s":
                  round(clock.now() - self.last_hb, 2) if self.last_hb else None, "model": self.model_name,
                  "sys_errors": self.sys_errors})
        return d
