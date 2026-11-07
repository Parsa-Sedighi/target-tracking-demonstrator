"""Gremsy-like MAVLink gimbal emulator, for testing the MAVLink backend with no hardware.

Speaks the subset of MAVLink that gSDK uses (both protocol flavours):
  HEARTBEAT (compid 154, MAV_TYPE_GIMBAL) at 1 Hz
  COMMAND_LONG: DO_MOUNT_CONTROL (angle / speed), USER_1 (motors), USER_2 (mode),
                SET_MESSAGE_INTERVAL  -> COMMAND_ACK
  GIMBAL_DEVICE_SET_ATTITUDE (v2: quaternion or angular velocity)
  -> MOUNT_ORIENTATION (v1) and GIMBAL_DEVICE_ATTITUDE_STATUS (v2) at the requested rate

Usage:
    python tools/fake_gremsy.py --listen 5760            # TCP server
    python -m aimbot -c jetson_hw --set sensors.vis.backend=sim ... --set gimbal.port=tcp:127.0.0.1:5760
    python tools/gimbal_test.py --port tcp:127.0.0.1:5760
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MAVLINK20", "1")   # Gremsy / gSDK use MAVLink 2

from pymavlink import mavutil  # noqa: E402

from aimbot.gimbal.mavlink_gremsy import euler_to_quat, quat_to_euler  # noqa: E402
from aimbot.gimbal.sim import SimGimbal  # noqa: E402


class FakeGremsy:
    def __init__(self, url: str, sysid: int = 1, compid: int = 154):
        self.conn = mavutil.mavlink_connection(url, source_system=sysid, source_component=compid)
        self.mav = self.conn.mav
        self.dyn = SimGimbal({"max_rate_dps": 100, "sim_cmd_latency_ms": 2, "sim_fb_latency_ms": 0,
                              "limits_deg": {"pan_min": -175, "pan_max": 175, "tilt_min": -120, "tilt_max": 120}})
        self.rates = {265: 20.0, 285: 20.0}   # Hz
        self.mode = 2
        self.motors = True
        self.flags = 4 | 8
        self.t0 = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cmds = 0

    def _tx(self, fn, *a, **k):
        with self.lock:
            try:
                fn(*a, **k)
            except Exception:
                pass

    def _ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000) & 0xFFFFFFFF

    def run(self) -> None:
        threading.Thread(target=self._telemetry, daemon=True).start()
        while not self.stop.is_set():
            msg = self.conn.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "COMMAND_LONG":
                self._command(msg)
            elif t == "GIMBAL_DEVICE_SET_ATTITUDE":
                self.cmds += 1
                self.flags = msg.flags
                if msg.flags & 1:
                    self.motors = False
                    self.dyn._set_motors(False)
                    continue
                if not self.motors:
                    self.motors = True
                    self.dyn._set_motors(True)
                if msg.flags & 2:
                    self.dyn._send_angles(0.0, 0.0)
                    continue
                q = list(msg.q)
                if not any(math.isnan(v) for v in q):
                    _, pitch, yaw = quat_to_euler(q)
                    self.dyn._send_angles(yaw, pitch)
                else:
                    self.dyn._send_rates(msg.angular_velocity_z, msg.angular_velocity_y)

    def _command(self, msg) -> None:
        c = msg.command
        p = [msg.param1, msg.param2, msg.param3, msg.param4, msg.param5, msg.param6, msg.param7]
        res = mavutil.mavlink.MAV_RESULT_ACCEPTED
        if c == mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL:
            self.cmds += 1
            pitch, _roll, yaw, mode = p[0], p[1], p[2], int(p[5])
            if mode == 2:
                self.dyn._send_rates(math.radians(yaw), math.radians(pitch))
            else:
                self.dyn._send_angles(math.radians(yaw), math.radians(pitch))
        elif c == 31010:       # USER_1: motors
            self.motors = bool(p[6])
            self.dyn._set_motors(self.motors)
        elif c == 31011:       # USER_2: mode
            self.mode = int(p[6])
        elif c == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
            mid, interval = int(p[0]), p[1]
            if interval > 0:
                self.rates[mid] = 1e6 / interval
            elif interval < 0:
                self.rates[mid] = 0.0
        else:
            res = mavutil.mavlink.MAV_RESULT_UNSUPPORTED
        self._tx(self.mav.command_ack_send, c, res, 0, 0, msg.get_srcSystem(), msg.get_srcComponent())

    def _telemetry(self) -> None:
        last = {265: 0.0, 285: 0.0, "hb": 0.0}
        while not self.stop.is_set():
            now = time.monotonic()
            if now - last["hb"] >= 1.0:
                last["hb"] = now
                self._tx(self.mav.heartbeat_send, mavutil.mavlink.MAV_TYPE_GIMBAL, mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                         0, 0, mavutil.mavlink.MAV_STATE_ACTIVE)
            pan, tilt, vp, vt = self.dyn._true_state(time.perf_counter())
            for mid in (265, 285):
                hz = self.rates.get(mid, 0.0)
                if hz > 0 and now - last[mid] >= 1.0 / hz:
                    last[mid] = now
                    if mid == 265:
                        self._tx(self.mav.mount_orientation_send, self._ms(), 0.0, math.degrees(tilt), math.degrees(pan),
                                 math.degrees(pan))
                    else:
                        q = euler_to_quat(0.0, tilt, pan)
                        self._tx(self.mav.gimbal_device_attitude_status_send, 0, 0, self._ms(),
                                 self.flags & ~1 if self.motors else self.flags | 1, q, 0.0, vt, vp, 0)
            time.sleep(0.002)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=5760, help="TCP port")
    ap.add_argument("--url", help="any pymavlink URL instead (e.g. a serial port for a loop-back cable)")
    a = ap.parse_args()
    url = a.url or f"tcpin:0.0.0.0:{a.listen}"
    print(f"fake Gremsy gimbal on {url} (Ctrl+C to stop)", flush=True)
    g = FakeGremsy(url)
    try:
        g.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
