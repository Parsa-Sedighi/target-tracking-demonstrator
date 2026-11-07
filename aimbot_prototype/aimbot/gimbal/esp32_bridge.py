"""Jetson <-> ESP32-S3 supervisory bridge (proposed protocol v0, see docs/ICD.md section 4).

Final architecture from the part list: Jetson --USB serial--> ESP32-S3 carrier PCB
--3.3 V UART/MAVLink--> Gremsy Mio, and TF02-Pro --UART--> ESP32. The ESP32 only
validates/forwards commands, reports LiDAR + power + faults, and runs a watchdog
(stop all motion if the Jetson goes quiet for > 500 ms, ES-11.3).

Frame:  A5 5A | type u8 | seq u8 | len u8 | payload[len] | crc16 u16 LE
CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over type..payload. Little-endian payloads.

This module is the Jetson side. tools/esp32_emulator.py implements the ESP32 side
for loop-back testing until the firmware exists.
"""

from __future__ import annotations

import logging
import struct
import threading
from typing import Callable, Optional

from .. import clock
from .base import GimbalBase

log = logging.getLogger(__name__)

SYNC = b"\xA5\x5A"

# Jetson -> ESP32
J2E_HEARTBEAT = 0x01   # <I B   t_ms, jetson_state
J2E_ANGLE = 0x10       # <i i   pan_cdeg, tilt_cdeg (centi-degrees)
J2E_RATE = 0x11        # <i i   pan_cdps, tilt_cdps (centi-deg/s)
J2E_MOTORS = 0x12      # <B     on
J2E_HOME = 0x13        # -
J2E_ESTOP = 0x20       # -
J2E_CLEAR_ESTOP = 0x21  # -
# ESP32 -> Jetson
E2J_HEARTBEAT = 0x81   # <I H   t_ms, status bits (0 gimbal link, 1 lidar ok, 2 hw e-stop, 3 watchdog tripped)
E2J_GIMBAL = 0x90      # <I i i i B  t_ms, pan_cdeg, tilt_cdeg, roll_cdeg, flags (bit0 motors on)
E2J_LIDAR = 0xA0       # <I H H H    t_ms, dist_cm, strength, temp_raw
E2J_POWER = 0xB0       # <I H H H    t_ms, bus_mV, gimbal_mA, fault bits
E2J_ACK = 0xF0         # <B B B      acked_type, acked_seq, result (0 ok)
E2J_FAULT = 0xF1       # <I H        t_ms, fault code

STATE_CODES = {"IDLE": 0, "SEARCHING": 1, "TRACKING": 2, "LOST": 3, "PAUSED": 4, "CALIBRATING": 5, "ESTOP": 6, "FAULT": 7}


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode(msg_type: int, seq: int, payload: bytes = b"") -> bytes:
    body = bytes([msg_type & 0xFF, seq & 0xFF, len(payload)]) + payload
    return SYNC + body + struct.pack("<H", crc16_ccitt(body))


class FrameParser:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.crc_errors = 0
        self.frames = 0

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        out = []
        buf = self.buf
        buf.extend(data)
        while True:
            i = buf.find(SYNC)
            if i < 0:
                del buf[:-1]
                break
            if i > 0:
                del buf[:i]
            if len(buf) < 5:
                break
            n = buf[4]
            total = 2 + 3 + n + 2
            if len(buf) < total:
                break
            body = bytes(buf[2:5 + n])
            crc = struct.unpack_from("<H", buf, 5 + n)[0]
            if crc16_ccitt(body) != crc:
                self.crc_errors += 1
                del buf[:2]
                continue
            out.append((body[0], body[1], body[3:]))
            self.frames += 1
            del buf[:total]
        return out


class Esp32Link:
    """Shared serial link (one per port) used by the gimbal and LiDAR backends."""

    _registry: dict[str, "Esp32Link"] = {}
    _reg_lock = threading.Lock()

    @classmethod
    def get(cls, port: str, baud: int = 921600) -> "Esp32Link":
        with cls._reg_lock:
            link = cls._registry.get(port)
            if link is None:
                link = cls(port, baud)
                cls._registry[port] = link
            return link

    def __init__(self, port: str, baud: int):
        self.port, self.baud = port, baud
        self.ser = None
        self.parser = FrameParser()
        self.handlers: dict[int, list[Callable[[float, int, bytes], None]]] = {}
        self._seq = 0
        self._tx_lock = threading.Lock()
        self._stop = threading.Event()
        self._started = False
        self.jetson_state = 0
        self.esp_status = 0
        self.last_rx = 0.0
        self._offset: Optional[float] = None   # host_time - esp_time
        self.acks = 0
        self.nacks = 0

    def on(self, msg_type: int, cb: Callable[[float, int, bytes], None]) -> None:
        self.handlers.setdefault(msg_type, []).append(cb)

    def start(self) -> None:
        if self._started:
            return
        import serial

        self.ser = serial.serial_for_url(self.port, baudrate=self.baud, timeout=0.02)
        self._stop.clear()
        threading.Thread(target=self._rx_loop, name="esp32-rx", daemon=True).start()
        threading.Thread(target=self._hb_loop, name="esp32-hb", daemon=True).start()
        self._started = True
        log.info("ESP32 bridge on %s", self.port)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self._started = False
        with Esp32Link._reg_lock:
            Esp32Link._registry.pop(self.port, None)

    def send(self, msg_type: int, payload: bytes = b"") -> int:
        with self._tx_lock:
            seq = self._seq
            self._seq = (self._seq + 1) & 0xFF
            if self.ser is not None:
                self.ser.write(encode(msg_type, seq, payload))
        return seq

    def esp_to_host(self, t_ms: int, t_rx: float) -> float:
        """Map an ESP32 millisecond timestamp onto clock.now() (min-delay filter)."""
        est = t_rx - t_ms / 1000.0
        if self._offset is None or est < self._offset:
            self._offset = est
        else:
            self._offset += 0.001 * (est - self._offset)  # follow slow drift
        return t_ms / 1000.0 + self._offset

    def _hb_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.send(J2E_HEARTBEAT, struct.pack("<IB", int(clock.now() * 1000) & 0xFFFFFFFF, self.jetson_state))
            except Exception as exc:
                log.debug("esp32 heartbeat failed: %s", exc)
            self._stop.wait(0.1)   # 10 Hz; ESP32 watchdog trips after 500 ms of silence

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self.ser.read(256)
            except Exception as exc:
                if self._stop.is_set():
                    break
                log.warning("esp32 read error: %s", exc)
                self._stop.wait(0.2)
                continue
            if not data:
                continue
            t_rx = clock.now()
            for mtype, seq, payload in self.parser.feed(data):
                self.last_rx = t_rx
                if mtype == E2J_HEARTBEAT and len(payload) >= 6:
                    self.esp_status = struct.unpack_from("<IH", payload)[1]
                elif mtype == E2J_ACK and len(payload) >= 3:
                    if payload[2] == 0:
                        self.acks += 1
                    else:
                        self.nacks += 1
                for cb in self.handlers.get(mtype, ()):
                    try:
                        cb(t_rx, seq, payload)
                    except Exception:
                        log.exception("esp32 handler failed for type 0x%02x", mtype)

    def info(self) -> dict:
        return {"port": self.port, "frames": self.parser.frames, "crc_errors": self.parser.crc_errors,
                "status_bits": self.esp_status, "rx_age_s": round(clock.now() - self.last_rx, 3) if self.last_rx else None,
                "acks": self.acks, "nacks": self.nacks}


class Esp32BridgeGimbal(GimbalBase):
    kind = "esp32"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.link = Esp32Link.get(str(self.cfg.get("port", "/dev/ttyACM0")), int(self.cfg.get("baud", 921600)))
        self.link.on(E2J_GIMBAL, self._on_gimbal)

    def _on_gimbal(self, t_rx: float, seq: int, payload: bytes) -> None:
        if len(payload) < 17:
            return
        t_ms, pan_c, tilt_c, _roll_c, flags = struct.unpack_from("<IiiiB", payload)
        t = self.link.esp_to_host(t_ms, t_rx)
        self._on_feedback(t, pan_c / 100.0 * 3.141592653589793 / 180.0, tilt_c / 100.0 * 3.141592653589793 / 180.0,
                          motors_on=bool(flags & 1), mode="esp32")

    def start(self) -> None:
        self.link.start()
        self._set_motors(True)

    def stop(self) -> None:
        self.link.stop()

    @staticmethod
    def _cdeg(rad: float) -> int:
        return int(round(rad * 18000.0 / 3.141592653589793))

    def _send_angles(self, pan: float, tilt: float) -> None:
        self.link.send(J2E_ANGLE, struct.pack("<ii", self._cdeg(pan), self._cdeg(tilt)))

    def _send_rates(self, pan_rate: float, tilt_rate: float) -> None:
        self.link.send(J2E_RATE, struct.pack("<ii", self._cdeg(pan_rate), self._cdeg(tilt_rate)))

    def _set_motors(self, on: bool) -> None:
        self.link.send(J2E_MOTORS, struct.pack("<B", 1 if on else 0))

    def estop(self) -> None:
        try:
            self.link.send(J2E_ESTOP)
        finally:
            super().estop()

    def reset_estop(self) -> None:
        self.link.send(J2E_CLEAR_ESTOP)
        super().reset_estop()

    def set_jetson_state(self, state: str) -> None:
        self.link.jetson_state = STATE_CODES.get(state, 0)

    def info(self) -> dict:
        d = super().info()
        d["link"] = self.link.info()
        return d
