"""Benewake TF02-Pro LiDAR: byte-stream parser + serial / simulated / bridged sources.

TF02-Pro standard 9-byte frame (UART, 115200 8N1, LVTTL 3.3 V, 100 Hz default):

    0x59 0x59 Dist_L Dist_H Strength_L Strength_H Temp_L Temp_H Checksum

* Dist in cm (default output), little-endian.
* Checksum = low 8 bits of the sum of bytes 0..7.
* Temp (deg C) = Temp/8 - 256.
* Strength < 60  -> Dist forced to 4500 (unreliable).  Strength = 65535 -> saturated, Dist = 0.
* Beam (detection angle) = 3 deg.

The same parser is used for the simulator (which generates real byte frames,
including occasional corrupted bytes) so the checksum/resync logic is
exercised continuously - this is the AIM-113/114 "zero-drop at 100 Hz with
checksum rejection" item.
"""

from __future__ import annotations

import logging
import random
import threading
from collections import deque
from typing import Callable, Optional

from .. import clock
from ..types import RangeReading

log = logging.getLogger(__name__)

HEADER = 0x59
FRAME_LEN = 9
UNRELIABLE_CM = 4500
MIN_STRENGTH = 60


def encode_frame(dist_cm: int, strength: int, temp_c: float = 35.0) -> bytes:
    dist_cm = max(0, min(65535, int(dist_cm)))
    strength = max(0, min(65535, int(strength)))
    temp_raw = max(0, min(65535, int(round((temp_c + 256.0) * 8))))
    b = bytearray([HEADER, HEADER, dist_cm & 0xFF, dist_cm >> 8, strength & 0xFF, strength >> 8,
                   temp_raw & 0xFF, temp_raw >> 8])
    b.append(sum(b) & 0xFF)
    return bytes(b)


class TF02Parser:
    """Incremental, allocation-light parser. feed() returns decoded (dist_cm, strength, temp_raw) tuples."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.frames_ok = 0
        self.checksum_errors = 0
        self.resync_bytes = 0
        self.bytes_in = 0

    def feed(self, data: bytes) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        if not data:
            return out
        self.bytes_in += len(data)
        buf = self._buf
        buf.extend(data)
        i = 0
        n = len(buf)
        while n - i >= FRAME_LEN:
            if buf[i] != HEADER or buf[i + 1] != HEADER:
                i += 1
                self.resync_bytes += 1
                continue
            if (sum(buf[i:i + 8]) & 0xFF) != buf[i + 8]:
                self.checksum_errors += 1
                i += 1          # slide one byte and look for the next header
                continue
            dist = buf[i + 2] | (buf[i + 3] << 8)
            strength = buf[i + 4] | (buf[i + 5] << 8)
            temp = buf[i + 6] | (buf[i + 7] << 8)
            out.append((dist, strength, temp))
            self.frames_ok += 1
            i += FRAME_LEN
        del buf[:i]
        if len(buf) > 4096:  # never grow without bound on garbage input
            del buf[:-FRAME_LEN]
        return out

    def stats(self) -> dict:
        return {"frames_ok": self.frames_ok, "checksum_errors": self.checksum_errors,
                "resync_bytes": self.resync_bytes, "bytes_in": self.bytes_in}


def decode_reading(t: float, dist_cm: int, strength: int, temp_raw: int, range_offset_m: float = 0.0) -> RangeReading:
    temp_c = temp_raw / 8.0 - 256.0
    valid = MIN_STRENGTH <= strength < 65535 and 0 < dist_cm < UNRELIABLE_CM
    return RangeReading(t=t, distance_m=dist_cm / 100.0 + (range_offset_m if valid else 0.0), strength=strength,
                        temp_c=temp_c, valid=valid, raw_cm=dist_cm)


class LidarBase:
    kind = "none"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.parser = TF02Parser()
        self.range_offset_m = float(self.cfg.get("range_offset_m", 0.0))
        self._latest: Optional[RangeReading] = None
        self._hist: deque[RangeReading] = deque(maxlen=512)
        self._lock = threading.Lock()
        self.rate = clock.RateMeter()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.listeners: list[Callable[[RangeReading], None]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _publish(self, rd: RangeReading) -> None:
        with self._lock:
            self._latest = rd
            self._hist.append(rd)
        self.rate.tick()
        for cb in self.listeners:
            try:
                cb(rd)
            except Exception:
                log.exception("lidar listener failed")

    def _handle_bytes(self, data: bytes, t: float) -> None:
        for dist, strength, temp in self.parser.feed(data):
            self._publish(decode_reading(t, dist, strength, temp, self.range_offset_m))

    def latest(self) -> Optional[RangeReading]:
        return self._latest

    def mean_over(self, seconds: float) -> tuple[Optional[float], int]:
        t0 = clock.now() - seconds
        with self._lock:
            vals = [r.distance_m for r in self._hist if r.t >= t0 and r.valid]
        if not vals:
            return None, 0
        return sum(vals) / len(vals), len(vals)

    def info(self) -> dict:
        r = self._latest
        d = {"kind": self.kind, "rate_hz": round(self.rate.value(), 1), "range_offset_m": self.range_offset_m}
        d.update(self.parser.stats())
        if r is not None:
            d.update({"distance_m": round(r.distance_m, 3), "strength": r.strength, "valid": r.valid,
                      "temp_c": round(r.temp_c, 1), "age_s": round(clock.now() - r.t, 3)})
        return d


class NoLidar(LidarBase):
    kind = "none"


class SerialLidar(LidarBase):
    """TF02-Pro straight on a USB-UART adapter (bench) - /dev/ttyUSBx or COMx."""

    kind = "serial"

    def start(self) -> None:
        import serial  # pyserial

        port = self.cfg.get("port", "/dev/ttyUSB1")
        baud = int(self.cfg.get("baud", 115200))
        self.ser = serial.serial_for_url(port, baudrate=baud, timeout=0.02)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lidar-serial", daemon=True)
        self._thread.start()
        log.info("TF02-Pro on %s @ %d baud", port, baud)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self.ser.read(64)
            except Exception as exc:
                log.warning("lidar serial error: %s", exc)
                clock.sleep_until(clock.now() + 0.2)
                continue
            if data:
                self._handle_bytes(data, clock.now())

    def stop(self) -> None:
        super().stop()
        try:
            self.ser.close()
        except Exception:
            pass


class SimLidar(LidarBase):
    """Generates genuine TF02-Pro byte frames from the simulator (and corrupts a few)."""

    kind = "sim"

    def __init__(self, cfg: dict, world, true_angles: Callable[[float], tuple]):
        super().__init__(cfg)
        self.world = world
        self.true_angles = true_angles
        self.hz = float(self.cfg.get("rate_hz", 100))
        self.noise_cm = float(self.cfg.get("noise_cm", 1.5))
        self.bias_cm = float(self.cfg.get("bias_cm", 2.5))  # deliberate bias so calibration has work to do
        self.corrupt_prob = float(self.cfg.get("corrupt_prob", 0.01))
        self.rng = random.Random(3)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lidar-sim", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        loop = clock.RateLoop(self.hz, self._stop)
        while not self._stop.is_set():
            t = loop.wait()
            pan, tilt = self.true_angles(t)
            d = self.world.range_along(t, pan, tilt)
            dist_cm = int(round(d * 100 + self.bias_cm + self.rng.gauss(0, self.noise_cm)))
            strength = 900 if d < 12 else 120
            frame = bytearray(encode_frame(dist_cm, strength, 38.0))
            if self.rng.random() < self.corrupt_prob:
                frame[self.rng.randrange(2, 8)] ^= 0x5A   # corrupt payload -> checksum must reject it
            if self.rng.random() < self.corrupt_prob:
                frame = bytearray(b"\x00\x13") + frame      # stray bytes -> resync
            self._handle_bytes(bytes(frame), t)


class ExternalLidar(LidarBase):
    """Readings delivered by another component (the ESP32 bridge)."""

    kind = "esp32"

    def push(self, t: float, dist_cm: int, strength: int, temp_raw: int) -> None:
        self.parser.frames_ok += 1
        self._publish(decode_reading(t, dist_cm, strength, temp_raw, self.range_offset_m))


def create_lidar(cfg: dict, world=None, true_angles=None) -> LidarBase:
    backend = str((cfg or {}).get("backend", "none")).lower()
    if backend == "sim":
        return SimLidar(cfg, world, true_angles)
    if backend in ("serial", "tf02", "tf02pro", "uart"):
        return SerialLidar(cfg)
    if backend == "esp32":
        return ExternalLidar(cfg)
    return NoLidar(cfg)
