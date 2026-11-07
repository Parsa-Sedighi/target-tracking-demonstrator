"""Operator Gateway (PRD architecture: Operator Stack <-> Operator Workstation).

The engine runs headless on the Jetson and serves status + video over TCP; the
PyQt6 GUI on the operator laptop connects with ``python -m aimbot --connect <ip>``.

Wire format (one TCP connection, both directions):  type(1 byte) | length(uint32 BE) | payload
    'S'  server -> client  status snapshot, JSON (status_hz, default 20 Hz)
    'V'  server -> client  video: frame_id (uint64 BE) + JPEG (video_hz, default 30 Hz, downscaled)
    'C'  client -> server  command, JSON  {"cmd": "start"} ...  (same commands as Engine.command)
    'R'  server -> client  command reply, JSON
Overlay coordinates in the status are normalised (0..1), so they match any video size.
No authentication: keep it on the lab network.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

HDR = struct.Struct(">cI")


def _send(sock: socket.socket, kind: bytes, payload: bytes, lock: threading.Lock) -> None:
    with lock:
        sock.sendall(HDR.pack(kind, len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock: socket.socket):
    h = _recv_exact(sock, HDR.size)
    if h is None:
        return None, None
    kind, n = HDR.unpack(h)
    if n > 50_000_000:
        raise ValueError("message too large")
    payload = _recv_exact(sock, n) if n else b""
    return kind, payload


class _Client:
    def __init__(self, server: "GatewayServer", sock: socket.socket, addr):
        self.server, self.sock, self.addr = server, sock, addr
        self.lock = threading.Lock()
        self.alive = True
        threading.Thread(target=self._reader, name=f"gw-rx-{addr}", daemon=True).start()
        threading.Thread(target=self._writer, name=f"gw-tx-{addr}", daemon=True).start()

    def _reader(self) -> None:
        try:
            while self.alive:
                kind, payload = _recv_msg(self.sock)
                if kind is None:
                    break
                if kind == b"C":
                    try:
                        cmd = json.loads(payload.decode("utf-8"))
                        res = self.server.engine.command(cmd)
                    except Exception as exc:
                        res = {"ok": False, "msg": str(exc)}
                    _send(self.sock, b"R", json.dumps(res).encode(), self.lock)
        except Exception as exc:
            log.info("gateway client %s reader ended: %s", self.addr, exc)
        self.alive = False

    def _writer(self) -> None:
        cfg = self.server.cfg
        status_dt = 1.0 / float(cfg.get("status_hz", 20))
        video_dt = 1.0 / float(cfg.get("video_hz", 30))
        q = int(cfg.get("jpeg_quality", 70))
        max_w = int(cfg.get("max_width", 960))
        t_status = t_video = 0.0
        last_fid = -1
        try:
            while self.alive:
                now = time.perf_counter()
                if now - t_video >= video_dt:
                    t_video = now
                    fr = self.server.engine.framebuf.latest()
                    if fr is not None and fr.frame_id != last_fid:
                        last_fid = fr.frame_id
                        img = fr.image
                        if img.shape[1] > max_w:
                            s = max_w / img.shape[1]
                            img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
                        if ok:
                            _send(self.sock, b"V", struct.pack(">Q", fr.frame_id) + jpg.tobytes(), self.lock)
                if now - t_status >= status_dt:
                    t_status = now
                    snap = self.server.engine.snapshot(with_frame=False)
                    snap.pop("frame", None)
                    snap["mode"] = "remote"
                    _send(self.sock, b"S", json.dumps(snap, default=_json_default).encode(), self.lock)
                time.sleep(0.004)
        except Exception as exc:
            log.info("gateway client %s writer ended: %s", self.addr, exc)
        self.alive = False
        try:
            self.sock.close()
        except Exception:
            pass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


class GatewayServer:
    def __init__(self, engine, cfg: Optional[dict] = None):
        self.engine = engine
        self.cfg = cfg or {}
        self.host = str(self.cfg.get("host", "0.0.0.0"))
        self.port = int(self.cfg.get("port", 5600))
        self._sock: Optional[socket.socket] = None
        self.clients: list[_Client] = []
        self._stop = threading.Event()

    def start(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(4)
        s.settimeout(0.5)
        self._sock = s
        self.port = s.getsockname()[1]
        threading.Thread(target=self._accept, name="gateway", daemon=True).start()
        log.info("operator gateway on %s:%d", self.host, self.port)

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                c, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c.settimeout(None)
            self.clients = [x for x in self.clients if x.alive] + [_Client(self, c, addr)]
            self.engine._event("gateway", f"operator connected from {addr[0]}")

    def stop(self) -> None:
        self._stop.set()
        for c in self.clients:
            c.alive = False
            try:
                c.sock.close()
            except Exception:
                pass
        if self._sock:
            self._sock.close()


class RemoteEngine:
    """GUI-side proxy with the same snapshot()/command() API as the local Engine."""

    def __init__(self, host: str, port: int = 5600):
        self.host, self.port = host, port
        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()
        self.connected = False
        self._status: dict = {}
        self._frame: Optional[np.ndarray] = None
        self._frame_id = -1
        self._stop = threading.Event()
        self.last_reply: dict = {}
        self.rx_video_fps = 0.0

    def start(self) -> None:
        threading.Thread(target=self._run, name="remote-engine", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

    def shutdown(self) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.host, self.port), timeout=3.0)
                s.settimeout(None)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock, self.connected = s, True
                n, t0 = 0, time.perf_counter()
                while not self._stop.is_set():
                    kind, payload = _recv_msg(s)
                    if kind is None:
                        break
                    if kind == b"S":
                        self._status = json.loads(payload.decode("utf-8"))
                    elif kind == b"V":
                        fid = struct.unpack(">Q", payload[:8])[0]
                        img = cv2.imdecode(np.frombuffer(payload[8:], np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            self._frame, self._frame_id = img, int(fid)
                            n += 1
                            if time.perf_counter() - t0 > 1.0:
                                self.rx_video_fps, n, t0 = n / (time.perf_counter() - t0), 0, time.perf_counter()
                    elif kind == b"R":
                        self.last_reply = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                log.info("gateway connection: %s", exc)
            self.connected = False
            time.sleep(1.0)

    def command(self, c: dict) -> dict:
        if not self.connected or self.sock is None:
            return {"ok": False, "msg": f"not connected to {self.host}:{self.port}"}
        try:
            _send(self.sock, b"C", json.dumps(c).encode(), self.lock)
            return {"ok": True, "msg": ""}
        except Exception as exc:
            return {"ok": False, "msg": str(exc)}

    def snapshot(self, with_frame: bool = True) -> dict:
        s = dict(self._status) if self._status else {"state": {"state": "CONNECTING", "reason": f"{self.host}:{self.port}"}}
        if not self.connected:
            s["state"] = {"state": "DISCONNECTED", "reason": f"connecting to {self.host}:{self.port} ..."}
        s["frame"] = self._frame if with_frame else None
        s["frame_id"] = self._frame_id
        s["mode"] = f"remote {self.host}:{self.port} ({self.rx_video_fps:.0f} fps video)"
        return s
