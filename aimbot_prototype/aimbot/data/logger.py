"""Asynchronous session logger (FR-07, ES-09.1, UC-6, TR-9).

One folder per session::

    logs/20261104_153012_sim/
        session.json        config snapshot, calibration, intrinsics, summary
        telemetry.csv       100 Hz control-loop rows (synchronised by t_mono and frame_id)
        frames.csv          index of every captured frame (+ which ones were saved)
        frames.bin          raw frames, lossless (mode: raw) - or frames/*.png (mode: png)
        detections.csv      every macro (YOLO) detection with track id
        events.jsonl        state changes, operator commands, faults, OCR, calibration

Writers run in their own threads behind queues; callers never block on disk
(PRD story 5: 1000 entries dispatched in < 50 ms). Telemetry queues are unbounded
(zero dropouts); the frame queue is bounded and drops are *counted* and logged.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .. import clock

log = logging.getLogger(__name__)

TELEMETRY_FIELDS = [
    "t_mono", "t_wall", "state", "frame_id", "frame_t_capture", "target_id", "target_class", "target_conf",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "u", "v", "dx", "dy", "err_az_deg", "err_el_deg",
    "kf_az_deg", "kf_el_deg", "kf_az_rate_dps", "kf_el_rate_dps",
    "pan_deg", "tilt_deg", "pan_rate_dps", "tilt_rate_dps", "cmd_mode", "cmd_pan", "cmd_tilt",
    "range_m", "range_valid", "range_on_target", "target_x_m", "target_y_m", "target_z_m",
    "latency_ms", "loop_dt_ms", "faults", "true_err_az_deg", "true_err_el_deg", "true_in_fov",
]
FRAME_FIELDS = ["frame_id", "t_capture", "t_host", "hw_timestamp_ns", "sensor", "width", "height", "saved",
                "file", "offset", "nbytes", "dtype", "channels"]
DET_FIELDS = ["frame_id", "t_capture", "t_done", "infer_ms", "track_id", "cls", "cls_raw", "conf",
              "x1", "y1", "x2", "y2", "roi"]


class SessionLogger:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.root = Path(os.path.expanduser(str(self.cfg.get("dir", "logs"))))
        self.frame_mode = str(self.cfg.get("frames", {}).get("mode", "png")).lower()
        self.frame_every = max(1, int(self.cfg.get("frames", {}).get("every_n", 6)))
        self.frame_qmax = int(self.cfg.get("frames", {}).get("max_queue", 48))
        self.min_free_gb = float(self.cfg.get("min_free_gb", 2.0))
        self.warn_free_gb = float(self.cfg.get("warn_free_gb", 10.0))
        self.active = False
        self.dir: Optional[Path] = None
        self._tq: queue.Queue = queue.Queue()
        self._fq: queue.Queue = queue.Queue(maxsize=self.frame_qmax)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.rows = 0
        self.frames_seen = 0
        self.frames_saved = 0
        self.frames_dropped = 0
        self.detections = 0
        self.events = 0
        self.free_gb = None
        self.disk_warning = ""
        self.frames_paused_for_space = False
        self.t_start = 0.0
        self.write_errors = 0
        self._bin_offset = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start_session(self, label: str, meta: dict) -> Path:
        if self.active:
            self.stop_session()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.root / f"{stamp}_{label}"
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.frame_mode == "png":
            (self.dir / "frames").mkdir(exist_ok=True)
        self.rows = self.frames_seen = self.frames_saved = self.frames_dropped = self.detections = self.events = 0
        self.write_errors = 0
        self._bin_offset = 0
        self.t_start = clock.now()
        self._meta = dict(meta)
        self._meta.update({"started_wall": datetime.now().isoformat(timespec="milliseconds"),
                           "t_mono_at_start": self.t_start, "frame_mode": self.frame_mode,
                           "frame_every_n": self.frame_every})
        self._write_json("session.json", self._meta)
        self._stop.clear()
        self._tq = queue.Queue()
        self._fq = queue.Queue(maxsize=self.frame_qmax)
        self._threads = [threading.Thread(target=self._telemetry_writer, name="log-telemetry", daemon=True),
                         threading.Thread(target=self._frame_writer, name="log-frames", daemon=True)]
        for t in self._threads:
            t.start()
        self.active = True
        log.info("logging session -> %s", self.dir)
        return self.dir

    def stop_session(self, summary: Optional[dict] = None) -> Optional[Path]:
        if not self.active:
            return None
        self.active = False
        # frame writer first (it still feeds index rows to the telemetry writer), then telemetry
        try:
            self._fq.put(None, timeout=15.0)
        except queue.Full:
            pass
        self._threads[1].join(timeout=30.0)
        self._tq.put(None)
        self._threads[0].join(timeout=15.0)
        meta = dict(self._meta)
        meta.update({"stopped_wall": datetime.now().isoformat(timespec="milliseconds"),
                     "duration_s": round(clock.now() - self.t_start, 3), "telemetry_rows": self.rows,
                     "frames_seen": self.frames_seen, "frames_saved": self.frames_saved,
                     "frames_dropped_by_logger": self.frames_dropped, "detections": self.detections,
                     "events": self.events, "summary": summary or {}})
        self._write_json("session.json", meta)
        log.info("logging stopped: %d rows, %d frames saved (%d dropped)", self.rows, self.frames_saved, self.frames_dropped)
        d = self.dir
        return d

    def _write_json(self, name: str, data: dict) -> None:
        assert self.dir is not None
        with open(self.dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)

    # ------------------------------------------------------------------ producers (never block)
    def log_telemetry(self, row: dict) -> None:
        if self.active:
            self._tq.put(("t", row))

    def log_detections(self, res) -> None:
        if not self.active:
            return
        for d in res.detections:
            self._tq.put(("d", {"frame_id": res.frame_id, "t_capture": f"{res.t_capture:.6f}", "t_done": f"{res.t_done:.6f}",
                                "infer_ms": f"{res.infer_ms:.2f}", "track_id": d.track_id, "cls": d.cls, "cls_raw": d.cls_raw,
                                "conf": f"{d.conf:.3f}", "x1": f"{d.bbox[0]:.1f}", "y1": f"{d.bbox[1]:.1f}",
                                "x2": f"{d.bbox[2]:.1f}", "y2": f"{d.bbox[3]:.1f}",
                                "roi": "" if res.roi is None else "%d %d %d %d" % tuple(int(v) for v in res.roi)}))

    def log_event(self, kind: str, data: Optional[dict] = None) -> None:
        if self.active:
            self._tq.put(("e", {"t_mono": clock.now(), "t_wall": time.time(), "kind": kind, **(data or {})}))

    def log_frame(self, frame) -> None:
        """Called once per processed frame (vision thread). Index rows are written in frame order."""
        if not self.active:
            return
        self.frames_seen += 1
        save = (self.frame_mode != "none" and not self.frames_paused_for_space
                and (frame.frame_id % self.frame_every == 0))
        extra: dict = {}
        if save:
            img = frame.raw if (frame.raw is not None and self.cfg.get("frames", {}).get("prefer_raw_sensor", True)) \
                else frame.image
            ch = 1 if img.ndim == 2 else img.shape[2]
            if self.frame_mode == "raw":
                extra = dict(file="frames.bin", offset=self._bin_offset, nbytes=int(img.nbytes), dtype=str(img.dtype),
                             channels=ch, width=img.shape[1], height=img.shape[0])
            else:
                extra = dict(file=f"frames/{frame.frame_id:08d}.png", offset=-1, nbytes=0, dtype=str(img.dtype),
                             channels=ch, width=img.shape[1], height=img.shape[0])
            try:
                self._fq.put_nowait((img, extra["file"]))
                if self.frame_mode == "raw":
                    self._bin_offset += int(img.nbytes)
            except queue.Full:
                self.frames_dropped += 1
                save, extra = False, {}
        self._tq.put(("f", (frame, save, extra)))

    # ------------------------------------------------------------------ writers
    def _telemetry_writer(self) -> None:
        assert self.dir is not None
        d = self.dir
        with open(d / "telemetry.csv", "w", newline="", encoding="utf-8") as ft, \
                open(d / "detections.csv", "w", newline="", encoding="utf-8") as fd, \
                open(d / "events.jsonl", "w", encoding="utf-8") as fe, \
                open(d / "frames.csv", "w", newline="", encoding="utf-8") as ff:
            wt = csv.DictWriter(ft, TELEMETRY_FIELDS, extrasaction="ignore")
            wd = csv.DictWriter(fd, DET_FIELDS, extrasaction="ignore")
            wf = csv.DictWriter(ff, FRAME_FIELDS, extrasaction="ignore")
            wt.writeheader()
            wd.writeheader()
            wf.writeheader()
            self._wf = wf
            self._ff = ff
            last_flush = clock.now()
            last_disk = 0.0
            while True:
                try:
                    item = self._tq.get(timeout=0.25)
                except queue.Empty:
                    item = ()
                if item is None:
                    break
                if item:
                    kind, data = item
                    if kind == "t":
                        wt.writerow(data)
                        self.rows += 1
                    elif kind == "d":
                        wd.writerow(data)
                        self.detections += 1
                    elif kind == "e":
                        fe.write(json.dumps(data, default=_json_default) + "\n")
                        self.events += 1
                    elif kind == "f":
                        self._index_frame(*data)
                now = clock.now()
                if now - last_flush > 1.0:
                    ft.flush(); fd.flush(); fe.flush(); ff.flush()
                    last_flush = now
                if now - last_disk > 2.0:
                    self._check_disk()
                    last_disk = now
            # drain whatever is left
            while True:
                try:
                    item = self._tq.get_nowait()
                except queue.Empty:
                    break
                if item and item[0] == "t":
                    wt.writerow(item[1]); self.rows += 1
                elif item and item[0] == "f":
                    self._index_frame(*item[1])
                elif item and item[0] == "d":
                    wd.writerow(item[1]); self.detections += 1
                elif item and item[0] == "e":
                    fe.write(json.dumps(item[1], default=_json_default) + "\n"); self.events += 1

    def _index_frame(self, frame, saved: bool, extra: dict) -> None:
        row = {"frame_id": frame.frame_id, "t_capture": f"{frame.t_capture:.6f}", "t_host": f"{frame.t_host:.6f}",
               "hw_timestamp_ns": frame.hw_timestamp_ns or "", "sensor": frame.sensor, "width": frame.width,
               "height": frame.height, "saved": int(saved), "file": "", "offset": -1, "nbytes": 0, "dtype": "",
               "channels": 0}
        row.update(extra or {})
        self._wf.writerow(row)

    def _frame_writer(self) -> None:
        assert self.dir is not None
        binf = open(self.dir / "frames.bin", "wb") if self.frame_mode == "raw" else None
        try:
            while True:
                item = self._fq.get()
                if item is None:
                    break
                img, name = item
                try:
                    if binf is not None:
                        binf.write(np.ascontiguousarray(img).tobytes())
                    else:
                        cv2.imwrite(str(self.dir / name), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    self.frames_saved += 1
                except Exception as exc:
                    self.write_errors += 1
                    log.warning("frame write failed: %s", exc)
        finally:
            if binf is not None:
                binf.close()

    def _check_disk(self) -> None:
        try:
            usage = shutil.disk_usage(self.dir if self.dir else self.root)
        except Exception:
            return
        self.free_gb = usage.free / 1e9
        if self.free_gb < self.min_free_gb:
            if not self.frames_paused_for_space:
                log.warning("disk nearly full (%.1f GB free): frame logging paused, telemetry continues", self.free_gb)
            self.frames_paused_for_space = True
            self.disk_warning = f"DISK LOW {self.free_gb:.1f} GB - frames paused"
        elif self.free_gb < self.warn_free_gb:
            self.disk_warning = f"disk {self.free_gb:.1f} GB free"
            self.frames_paused_for_space = False
        else:
            self.disk_warning = ""
            self.frames_paused_for_space = False

    def info(self) -> dict:
        if self.free_gb is None:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                self.free_gb = shutil.disk_usage(self.root).free / 1e9
            except Exception:
                pass
        return {"active": self.active, "dir": str(self.dir) if self.dir else "", "rows": self.rows,
                "frames_saved": self.frames_saved, "frames_dropped": self.frames_dropped,
                "frame_queue": self._fq.qsize(), "free_gb": round(self.free_gb, 1) if self.free_gb else None,
                "warning": self.disk_warning, "mode": self.frame_mode, "every_n": self.frame_every}


def _json_default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
