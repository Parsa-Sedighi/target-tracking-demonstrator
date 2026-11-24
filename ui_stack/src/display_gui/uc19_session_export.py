"""UC-19: Export a recorded session in open formats (ES-09.1).

A session exports to a plain folder anyone can open without AimBot:

    session_export/
        manifest.json      field list, units, frame size, export time
        records.csv        one row per frame, every ES-09.1 field
        frames/            lossless 8-bit PNG per frame, named by frame index
                           and timestamp, with the timestamp also in a tEXt chunk

The PNG writer uses only the standard library (zlib), so exporting needs no
image package, and any image viewer, Python, MATLAB or Excel can read the result.
"""
from __future__ import annotations

import csv
import json
import os
import struct
import zlib
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    """One ES-09.1 record: everything logged for a single camera frame."""
    timestamp_ns: int
    pan_measured_deg: float
    tilt_measured_deg: float
    pan_commanded_deg: float
    tilt_commanded_deg: float
    target_id: int | None
    target_class: str
    confidence: float
    centroid_x_px: float
    centroid_y_px: float
    bbox_x_px: float
    bbox_y_px: float
    bbox_w_px: float
    bbox_h_px: float
    tracking_error_deg: float
    track_state: str
    range_m: float | None
    active_faults: str


ES_09_1_FIELDS = [f.name for f in fields(FrameRecord)]
UNITS = {
    "timestamp_ns": "nanoseconds since the Unix epoch (camera hardware clock)",
    "pan_measured_deg": "deg", "tilt_measured_deg": "deg",
    "pan_commanded_deg": "deg", "tilt_commanded_deg": "deg",
    "confidence": "0-1", "centroid_x_px": "px", "centroid_y_px": "px",
    "bbox_x_px": "px", "bbox_y_px": "px", "bbox_w_px": "px", "bbox_h_px": "px",
    "tracking_error_deg": "deg", "range_m": "m (blank if no LiDAR range)",
    "active_faults": "semicolon-separated fault codes, blank if none",
}


def export_session(frames_and_records: Iterable[tuple[np.ndarray, FrameRecord]],
                   out_dir: str | os.PathLike) -> Path:
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frame_shape = None
    count = 0
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["frame_index", "frame_file", *ES_09_1_FIELDS])
        writer.writeheader()
        for index, (frame, record) in enumerate(frames_and_records):
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frame_shape = frame_shape or list(frame.shape)
            name = f"frame_{index:06d}_{record.timestamp_ns}.png"
            write_png(out / "frames" / name, frame, {"timestamp_ns": str(record.timestamp_ns)})
            row = {k: ("" if v is None else v) for k, v in asdict(record).items()}
            writer.writerow({"frame_index": index, "frame_file": f"frames/{name}", **row})
            count += 1
    (out / "manifest.json").write_text(json.dumps({
        "format": "aimbot-session-export/1",
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": count,
        "frame_shape": frame_shape,
        "frame_format": "PNG, 8-bit, lossless",
        "fields": ES_09_1_FIELDS,
        "units": UNITS,
    }, indent=2))
    return out


def write_png(path: str | os.PathLike, image: np.ndarray, text: dict[str, str] | None = None) -> None:
    """Write an 8-bit grayscale or RGB image as a lossless PNG (standard library only)."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim == 2:
        color_type, channels = 0, 1
    elif image.ndim == 3 and image.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise ValueError("expected an HxW or HxWx3 uint8 image")
    height, width = image.shape[:2]
    rows = image.reshape(height, width * channels)
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    for key, value in (text or {}).items():
        png += chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)
