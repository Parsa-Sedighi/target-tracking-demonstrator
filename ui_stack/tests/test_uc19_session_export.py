"""Acceptance test for UC-19: review a recorded session outside AimBot (ES-09.1).

UC-19 acceptance test (PRD v2, section 7):
    On a computer without AimBot installed, a session's records and frames
    open with common tools, with every ES-09.1 field readable and timestamps
    matching frames.

AimBot code is used only to create the export. Everything that reads it back
below uses the Python standard library alone (csv, json, zlib, struct), the
way an analyst without AimBot would.
"""
import csv
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from aimbot.uc19_session_export import FrameRecord, export_session

ES_09_1 = [
    "timestamp_ns", "pan_measured_deg", "tilt_measured_deg", "pan_commanded_deg",
    "tilt_commanded_deg", "target_id", "target_class", "confidence", "centroid_x_px",
    "centroid_y_px", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
    "tracking_error_deg", "track_state", "range_m", "active_faults",
]


def _session(n=50):
    rng = np.random.default_rng(19)
    for i in range(n):
        frame = rng.integers(0, 256, (48, 64), dtype=np.uint8)
        yield frame, FrameRecord(
            timestamp_ns=1_795_000_000_000_000_000 + i * 7_692_308,
            pan_measured_deg=0.1 * i, tilt_measured_deg=-0.05 * i,
            pan_commanded_deg=0.1 * i + 0.02, tilt_commanded_deg=-0.05 * i,
            target_id=3 if i > 4 else None, target_class="drone" if i > 4 else "",
            confidence=0.9, centroid_x_px=32.0, centroid_y_px=24.0,
            bbox_x_px=20.0, bbox_y_px=12.0, bbox_w_px=24.0, bbox_h_px=24.0,
            tracking_error_deg=0.12, track_state="TRACKING" if i > 4 else "SEARCHING",
            range_m=3.05 if i % 3 else None, active_faults="" if i < 40 else "STORAGE_LOW",
        )


def _read_png_stdlib(path: Path):
    """Minimal PNG reader using only the standard library (8-bit, filter 0)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, text = 8, b"", {}
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
        elif kind == b"IDAT":
            idat += body
        elif kind == b"tEXt":
            key, value = body.split(b"\x00", 1)
            text[key.decode()] = value.decode()
        pos += 12 + length
    channels = 1 if color == 0 else 3
    raw = zlib.decompress(idat)
    stride = width * channels + 1
    rows = [raw[y * stride + 1:(y + 1) * stride] for y in range(height)]
    return width, height, rows, text


def test_session_opens_with_standard_tools_and_every_field_is_readable(tmp_path):
    originals = list(_session())
    out = export_session(iter(originals), tmp_path / "export")

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["frame_count"] == len(originals)
    assert manifest["fields"] == ES_09_1

    with open(out / "records.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(originals)
    assert set(ES_09_1) <= set(rows[0].keys())

    for row, (frame, record) in zip(rows, originals):
        assert int(row["timestamp_ns"]) == record.timestamp_ns
        assert float(row["pan_measured_deg"]) == record.pan_measured_deg
        assert row["track_state"] == record.track_state
        assert row["range_m"] == ("" if record.range_m is None else str(record.range_m))

        frame_path = out / row["frame_file"]
        assert row["timestamp_ns"] in frame_path.name
        width, height, pixel_rows, text = _read_png_stdlib(frame_path)
        assert (width, height) == (frame.shape[1], frame.shape[0])
        assert text["timestamp_ns"] == row["timestamp_ns"]
        assert b"".join(pixel_rows) == frame.tobytes(), "frame was not stored losslessly"
    print(f"UC-19 metric: {len(rows)} frames exported, all {len(ES_09_1)} ES-09.1 fields readable")


def test_every_frame_file_listed_in_the_csv_exists(tmp_path):
    out = export_session(_session(10), tmp_path / "export")
    with open(out / "records.csv", newline="") as f:
        listed = {row["frame_file"] for row in csv.DictReader(f)}
    on_disk = {f"frames/{p.name}" for p in (out / "frames").glob("*.png")}
    assert listed == on_disk
