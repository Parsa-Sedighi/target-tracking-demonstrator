"""Gimbal backends: sim | none | gremsy (MAVLink direct, e.g. via FT232) | esp32 (via the supervisory PCB)."""

from __future__ import annotations

from .base import FixedGimbal, GimbalBase


def create_gimbal(cfg: dict) -> GimbalBase:
    backend = str((cfg or {}).get("backend", "sim")).lower()
    if backend == "sim":
        from .sim import SimGimbal

        return SimGimbal(cfg)
    if backend in ("none", "fixed", "dryrun"):
        return FixedGimbal(cfg)
    if backend in ("gremsy", "mavlink", "mio"):
        from .mavlink_gremsy import MavlinkGremsyGimbal

        return MavlinkGremsyGimbal(cfg)
    if backend == "esp32":
        from .esp32_bridge import Esp32BridgeGimbal

        return Esp32BridgeGimbal(cfg)
    raise ValueError(f"unknown gimbal backend {backend!r}")
