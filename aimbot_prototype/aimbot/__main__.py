"""Command-line entry point.

    python -m aimbot                                  # simulator + operator GUI
    python -m aimbot -c webcam                        # your webcam + YOLO, no gimbal
    python -m aimbot -c jetson_hw                     # Jetson with the real hardware + GUI on its monitor
    python -m aimbot -c jetson_headless --headless    # Jetson engine only (operator gateway on :5600)
    python -m aimbot --connect 192.168.1.50           # GUI on the laptop, driving the Jetson
    python -m aimbot --headless --auto-start --duration 30 --scenario ball_drop   # scripted run + summary
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


def run_headless(engine, duration: float = 0.0, status_every: float = 1.0) -> dict:
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    try:
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except Exception:
        pass
    t0 = time.time()
    next_status = t0
    while not stop["flag"]:
        now = time.time()
        if duration and now - t0 >= duration:
            break
        if now >= next_status:
            next_status += status_every
            s = engine.snapshot(with_frame=False)
            tel, lat, jit = s["telemetry"], s["latency"], s["jitter"]
            lk = s["overlay"].get("lock") if s["overlay"] else None
            err = ""
            if tel.get("err_az_deg"):
                err = f"err az {float(tel['err_az_deg']):+6.2f} el {float(tel['err_el_deg']):+6.2f} deg"
            p95 = f"{lat['cmd_p95']:.0f}" if lat.get("cmd_p95") is not None else "-"
            print(f"[{now - t0:6.1f}s] {s['state']['state']:<11} "
                  f"{(lk['cls'] + ' #' + str(lk['tid'])) if lk else '-':<12} {err:<34} "
                  f"pan {float(tel.get('pan_deg') or 0):+7.2f} tilt {float(tel.get('tilt_deg') or 0):+6.2f}  "
                  f"cam {s['rates']['camera']:5.1f} det {s['rates']['macro']:4.1f} ctl {s['rates']['control']:5.1f} Hz  "
                  f"lat p95 {p95} ms  jitter {jit.get('jitter_dia_in_20ft', float('nan')):.2f} in@20ft", flush=True)
        time.sleep(0.05)
    summary = engine.summary()
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aimbot", description="AimBot target tracking demonstrator")
    ap.add_argument("-c", "--config", default="default.yaml", help="config file (name in config/ or a path)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--headless", action="store_true", help="no GUI (engine + operator gateway)")
    ap.add_argument("--no-gateway", action="store_true", help="headless without the operator gateway")
    ap.add_argument("--connect", metavar="HOST[:PORT]", help="GUI only, connect to a headless engine")
    ap.add_argument("--auto-start", action="store_true", help="start tracking immediately")
    ap.add_argument("--duration", type=float, default=0.0, help="headless: stop after N seconds")
    ap.add_argument("--target", help="target class to seek (auto, any, drone, ball, card, hand, person ...)")
    ap.add_argument("--scenario", help="simulator scenario")
    ap.add_argument("--video", metavar="FILE", help="play this video file through the active camera (loops)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    _setup_logging(args.log_level)

    if args.connect:
        from .operator.gateway import RemoteEngine
        from .ui.main_window import run_gui

        host, _, port = args.connect.partition(":")
        remote = RemoteEngine(host, int(port or 5600))
        remote.start()
        return run_gui(remote)

    from .config import load_config
    from .engine import Engine

    cfg = load_config(args.config, args.set)
    if args.video:                           # a file path is not parsed as YAML (#, quotes, spaces are fine)
        video = Path(args.video).expanduser().resolve()
        if not video.is_file():
            print(f"video not found: {video}", file=sys.stderr)
            return 2
        active = str(cfg.get_path("camera.active", None) or next(iter(cfg.get_path("sensors", {}) or {}), "vis"))
        cfg.set_path(f"sensors.{active}.backend", "opencv")
        cfg.set_path(f"sensors.{active}.source", str(video))
    if args.scenario:
        cfg.set_path("sim.scenario", args.scenario)
    if args.headless and not args.no_gateway and "gateway.enabled=false" not in [s.replace(" ", "") for s in args.set]:
        cfg.set_path("gateway.enabled", True)
    if args.no_gateway:
        cfg.set_path("gateway.enabled", False)

    engine = Engine(cfg)
    engine.start()
    if args.target:
        engine.command({"cmd": "set_target_class", "cls": args.target})
    if args.auto_start:
        engine.command({"cmd": "start"})
    try:
        if args.headless:
            if engine.gateway is not None:
                print(f"operator gateway listening on port {engine.gateway.port} "
                      f"(laptop: python -m aimbot --connect <jetson-ip>)", flush=True)
            summary = run_headless(engine, args.duration)
            print("SUMMARY " + json.dumps(summary, default=str), flush=True)
            return 0
        from .ui.main_window import run_gui

        return run_gui(engine)
    finally:
        engine.shutdown()
        if engine.logger.dir:
            print(f"last log folder: {engine.logger.dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
