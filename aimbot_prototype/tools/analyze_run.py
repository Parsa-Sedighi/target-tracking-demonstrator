"""Post-test analysis of a logged session (UC-6, V-06, ES-02.x, ES-03.x, ES-05.1).

    python tools/analyze_run.py logs/20261104_153012_track_mixed           # summary + plots
    python tools/analyze_run.py logs/<session> --no-plots --json out.json

Reads only the CSV/JSON files the logger wrote, so the same numbers can be reproduced
in Excel/MATLAB/pandas by the sponsor. Prints pass/fail against the PRD/SDR thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load(session: Path):
    rows = list(csv.DictReader(open(session / "telemetry.csv", newline="", encoding="utf-8")))
    frames = list(csv.DictReader(open(session / "frames.csv", newline="", encoding="utf-8")))
    events = [json.loads(l) for l in open(session / "events.jsonl", encoding="utf-8") if l.strip()]
    meta = json.loads((session / "session.json").read_text())
    return rows, frames, events, meta


def analyze(session: Path) -> dict:
    rows, frames, events, meta = load(session)
    t = np.array([_f(r["t_mono"]) for r in rows])
    state = np.array([r["state"] for r in rows])
    out: dict = {"session": session.name, "rows": len(rows), "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0}
    # control loop timing / dropouts (FR-07, PRD story 5)
    dt = np.diff(t) * 1000.0
    if len(dt):
        period = float(np.median(dt))
        out["telemetry_rate_hz"] = float(1000.0 / np.mean(dt))
        out["telemetry_dt_ms_p99"] = float(np.percentile(dt, 99))
        out["telemetry_dt_ms_max"] = float(np.max(dt))
        # dropouts = whole rows missing from the schedule (a late row that is caught up is not lost)
        nominal = 1000.0 / float((meta.get("config", {}).get("control", {}) or {}).get("rate_hz", 1000.0 / period))
        big = dt > 2.5 * nominal
        out["telemetry_missed_ticks"] = int(np.sum(np.round(dt[big] / nominal) - 1))
    # state time budget
    out["time_in_state_s"] = {s: float(np.sum(state == s)) / max(1.0, out.get("telemetry_rate_hz", 100.0))
                              for s in sorted(set(state))}
    trk_all = state == "TRACKING"
    # "settled" = tracking, excluding the first 0.5 s after each acquisition (initial slew onto the target)
    settle = np.zeros(len(t), bool)
    last_acq = -1e9
    for i in range(len(t)):
        if trk_all[i] and (i == 0 or not trk_all[i - 1]):
            last_acq = t[i]
        settle[i] = trk_all[i] and (t[i] - last_acq) >= 0.5
    trk = settle
    out["tracking_s"] = float(np.sum(trk_all)) / 100.0
    # tracking error from vision (what the system believes)
    eaz = np.array([_f(r["err_az_deg"]) for r in rows])
    eel = np.array([_f(r["err_el_deg"]) for r in rows])
    err = np.hypot(eaz, eel)
    m = trk & np.isfinite(err)
    if m.any():
        out["meas_err_rms_deg"] = float(np.sqrt(np.mean(err[m] ** 2)))
        out["meas_err_p95_deg"] = float(np.percentile(err[m], 95))
        out["meas_err_max_deg"] = float(np.max(err[m]))
    # ground truth (simulator only)
    taz = np.array([_f(r.get("true_err_az_deg")) for r in rows])
    tel = np.array([_f(r.get("true_err_el_deg")) for r in rows])
    terr = np.hypot(taz, tel)
    mt = trk & np.isfinite(terr)
    if mt.any():
        out["true_err_rms_deg"] = float(np.sqrt(np.mean(terr[mt] ** 2)))
        out["true_err_p95_deg"] = float(np.percentile(terr[mt], 95))
        out["true_err_max_deg"] = float(np.max(terr[mt]))
    # NFR-05 pointing jitter: high-frequency part of the pointing error (error minus its 0.25 s moving
    # average, i.e. what shakes the laser spot - not the slow lag while following a manoeuvre),
    # std-dev per 1 s window, as a circle diameter (2 sigma) at 20 ft. Uses truth when available.
    src_a, src_e, src_m = (taz, tel, mt) if mt.sum() > 50 else (eaz, eel, m)
    if src_m.sum() > 50:
        k = math.radians(1.0) * 240.0
        ts, a, e = t[src_m], src_a[src_m], src_e[src_m]
        hz = out.get("telemetry_rate_hz", 100.0)
        n = max(3, int(round(0.25 * hz)))
        ker = np.ones(n) / n
        ha = a - np.convolve(a, ker, mode="same")
        he = e - np.convolve(e, ker, mode="same")
        sds, start = [], ts[0]
        while start < ts[-1]:
            w = (ts >= start) & (ts < start + 1.0)
            w[:n] = False
            w[-n:] = False
            if w.sum() > 20 and (ts[w][-1] - ts[w][0]) > 0.8:
                sds.append(math.sqrt(np.var(ha[w]) + np.var(he[w])))
            start += 1.0
        if sds:
            out["jitter_dia_in_20ft_median"] = float(np.median(sds) * 2 * k)
            out["jitter_dia_in_20ft_p90"] = float(np.percentile(sds, 90) * 2 * k)
            out["jitter_source"] = "truth" if src_m is mt else "measured"
    # ES-02.2: target inside the field of view while the system is tracking it
    fov = np.array([_f(r.get("true_in_fov")) for r in rows])
    active = trk_all & np.isfinite(fov)
    if active.any():
        out["target_in_fov_pct"] = float(100.0 * np.mean(fov[active]))
    # latency (ES-05.1: capture -> actuator command, 95th percentile <= 100 ms)
    lat = np.array([_f(r["latency_ms"]) for r in rows])
    lat = lat[np.isfinite(lat)]
    if len(lat):
        out["latency_ms_p50"] = float(np.percentile(lat, 50))
        out["latency_ms_p95"] = float(np.percentile(lat, 95))
        out["latency_ms_max"] = float(np.max(lat))
    # range (UC-5)
    rng = np.array([_f(r["range_m"]) for r in rows])
    on = np.array([r["range_on_target"] == "1" for r in rows])
    if (on & np.isfinite(rng)).any():
        out["range_on_target_mean_m"] = float(np.nanmean(rng[on]))
    # re-acquisition (ES-03.1) from the event log
    lost_t, reacq, alerts = None, [], 0
    for ev in events:
        if ev.get("kind") != "state":
            continue
        if ev.get("new") == "SEARCHING" and ev.get("old") == "TRACKING":
            lost_t = ev["t_mono"]
        elif ev.get("new") == "TRACKING" and lost_t is not None:
            reacq.append(ev["t_mono"] - lost_t)
            lost_t = None
        elif ev.get("new") == "LOST":
            alerts += 1
            lost_t = None
    out["losses"] = len(reacq) + alerts
    out["reacquisitions"] = len(reacq)
    out["reacquire_s_max"] = float(max(reacq)) if reacq else None
    out["lost_alerts"] = alerts
    # frame log integrity (UC-6: intact frames, no dropped data)
    ids = [int(f["frame_id"]) for f in frames]
    saved = [f for f in frames if f["saved"] in ("1", "True")]
    missing_files = 0
    for f in saved[:5000]:
        if f["file"].startswith("frames/") and not (session / f["file"]).exists():
            missing_files += 1
    if (session / "frames.bin").exists():
        size = (session / "frames.bin").stat().st_size
        expect = sum(int(f["nbytes"]) for f in saved)
        out["frames_bin_ok"] = size == expect
    out["frames_indexed"] = len(frames)
    out["frames_saved"] = len(saved)
    out["frame_ids_contiguous"] = bool(ids) and ids == list(range(ids[0], ids[0] + len(ids)))
    out["frames_missing_files"] = missing_files
    out["dropped_by_logger"] = meta.get("frames_dropped_by_logger")
    return out


CHECKS = [
    ("latency_ms_p95", "<=", 100.0, "ES-05.1 / NFR-04 capture -> command p95 <= 100 ms"),
    ("target_in_fov_pct", ">=", 95.0, "ES-02.2 target inside FOV >= 95 % of frames"),
    ("reacquire_s_max", "<=", 2.0, "ES-03.1 resume tracking within 2 s"),
    ("jitter_dia_in_20ft_median", "<=", 1.0, "NFR-05 jitter within 1 in circle at 20 ft (median 1 s window)"),
    ("telemetry_missed_ticks", "<=", 0, "FR-07 no telemetry dropouts"),
    ("frame_ids_contiguous", "==", True, "UC-6 every captured frame indexed"),
]


def check(res: dict) -> list:
    out = []
    for key, op, thr, label in CHECKS:
        v = res.get(key)
        if v is None:
            out.append((label, None, "n/a"))
            continue
        ok = {"<=": v <= thr, ">=": v >= thr, "==": v == thr}[op]
        out.append((label, bool(ok), v))
    return out


def plots(session: Path, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, _, _, _ = load(session)
    t = np.array([_f(r["t_mono"]) for r in rows])
    t = t - t[0]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, [_f(r["pan_deg"]) for r in rows], label="pan (meas)")
    ax[0].plot(t, [_f(r["tilt_deg"]) for r in rows], label="tilt (meas)")
    ax[0].plot(t, [_f(r["kf_az_deg"]) for r in rows], "--", lw=0.8, label="target az (KF)")
    ax[0].plot(t, [_f(r["kf_el_deg"]) for r in rows], "--", lw=0.8, label="target el (KF)")
    ax[0].set_ylabel("deg")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(t, [_f(r["err_az_deg"]) for r in rows], lw=0.8, label="err az (vision)")
    ax[1].plot(t, [_f(r["err_el_deg"]) for r in rows], lw=0.8, label="err el (vision)")
    if any(r.get("true_err_az_deg") for r in rows):
        ax[1].plot(t, [_f(r["true_err_az_deg"]) for r in rows], ":", lw=0.8, label="err az (truth)")
        ax[1].plot(t, [_f(r["true_err_el_deg"]) for r in rows], ":", lw=0.8, label="err el (truth)")
    ax[1].set_ylabel("pointing error, deg")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, [_f(r["latency_ms"]) for r in rows], ".", ms=2, label="capture->command latency")
    ax[2].axhline(100, color="r", lw=0.8, label="100 ms budget")
    ax[2].set_ylabel("ms")
    ax[2].set_xlabel("time, s")
    ax[2].legend(loc="upper right", fontsize=8)
    states = [r["state"] for r in rows]
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            for a in ax:
                a.axvline(t[i], color="k", lw=0.3, alpha=0.4)
            ax[0].text(t[i], ax[0].get_ylim()[1], states[i], fontsize=6, rotation=90, va="top")
    fig.suptitle(session.name)
    fig.tight_layout()
    p = out_dir / "analysis.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return [p]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="log folder (or a parent folder: the newest session is used)")
    ap.add_argument("--json", help="write results to this JSON file")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    p = Path(a.session)
    if not (p / "telemetry.csv").exists():
        subs = sorted([d for d in p.iterdir() if (d / "telemetry.csv").exists()])
        if not subs:
            print("no session found in", p)
            return 1
        p = subs[-1]
    res = analyze(p)
    print(json.dumps(res, indent=2, default=str))
    print("\nRequirement checks")
    for label, ok, v in check(res):
        mark = "n/a " if ok is None else ("PASS" if ok else "FAIL")
        vv = f"{v:.3g}" if isinstance(v, float) else v
        print(f"  [{mark}] {label}: {vv}")
    if a.json:
        Path(a.json).write_text(json.dumps({"results": res, "checks": check(res)}, indent=2, default=str))
    if not a.no_plots:
        try:
            for f in plots(p, p):
                print("plot:", f)
        except Exception as exc:
            print("plotting skipped:", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
