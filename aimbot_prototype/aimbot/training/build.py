"""Merge the raw datasets into one YOLO dataset for YOLO26 training.

    raw:  datasets/raw/<source>/...           (tools/fetch_datasets.py puts them there)
    out:  datasets/aimbot_drones/
            images/{train,val,test}/<source>/<group>__<frame>.jpg
            labels/{train,val,test}/<source>/<group>__<frame>.txt
            aimbot_drones.yaml                 Ultralytics data file (all sources)
            sources/<source>.json              per-source build statistics
            build_report.md / .json            what went in: images, boxes per class, box sizes
            previews/<source>.jpg              random labelled samples - LOOK AT THESE before training

What the builder does to each dataset:
  * split by video / sequence (never by frame): the dataset's own split if it has one,
    otherwise a stable hash of the clip name (val_frac of the clips go to val);
  * keeps every ``stride``-th annotated frame of a video (consecutive frames are nearly
    identical) and every ``stride * neg_stride_mult``-th frame without targets as background;
  * maps every label onto the unified class list (aimbot/training/schema.py);
  * stores the full frame (long side <= max_side) and, for large frames, a native-resolution
    crop around the targets: AimBot runs YOLO on native-resolution crops around the
    locked target while tracking, and tiny drones in 1080p/4K frames vanish when the
    whole frame is shrunk to 640 px;
  * drops frames that would teach the wrong thing (aimbot/training/quality.py): label glitches,
    degenerate or unlearnably small boxes, black / washed-out / unusually blurry frames, labelled
    targets that are not there, near-duplicates and anything in exclude_list. Every removal is
    counted per reason in the report, with samples in previews/<source>_rejected.jpg.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import quality as Q
from .schema import CLASSES
from .sources import Group, Item, box_stats, get_source

log = logging.getLogger(__name__)

DEFAULTS = {
    "output": "datasets/aimbot_drones",
    "raw_root": "datasets/raw",
    "classes": ["drone", "bird", "airplane", "helicopter"],
    "val_frac": 0.15,
    "seed": 0,
    "max_side": 1280,
    "jpeg_quality": 90,
    "min_box_px": 2.0,
    "stride": 1,
    "neg_stride_mult": 5,
    "max_per_group": 0,
    "crop": {"enabled": True, "size": 640, "trigger": 960, "per_image": 1, "min_visible": 0.4, "small_px_at_640": 40},
    "quality": dict(Q.DEFAULTS),
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def stable_frac(text: str) -> float:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:120]


def choose_split(source: str, group: Group, val_frac: float) -> str:
    if group.split in ("train", "val", "test"):
        return group.split
    return "val" if stable_frac(f"{source}/{group.name}") < val_frac else "train"


def select_items(items: list, stride: int, neg_stride: int, max_per_group: int, salt: str) -> list[int]:
    """Indices of the items to keep: every stride-th positive, every neg_stride-th negative."""
    stride, neg_stride = max(1, int(stride)), max(1, int(neg_stride))
    pos = [i for i, it in enumerate(items) if it.boxes]
    neg = [i for i, it in enumerate(items) if not it.boxes]
    off = int(stable_frac(salt) * 1000)
    sel = pos[off % stride::stride] + neg[off % neg_stride::neg_stride]
    sel.sort()
    if max_per_group and len(sel) > max_per_group:
        step = len(sel) / max_per_group
        sel = [sel[int(k * step)] for k in range(max_per_group)]
    return sel


# ----------------------------------------------------------------------------- geometry helpers
def to_pixel_boxes(item: Item, w: int, h: int) -> list:
    if not item.norm:
        return list(item.boxes)
    return [(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h) for c, cx, cy, bw, bh in item.boxes]


def clip_boxes(boxes: list, w: int, h: int, min_px: float, min_visible: float = 0.0) -> tuple[list, int]:
    out, dropped = [], 0
    for c, x1, y1, x2, y2 in boxes:
        a0 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
        a1 = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        if a0 <= 0 or a1 <= 0:
            if a0 > 0:
                dropped += 1
            continue
        if (cx2 - cx1) < min_px or (cy2 - cy1) < min_px or (min_visible and a1 / a0 < min_visible):
            dropped += 1
            continue
        out.append((c, cx1, cy1, cx2, cy2))
    return out, dropped


def yolo_lines(boxes: list, w: int, h: int, cls_index: dict) -> list[str]:
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        k = cls_index.get(c)
        if k is None:
            continue
        lines.append(f"{k} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def pick_crop(boxes: list, w: int, h: int, size: int, rng: random.Random, tries: int = 8):
    """A size x size window containing one target, chosen to cut as few other targets as possible."""
    s = int(min(size, w, h))
    best, best_cut = None, None
    for _ in range(tries):
        c, x1, y1, x2, y2 = boxes[rng.randrange(len(boxes))]
        lo_x, hi_x = max(0, int(math.ceil(x2)) - s), min(w - s, int(x1))
        lo_y, hi_y = max(0, int(math.ceil(y2)) - s), min(h - s, int(y1))
        if hi_x < lo_x:
            lo_x = hi_x = int(min(max(0, (x1 + x2) / 2 - s / 2), w - s))
        if hi_y < lo_y:
            lo_y = hi_y = int(min(max(0, (y1 + y2) / 2 - s / 2), h - s))
        x0, y0 = rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
        cut = 0
        for _, a1, b1, a2, b2 in boxes:
            ia = max(0.0, min(a2, x0 + s) - max(a1, x0)) * max(0.0, min(b2, y0 + s) - max(b1, y0))
            ar = max(1e-9, (a2 - a1) * (b2 - b1))
            if 0 < ia / ar < 0.99:
                cut += 1
        if best is None or cut < best_cut:
            best, best_cut = (x0, y0, s), cut
            if cut == 0:
                break
    return best


# ----------------------------------------------------------------------------- per-group worker
def _thumb(path: Path, img, boxes: list, reason: str, tile: int = 256, measured=None,
           measured_color=(0, 230, 255)) -> None:
    """Labelled thumbnail of a rejected frame: a native-resolution zoom on the first box (whole frame if none)."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    if boxes:
        _, x1, y1, x2, y2 = boxes[0]
        side = int(max(96.0, 6 * max(x2 - x1, y2 - y1)))
        side = min(side, w, h)
        ox = int(max(0, min(w - side, (x1 + x2) / 2 - side / 2)))
        oy = int(max(0, min(h - side, (y1 + y2) / 2 - side / 2)))
    else:
        side, ox, oy = max(w, h), 0, 0
    view = img[oy:oy + side, ox:ox + side]
    s = tile / max(view.shape[:2])
    t = cv2.resize(view, (max(1, int(view.shape[1] * s)), max(1, int(view.shape[0] * s))),
                   interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(t, (int((x1 - ox) * s) - 1, int((y1 - oy) * s) - 1), (int((x2 - ox) * s) + 1, int((y2 - oy) * s) + 1),
                      (0, 0, 255), 1)
    if measured is not None:                           # the object as measured in the pixels (box-fit checks)
        x1, y1, x2, y2 = measured
        cv2.rectangle(t, (int((x1 - ox) * s), int((y1 - oy) * s)), (int((x2 - ox) * s), int((y2 - oy) * s)),
                      measured_color, 1)
    canvas = np.zeros((tile, tile, 3), np.uint8)
    canvas[: t.shape[0], : t.shape[1]] = t[:tile, :tile]
    cv2.putText(canvas, reason, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def process_group(task: dict) -> dict:
    """Extract, quality-check and write the selected frames of one group. Runs in a worker process."""
    import cv2

    g: Group = task["group"]
    st = task["settings"]
    split = task["split"]
    out = Path(task["out"])
    q = task.get("quality") or {"enabled": False}
    use_q = bool(q.get("enabled", True))
    excl = task.get("exclude") or []
    cls_index = {c: i for i, c in enumerate(st["classes"])}
    img_dir = out / "images" / split / g.source
    lab_dir = out / "labels" / split / g.source
    rej_dir = out / "rejected" / g.source
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "crops": 0, "negatives": 0, "boxes": Counter(), "dropped_small": 0,
             "dropped_class": Counter(), "unreadable": 0, "sizes_native": [], "sizes_640": [], "resolutions": Counter(),
             "rejected": Counter(), "full_skipped_tiny": 0, "tightened": 0, "tightened_frames": 0}
    sel = task["selected"]
    if not sel:
        return {"source": g.source, "split": split, "group": g.name, "stats": stats}
    rng = random.Random(stable_frac(f"crop/{g.source}/{g.name}"))
    q_jpg = [int(cv2.IMWRITE_JPEG_QUALITY), int(st["jpeg_quality"])]
    max_side, min_px = int(st["max_side"]), float(st["min_box_px"])
    crop = st.get("crop") or {}
    jumps = Q.temporal_outliers(g.items, q) if use_q else set()
    thumbs_done: set = set()
    kept: list = []                      # (item index, sharpness, [written paths], img for thumbnail)
    last_hash, last_boxes = None, None

    def reject(i: int, reason: str, img=None, boxes=None, measured=None) -> None:
        stats["rejected"][reason] += 1
        if img is not None and reason not in thumbs_done:      # one sample per reason per clip
            thumbs_done.add(reason)
            stem = safe_name(f"{g.name}__{g.items[i].name or ('f%06d' % g.items[i].frame)}")
            _thumb(rej_dir / f"{reason}__{stem}.jpg", img, boxes or [], reason, measured=measured)

    def frames():
        if g.video is not None:
            cap = cv2.VideoCapture(str(g.video))
            want = {g.items[i].frame: i for i in sel}
            last = max(want)
            f = 0
            while f <= last:
                if not cap.grab():
                    break
                if f in want:
                    ok, img = cap.retrieve()
                    yield want[f], (img if ok else None)
                f += 1
            cap.release()
        else:
            for i in sel:
                img = cv2.imread(str(g.items[i].image), cv2.IMREAD_COLOR)
                yield i, img

    for i, img in frames():
        it = g.items[i]
        if img is None:
            stats["unreadable"] += 1
            continue
        h, w = img.shape[:2]
        stats["resolutions"][f"{w}x{h}"] += 1
        raw = to_pixel_boxes(it, w, h)
        known = []
        for b in raw:
            if b[0] in cls_index:
                known.append(b)
            else:
                stats["dropped_class"][b[0]] += 1
        boxes, nd = clip_boxes(known, w, h, 1.0)
        stats["dropped_small"] += nd
        stem = safe_name(f"{g.name}__{it.name or ('f%06d' % it.frame)}")
        if excl and any(fnmatch.fnmatchcase(f"{g.source}/{stem}", pat) for pat in excl):
            reject(i, "excluded")
            continue
        if use_q:
            gray = Q.small_gray(img)
            # an empty box in the label file (clip_boxes drops it) would leave a frame that shows the target
            # as a "background" image: judge the labels as written
            empty = any(b[3] - b[1] <= 0 or b[4] - b[2] <= 0 for b in known)
            reason = ("degenerate_box" if empty else None) or Q.label_reason(boxes, w, h, q) \
                or ("label_jump" if i in jumps else None) or Q.exposure_reason(gray, q, img, boxes)
            bad_box, measured = None, None
            fit_mode = str(q.get("box_fit", "tighten")).lower()
            fit_mode = {"true": "tighten", "false": "off", "none": "off"}.get(fit_mode, fit_mode)
            full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if boxes and (
                q.get("invisible_target", False) or fit_mode != "off") else None
            if reason is None and full_gray is not None and q.get("invisible_target", False):
                bad_box = Q.invisible_target(full_gray, boxes, q)
                if bad_box is not None:
                    reason = "invisible_target"
            if reason is None and full_gray is not None and fit_mode in ("tighten", "drop"):
                fits = Q.fit_boxes(full_gray, boxes, q)
                bad = [f for f in fits if f[1] == "misplaced_box" or fit_mode == "drop"]
                if bad:
                    bad.sort(key=lambda f: f[1] != "misplaced_box")
                    bad_box, reason, measured = bad[0]
                elif fits:                                     # tighten the loose labels, keep the frame
                    orig = list(boxes)
                    for k, _, t in fits:
                        boxes[k] = Q.tighten_box(boxes[k], t, q)
                    stats["tightened"] += len(fits)
                    stats["tightened_frames"] += 1
                    if "tightened" not in thumbs_done:
                        thumbs_done.add("tightened")
                        k0 = fits[0][0]
                        _thumb(out / "tightened" / g.source / f"tightened__{stem}.jpg", img,
                               [orig[k0]] + [b for j, b in enumerate(orig) if j != k0], "tightened",
                               measured=boxes[k0][1:], measured_color=(80, 220, 80))
            if reason is None and int(q.get("dedup_hamming", 0)) > 0:
                hsh = Q.dhash(gray)
                if (last_hash is not None and Q.hamming(hsh, last_hash) <= int(q["dedup_hamming"])
                        and Q.same_layout(boxes, last_boxes or [])):
                    reason = "duplicate"
                else:
                    last_hash, last_boxes = hsh, boxes
            if reason is not None:
                # zoom the thumbnail on the box that caused it
                reject(i, reason, img, ([boxes[bad_box]] + [b for k, b in enumerate(boxes) if k != bad_box])
                       if bad_box is not None else boxes, measured)
                continue
            sharp = Q.sharpness(gray)
        else:
            sharp = 0.0
        written: list = []
        s640 = 640.0 / max(w, h)
        scale = min(1.0, max_side / max(w, h))
        full = img if scale >= 1.0 else cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        fw, fh = full.shape[1], full.shape[0]
        fboxes = [(c, x1 * fw / w, y1 * fh / h, x2 * fw / w, y2 * fh / h) for c, x1, y1, x2, y2 in boxes]
        fboxes, nd = clip_boxes(fboxes, fw, fh, min_px)
        n_img = 0
        if nd and use_q:
            # a target became too small to label after downscaling: the full frame would show it
            # unlabelled (teaching "this is background"), so keep only the native-resolution crop
            stats["full_skipped_tiny"] += 1
        else:
            stats["dropped_small"] += nd
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), full, q_jpg)
            (lab_dir / f"{stem}.txt").write_text("\n".join(yolo_lines(fboxes, fw, fh, cls_index)) + ("\n" if fboxes else ""))
            written += [img_dir / f"{stem}.jpg", lab_dir / f"{stem}.txt"]
            n_img += 1
        small = float(crop.get("small_px_at_640", 40))
        n_crop = 0
        if (crop.get("enabled", True) and boxes and max(w, h) > int(crop.get("trigger", 960))
                and (small <= 0 or min(max(x2 - x1, y2 - y1) for _, x1, y1, x2, y2 in boxes) * s640 < small)):
            for k in range(int(crop.get("per_image", 1))):
                win = pick_crop(boxes, w, h, int(crop.get("size", 640)), rng)
                if win is None:
                    break
                x0, y0, s = win
                cb = [(c, x1 - x0, y1 - y0, x2 - x0, y2 - y0) for c, x1, y1, x2, y2 in boxes]
                cb, _ = clip_boxes(cb, s, s, min_px, float(crop.get("min_visible", 0.4)))
                if not cb:
                    continue
                cv2.imwrite(str(img_dir / f"{stem}_c{k}.jpg"), img[y0:y0 + s, x0:x0 + s], q_jpg)
                (lab_dir / f"{stem}_c{k}.txt").write_text("\n".join(yolo_lines(cb, s, s, cls_index)) + "\n")
                written += [img_dir / f"{stem}_c{k}.jpg", lab_dir / f"{stem}_c{k}.txt"]
                n_crop += 1
        if not written:
            continue
        kept.append((i, sharp, written, n_img, n_crop, bool(n_img and not fboxes), boxes, s640, fboxes if n_img else []))
    # frames much blurrier than the rest of their clip (focus hunting, zoom transitions)
    blurry = set()
    if use_q and float(q.get("sharpness_rel", 0)) > 0 and len(kept) >= int(q.get("min_clip_frames", 8)):
        med = sorted(k[1] for k in kept)[len(kept) // 2]
        thr = float(q["sharpness_rel"]) * med
        for k in kept:
            if k[1] < thr:
                blurry.add(k[0])
    for i, sharp, written, n_img, n_crop, is_neg, boxes, s640, fboxes in kept:
        if i in blurry:
            first = next((pth for pth in written if pth.suffix == ".jpg"), None)
            im = cv2.imread(str(first)) if (first is not None and "blurry" not in thumbs_done) else None
            # the full frame (if written) is first, with fboxes in its own pixels; a crop gets no boxes drawn
            reject(i, "blurry", im, fboxes if (im is not None and n_img) else [])
            for pth in written:
                try:
                    pth.unlink()
                except OSError:
                    pass
            continue
        stats["images"] += n_img
        stats["crops"] += n_crop
        if is_neg:
            stats["negatives"] += 1
        for c, x1, y1, x2, y2 in boxes:
            stats["boxes"][c] += 1
            stats["sizes_native"].append((x2 - x1, y2 - y1))
            stats["sizes_640"].append(((x2 - x1) * s640, (y2 - y1) * s640))
    for key in ("sizes_native", "sizes_640"):
        v = stats[key]
        if len(v) > 400:
            stats[key] = random.Random(1).sample(v, 400)
    return {"source": g.source, "split": split, "group": g.name, "stats": stats}


# ----------------------------------------------------------------------------- driver
def load_settings(path: Optional[Path]) -> dict:
    cfg = dict(DEFAULTS)
    if path is not None and Path(path).exists():
        import yaml

        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg


def build(cfg: dict, base: Path, only: Optional[list] = None, clean: bool = False, workers: int = 0,
          preview: bool = True, dry_run: bool = False, progress=print) -> dict:
    """Build (or rebuild some sources of) the merged dataset. Returns the report dict."""
    out = (base / cfg["output"]).resolve() if not Path(cfg["output"]).is_absolute() else Path(cfg["output"])
    raw_root = (base / cfg["raw_root"]).resolve() if not Path(cfg["raw_root"]).is_absolute() else Path(cfg["raw_root"])
    classes = [c for c in (cfg.get("classes") or CLASSES)]
    bad = [c for c in classes if c not in CLASSES]
    if bad:
        raise ValueError(f"classes {bad} are not in the unified list {CLASSES} (aimbot/training/schema.py)")
    manifest = out / "build_manifest.json"
    if clean and out.exists() and not dry_run:
        shutil.rmtree(out)
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get("classes") != classes:
            raise SystemExit(f"{out} was built with classes {old.get('classes')}; the class list is now {classes}. "
                             "Re-run with --clean to rebuild every source with the new class indices.")
    sources_cfg = cfg.get("sources") or {}
    keys = [k for k, v in sources_cfg.items() if (v or {}).get("enabled", True)]
    if only:
        unknown = [k for k in only if k not in sources_cfg]
        if unknown:
            raise SystemExit(f"unknown source(s) {unknown}; defined in sources.yaml: {list(sources_cfg)}")
        keys = [k for k in only]
    settings_common = {k: cfg[k] for k in ("max_side", "jpeg_quality", "min_box_px", "crop")}
    settings_common["classes"] = classes
    q_global = Q.settings(cfg.get("quality"))
    exclude = load_exclude_list(q_global.get("exclude_list"), base)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    skipped, built = {}, {}
    t_start = time.time()
    for key in keys:
        scfg = _merge({k: cfg[k] for k in ("stride", "neg_stride_mult", "max_per_group", "val_frac")}, sources_cfg.get(key) or {})
        src = get_source(key, scfg)
        root = raw_root / str(scfg.get("path", key))
        if not src.present(root):
            skipped[key] = f"no data in {root}"
            progress(f"[{key}] skipped: no images/videos in {root}  ({src.access or 'see docs/TRAINING.md'})")
            continue
        progress(f"[{key}] reading annotations in {root} ...")
        t0 = time.time()
        groups = list(src.groups(root, scfg))
        if not groups:
            skipped[key] = "no annotated images/videos recognised"
            progress(f"[{key}] skipped: nothing recognised (run tools/build_dataset.py --inspect {key})")
            continue
        q_src = Q.settings({**q_global, **(scfg.get("quality") or {})})
        splits = [choose_split(key, g, float(scfg["val_frac"])) for g in groups]
        if "val" not in splits and len(groups) >= 3 and float(scfg["val_frac"]) > 0 and not any(g.split for g in groups):
            j = min(range(len(groups)), key=lambda i: stable_frac(f"{key}/{groups[i].name}"))
            splits[j] = "val"
        tasks = []
        for g, sp in zip(groups, splits):
            stride = int(scfg.get("stride", 1)) if sp == "train" else int(scfg.get("val_stride", scfg.get("stride", 1)))
            sel = select_items(g.items, stride, stride * int(scfg.get("neg_stride_mult", 5)),
                               int(scfg.get("max_per_group", 0) or 0), f"{key}/{g.name}")
            tasks.append({"group": g, "split": sp, "selected": sel, "settings": settings_common, "out": str(out),
                          "quality": q_src, "exclude": exclude})
        n_sel = sum(len(t["selected"]) for t in tasks)
        progress(f"[{key}] {len(groups)} clips/folders ({Counter(splits)}), {sum(len(g.items) for g in groups)} "
                 f"annotated frames -> {n_sel} selected  ({time.time() - t0:.0f} s)")
        if dry_run:
            built[key] = {"groups": len(groups), "selected": n_sel, "splits": dict(Counter(splits))}
            continue
        for sp in ("train", "val", "test"):
            for kind in ("images", "labels"):
                d = out / kind / sp / key
                if d.exists():
                    shutil.rmtree(d)
        for sub in ("rejected", "tightened"):
            if (out / sub / key).exists():
                shutil.rmtree(out / sub / key)
        agg = _new_agg()
        done = 0
        t1 = time.time()
        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process_group, t) for t in tasks]
                for f in as_completed(futs):
                    _accumulate(agg, f.result())
                    done += 1
                    if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                        progress(f"[{key}]   {done}/{len(tasks)} clips, {agg['images']} images, {time.time() - t1:.0f} s")
        else:
            for t in tasks:
                _accumulate(agg, process_group(t))
                done += 1
                if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                    progress(f"[{key}]   {done}/{len(tasks)} clips, {agg['images']} images, {time.time() - t1:.0f} s")
        rep = _finish_agg(agg, src, scfg, root)
        (out / "sources").mkdir(parents=True, exist_ok=True)
        (out / "sources" / f"{key}.json").write_text(json.dumps(rep, indent=1))
        built[key] = rep
        progress(f"[{key}] done: {rep['images_total']} images ({rep['crops_total']} crops, {rep['negatives_total']} background)"
                 f" in {time.time() - t1:.0f} s")
    if dry_run:
        return {"dry_run": True, "built": built, "skipped": skipped}
    out.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"classes": classes, "updated": _dt.datetime.now().isoformat(timespec="seconds")}))
    report = write_report(out, classes, skipped, time.time() - t_start)
    if preview:
        for key in built:
            try:
                make_preview(out, key, classes)
                make_rejected_preview(out, key)
                make_rejected_preview(out, key, sub="tightened")
            except Exception as exc:                      # preview is a convenience
                log.warning("preview for %s failed: %s", key, exc)
    return report


def _new_agg() -> dict:
    return {"images": 0, "by_split": defaultdict(lambda: Counter()), "groups": Counter(), "sizes_native": [],
            "sizes_640": [], "dropped_small": 0, "dropped_class": Counter(), "unreadable": 0, "resolutions": Counter(),
            "rejected": Counter(), "full_skipped_tiny": 0, "tightened": 0, "tightened_frames": 0}


def _accumulate(agg: dict, res: dict) -> None:
    st, sp = res["stats"], res["split"]
    b = agg["by_split"][sp]
    b["images"] += st["images"]
    b["crops"] += st["crops"]
    b["negatives"] += st["negatives"]
    for c, n in st["boxes"].items():
        b[f"box:{c}"] += n
    agg["groups"][sp] += 1
    agg["images"] += st["images"] + st["crops"]
    agg["sizes_native"] += st["sizes_native"]
    agg["sizes_640"] += st["sizes_640"]
    agg["dropped_small"] += st["dropped_small"]
    agg["dropped_class"].update(st["dropped_class"])
    agg["unreadable"] += st["unreadable"]
    agg["resolutions"].update(st["resolutions"])
    agg["rejected"].update(st.get("rejected", {}))
    agg["full_skipped_tiny"] += st.get("full_skipped_tiny", 0)
    agg["tightened"] += st.get("tightened", 0)
    agg["tightened_frames"] += st.get("tightened_frames", 0)


def _finish_agg(agg: dict, src, scfg: dict, root: Path) -> dict:
    splits = {}
    for sp, c in agg["by_split"].items():
        splits[sp] = {"clips": agg["groups"][sp], "images": c["images"], "crops": c["crops"], "background": c["negatives"],
                      "boxes": {k[4:]: v for k, v in sorted(c.items()) if k.startswith("box:")}}
    return {
        "source": src.key, "title": src.title, "homepage": src.homepage, "license": src.license, "root": str(root),
        "stride": scfg.get("stride", 1), "splits": splits,
        "images_total": sum(s["images"] + s["crops"] for s in splits.values()),
        "crops_total": sum(s["crops"] for s in splits.values()),
        "negatives_total": sum(s["background"] for s in splits.values()),
        "box_px_native": box_stats(agg["sizes_native"]), "box_px_at_640": box_stats(agg["sizes_640"]),
        "dropped_small": agg["dropped_small"], "dropped_not_in_classes": dict(agg["dropped_class"]),
        "unreadable_frames": agg["unreadable"], "resolutions": dict(agg["resolutions"].most_common(6)),
        "rejected": dict(agg["rejected"]), "full_frame_skipped_tiny_target": agg["full_skipped_tiny"],
        "boxes_tightened": agg["tightened"], "frames_with_tightened_boxes": agg["tightened_frames"],
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def write_report(out: Path, classes: list, skipped: dict, seconds: float) -> dict:
    reps = []
    for p in sorted((out / "sources").glob("*.json")):
        try:
            reps.append(json.loads(p.read_text()))
        except Exception:
            pass
    has_test = any("test" in r["splits"] for r in reps)
    yaml_lines = [f"# AimBot detector dataset - written by tools/build_dataset.py, {_dt.date.today().isoformat()}",
                  f"# sources: {', '.join(r['source'] for r in reps)}",
                  f"path: {out.as_posix()}", "train: images/train", "val: images/val"]
    if has_test:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    yaml_lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    (out / "aimbot_drones.yaml").write_text("\n".join(yaml_lines) + "\n")
    tot = Counter()
    md = ["# AimBot detector dataset - build report", "",
          f"Classes: {', '.join(f'{i} {c}' for i, c in enumerate(classes))}", "",
          "| source | split | clips | images | crops | background | " + " | ".join(classes) + " |",
          "|---|---|---:|---:|---:|---:|" + "---:|" * len(classes)]
    for r in reps:
        for sp in ("train", "val", "test"):
            s = r["splits"].get(sp)
            if not s:
                continue
            md.append(f"| {r['source']} | {sp} | {s['clips']} | {s['images']} | {s['crops']} | {s['background']} | "
                      + " | ".join(str(s["boxes"].get(c, 0)) for c in classes) + " |")
            tot[f"{sp}:images"] += s["images"] + s["crops"]
            for c in classes:
                tot[f"{sp}:{c}"] += s["boxes"].get(c, 0)
    md += ["", f"Total images: train {tot['train:images']}, val {tot['val:images']}, test {tot['test:images']}", ""]
    md += ["| source | median box (native px) | median box at 640 | dropped <2 px | dropped (class not listed) | licence |",
           "|---|---|---|---:|---|---|"]
    for r in reps:
        bn, b6 = r.get("box_px_native") or {}, r.get("box_px_at_640") or {}
        md.append(f"| {r['source']} | {bn.get('median_w', '-')} x {bn.get('median_h', '-')} | {b6.get('median_w', '-')} x "
                  f"{b6.get('median_h', '-')} | {r['dropped_small']} | {r['dropped_not_in_classes'] or '-'} | {r['license']} |")
    reasons = sorted({r for rep in reps for r in (rep.get("rejected") or {})})
    if reasons:
        md += ["", "Frames removed by the quality filter (datasets/sources.yaml `quality:`; samples in "
               "`previews/<source>_rejected.jpg`):", "",
               "| source | " + " | ".join(reasons) + " |", "|---|" + "---:|" * len(reasons)]
        for rep in reps:
            rj = rep.get("rejected") or {}
            md.append(f"| {rep['source']} | " + " | ".join(str(rj.get(r, 0)) for r in reasons) + " |")
        md += [""] + [f"* `{r}`: {Q.REASONS.get(r, r)}" for r in reasons]
    tight = [(rep["source"], rep.get("boxes_tightened", 0), rep.get("frames_with_tightened_boxes", 0)) for rep in reps]
    if any(t[1] for t in tight):
        md += ["", "Loose labels tightened to the measured object (frames kept; `box_fit: tighten`, samples in "
               "`previews/<source>_tightened.jpg`, red = original label, green = new):", ""]
        md += [f"* {src}: {n} boxes in {nf} frames" for src, n, nf in tight if n]
    if skipped:
        md += ["", "Not included in this build:", ""] + [f"* **{k}**: {v}" for k, v in skipped.items()]
    md += ["", "Check `previews/<source>.jpg` before training: boxes must sit on the targets.", ""]
    (out / "build_report.md").write_text("\n".join(md))
    report = {"classes": classes, "sources": reps, "skipped": skipped, "totals": dict(tot), "seconds": round(seconds, 1)}
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    return report


def make_preview(out: Path, key: str, classes: list, n: int = 12, tile: int = 320) -> Optional[Path]:
    import cv2
    import numpy as np

    cands = sorted((out / "labels" / "train" / key).glob("*.txt")) + sorted((out / "labels" / "val" / key).glob("*.txt"))
    cands = [c for c in cands if c.stat().st_size > 0]
    if not cands:
        return None
    rng = random.Random(7)
    picks = rng.sample(cands, min(n, len(cands)))
    tiles = []
    colors = [(255, 200, 80), (80, 200, 255), (120, 255, 120), (200, 120, 255), (60, 230, 240), (255, 255, 255), (120, 140, 255)]
    for lab in picks:
        img_p = out / "images" / lab.parent.parent.name / key / f"{lab.stem}.jpg"      # labels/<split>/<key>/x.txt
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = tile / max(w, h)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        for ln in lab.read_text().splitlines():
            k, cx, cy, bw, bh = ln.split()
            k = int(k)
            x1, y1 = (float(cx) - float(bw) / 2) * w * s, (float(cy) - float(bh) / 2) * h * s
            x2, y2 = (float(cx) + float(bw) / 2) * w * s, (float(cy) + float(bh) / 2) * h * s
            col = colors[k % len(colors)]
            cv2.rectangle(canvas, (int(x1) - 1, int(y1) - 1), (int(x2) + 1, int(y2) + 1), col, 1)
            cv2.putText(canvas, classes[k] if k < len(classes) else str(k), (int(x1), max(9, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        cv2.putText(canvas, img_p.stem[:44], (3, tile - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    (out / "previews").mkdir(exist_ok=True)
    p = out / "previews" / f"{key}.jpg"
    cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return p


def load_exclude_list(path, base: Path) -> list:
    """Patterns "<source>/<image stem>" (fnmatch wildcards allowed), one per line, '#' comments."""
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    if not p.exists():
        log.warning("exclude_list %s not found", p)
        return []
    pats = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.split("#", 1)[0].strip().replace("\\", "/")
        if not ln:
            continue
        ln = ln[:-4] if ln.lower().endswith((".jpg", ".txt")) else ln
        ln = re.sub(r"_c\d+$", "", ln)                     # a crop: exclude the frame it came from
        pats.append(ln)
    return pats


def make_rejected_preview(out: Path, key: str, n: int = 16, tile: int = 256, sub: str = "rejected") -> Optional[Path]:
    import cv2
    import numpy as np

    d = out / sub / key
    if not d.exists():
        return None
    files = sorted(d.glob("*.jpg"))
    if not files:
        return None
    by_reason: dict = defaultdict(list)
    for f in files:
        by_reason[f.name.split("__", 1)[0]].append(f)
    rng = random.Random(3)
    picks = []
    while len(picks) < n and any(by_reason.values()):        # round-robin over reasons
        for r in sorted(by_reason):
            if by_reason[r] and len(picks) < n:
                picks.append(by_reason[r].pop(rng.randrange(len(by_reason[r]))))
    tiles = []
    for f in picks:
        im = cv2.imread(str(f))
        if im is None:
            continue
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[: im.shape[0], : im.shape[1]] = im[:tile, :tile]
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    (out / "previews").mkdir(exist_ok=True)
    p = out / "previews" / f"{key}_{sub}.jpg"
    cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return p


def inspect(cfg: dict, base: Path, key: str, progress=print) -> None:
    """Print what a reader finds in a raw folder (formats, groups, example boxes) without writing anything."""
    raw_root = (base / cfg["raw_root"]).resolve()
    scfg = (cfg.get("sources") or {}).get(key) or {}
    src = get_source(key, scfg)
    root = raw_root / str(scfg.get("path", key))
    progress(f"{key}: reader {type(src).__name__}, folder {root}, present: {src.present(root)}")
    n_g = n_i = n_b = 0
    cls = Counter()
    for g in src.groups(root, scfg):
        n_g += 1
        n_i += len(g.items)
        for it in g.items:
            n_b += len(it.boxes)
            cls.update(b[0] for b in it.boxes)
        if n_g <= 5:
            ex = next((it for it in g.items if it.boxes), None)
            progress(f"  group {g.name!r} split={g.split} modality={g.modality} items={len(g.items)} video={g.video}"
                     f" notes={g.notes} example={ex.boxes[:2] if ex else None} {'(normalised)' if ex and ex.norm else ''}")
    progress(f"  {n_g} groups, {n_i} items, {n_b} boxes, classes {dict(cls)}")
