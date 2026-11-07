"""Find suspicious labels with a trained model, so you can review and exclude them before the final training.

    python tools/audit_labels.py --model models/aimbot_yolo26n.pt               # every source, train + val
    python tools/audit_labels.py --model runs/detect/quick/weights/best.pt --sources dut_anti_uav

For each image the model's detections are compared with the labels:
  missed_target   no detection of any class on a labelled target (even at confidence 0.05): often an
                  empty or misplaced box
  missing_label   a confident detection (>= 0.5) that is not any labelled object: an unlabelled drone
                  or bird (e.g. the birds in Drone-vs-Bird, a second drone nobody boxed)
  wrong_class     a confident detection of another class on a label (review only)
  loose_box       the label is much bigger than a confident detection lying inside it

Output in <dataset>/audit/: audit.csv (every flag), <flag>.jpg review sheets (most suspicious first:
labels green, detections red) and exclude_suggested.txt, where missing_label and loose_box lines
are active and missed_target / wrong_class lines are commented out (--strict activates them too).
then in datasets/sources.yaml set
    quality: {exclude_list: datasets/aimbot_drones/audit/exclude_suggested.txt}
and rebuild (python tools/build_dataset.py --clean) and retrain.

Flags are candidates, not verdicts: always look at the sheets. Use a well-trained model (the full run
on the GPU) - a weak model mostly flags its own mistakes. A model is also less critical of labels it
was trained on, so this finds the clear cases, not all of them; strongest is auditing a dataset with a
model trained without it, or auditing the val split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "datasets" / "aimbot_drones" / "aimbot_drones.yaml"
COLORS = {"gt": (80, 220, 80), "pred": (60, 60, 255)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def read_labels(path: Path, w: int, h: int) -> list:
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        t = ln.split()
        if len(t) != 5:
            continue
        k, cx, cy, bw, bh = int(t[0]), *(float(v) for v in t[1:])
        out.append((k, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def overlap_frac(a, b) -> float:
    """Intersection over the area of a (how much of a lies inside b)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def same_object(a, b) -> bool:
    """Boxes that describe one object: overlapping, or one (mostly) inside the other (loose boxes)."""
    return iou(a, b) >= 0.3 or overlap_frac(a, b) >= 0.6 or overlap_frac(b, a) >= 0.6


def audit_image(gt: list, pred: list, conf_hi: float = 0.5) -> list:
    """gt: [(cls, x1, y1, x2, y2)], pred: [(cls, conf, x1, y1, x2, y2)] -> [(flag, cls, score, iou, box)].

    missed_target: no detection of ANY class anywhere on the label (a class mix-up is a model problem,
    not a box problem). missing_label: a confident detection that is not the same object as any label.
    wrong_class: a confident detection of another class on a label with no same-class detection
    (review only - the label is often right and the model wrong). loose_box: a confident same-class
    detection lies inside the label and is less than half its area (the label is the loose box, not the
    model's; score = label area / detection area).
    """
    flags = []
    for g in gt:
        side = max(g[3] - g[1], g[4] - g[2])
        if side >= 8 and not any(iou(g[1:], p[2:]) >= 0.1 or same_object(g[1:], p[2:]) for p in pred):
            flags.append(("missed_target", g[0], 1.0, 0.0, g[1:]))
            continue
        same = [(iou(g[1:], p[2:]), p) for p in pred if p[0] == g[0]]
        best_iou, best = max(same, key=lambda x: x[0], default=(0.0, None))
        if 0.1 <= best_iou < 0.5 and best is not None and best[1] >= 0.4:
            pa = (best[4] - best[2]) * (best[5] - best[3])
            ga = (g[3] - g[1]) * (g[4] - g[2])
            # only when the LABEL is the loose one: a confident detection lying inside it, much smaller
            if ga >= 2.0 * pa and overlap_frac(best[2:], g[1:]) >= 0.8:
                flags.append(("loose_box", g[0], ga / max(pa, 1e-9), best_iou, g[1:]))
    for p in pred:
        if p[1] < conf_hi:
            continue
        if not any(same_object(p[2:], g[1:]) for g in gt):
            flags.append(("missing_label", p[0], p[1], max((iou(p[2:], g[1:]) for g in gt), default=0.0), p[2:]))
            continue
        on = [g for g in gt if iou(p[2:], g[1:]) >= 0.5]
        if on and all(g[0] != p[0] for g in on) and not any(
                q[0] == g[0] and q[1] >= 0.25 and iou(q[2:], g[1:]) >= 0.5 for g in on for q in pred):
            flags.append(("wrong_class", p[0], p[1], max(iou(p[2:], g[1:]) for g in on), p[2:]))
    return flags


def review_sheet(rows: list, names, out: Path, n: int = 24, tile: int = 320) -> None:
    import cv2
    import numpy as np

    tiles = []
    for r in rows[:n]:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = r["box"]
        # zoom on the flagged box (context 4x its size, at least 160 px)
        side = max(160.0, 4 * max(x2 - x1, y2 - y1))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox, oy = int(max(0, min(w - side, cx - side / 2))), int(max(0, min(h - side, cy - side / 2)))
        crop = img[oy:oy + int(side), ox:ox + int(side)]
        s = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        for kind, boxes in (("gt", r["gt"]), ("pred", r["pred"])):
            for b in boxes:
                bx = b[1:] if kind == "gt" else b[2:]
                p1 = (int((bx[0] - ox) * s), int((bx[1] - oy) * s))
                p2 = (int((bx[2] - ox) * s), int((bx[3] - oy) * s))
                cv2.rectangle(crop, p1, p2, COLORS[kind], 1)
                if kind == "pred":
                    cv2.putText(crop, f"{names[b[0]]} {b[1]:.2f}", (p1[0], max(10, p1[1] - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS[kind], 1, cv2.LINE_AA)
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        cv2.putText(canvas, f"{r['flag']} {names[r['cls']]} s={r['score']:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, Path(r["path"]).stem[:46], (3, tile - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                    cv2.LINE_AA)
        tiles.append(canvas)
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="trained weights (.pt)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default="train,val")
    ap.add_argument("--sources", default="", help="comma-separated (default: all)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=0, help="per source and split (0 = all)")
    ap.add_argument("--conf-hi", type=float, default=0.5, help="confidence for missing_label / wrong_class")
    ap.add_argument("--strict", action="store_true",
                    help="activate every flag (missed_target and wrong_class too): cleanest labels, but it also "
                         "removes hard-but-correct examples - review the sheets first")
    ap.add_argument("--suggest", default="missing_label,loose_box",
                    help="flags whose lines are ACTIVE in exclude_suggested.txt; the others are written commented "
                         "out (on real data most missed_target / wrong_class flags were the model's mistakes)")
    a = ap.parse_args()
    import yaml
    from ultralytics import YOLO

    data = Path(a.data)
    d = yaml.safe_load(data.read_text())
    base = Path(d.get("path", data.parent))
    names = d["names"]
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    model = YOLO(a.model)
    want = {s.strip() for s in a.sources.split(",") if s.strip()}
    out_dir = base / "audit"
    out_dir.mkdir(exist_ok=True)
    rows, per_src = [], defaultdict(Counter)
    for split in [s.strip() for s in a.split.split(",") if s.strip()]:
        sd = base / "images" / split
        if not sd.exists():
            continue
        for src in sorted(p for p in sd.iterdir() if p.is_dir() and (not want or p.name in want)):
            imgs = sorted(src.glob("*.jpg"))
            if a.max_images:
                imgs = imgs[:: max(1, len(imgs) // a.max_images)][: a.max_images]
            print(f"[{src.name}/{split}] {len(imgs)} images", flush=True)
            for k in range(0, len(imgs), 256):
                chunk = imgs[k:k + 256]
                for img_path, res in zip(chunk, model.predict([str(p) for p in chunk], imgsz=a.imgsz, conf=0.05,
                                                              device=a.device, batch=a.batch, verbose=False,
                                                              stream=True)):
                    h, w = res.orig_shape
                    lab = base / "labels" / split / src.name / f"{img_path.stem}.txt"
                    gt = read_labels(lab, w, h)
                    b = res.boxes
                    pred = []
                    if b is not None and len(b):
                        for (x1, y1, x2, y2), c, kk in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                                           b.cls.cpu().numpy().astype(int)):
                            pred.append((int(kk), float(c), float(x1), float(y1), float(x2), float(y2)))
                    per_src[(src.name, split)]["images"] += 1
                    fl = audit_image(gt, pred, a.conf_hi)
                    if fl:
                        per_src[(src.name, split)]["flagged_images"] += 1
                    for flag, cls, score, ov, box in fl:
                        per_src[(src.name, split)][flag] += 1
                        rows.append({"path": str(img_path), "source": src.name, "split": split, "flag": flag,
                                     "cls": cls, "score": score, "iou": ov, "box": box, "gt": gt, "pred": pred})
    if not rows and not per_src:
        print(f"no images found under {base}/images")
        return 1
    with (out_dir / "audit.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "split", "image", "flag", "class", "score", "iou", "x1", "y1", "x2", "y2"])
        for r in sorted(rows, key=lambda r: (r["flag"], -r["score"])):
            wr.writerow([r["source"], r["split"], Path(r["path"]).name, r["flag"], names[r["cls"]], f"{r['score']:.3f}",
                         f"{r['iou']:.3f}", *(f"{v:.1f}" for v in r["box"])])
    by_flag = defaultdict(list)
    for r in rows:
        by_flag[r["flag"]].append(r)
    for flag, rs in by_flag.items():
        rs.sort(key=lambda r: -r["score"])
        review_sheet(rs, names, out_dir / f"{flag}.jpg")
    active = {s.strip() for s in a.suggest.split(",") if s.strip()}
    if a.strict:
        active |= {"missing_label", "loose_box", "missed_target", "wrong_class"}
    sections = [("missing_label", "confident detections that are no labelled object (usually an unlabelled target)"),
                ("loose_box", "label much bigger than a confident detection inside it (mostly cluttered backgrounds; "
                              "the builder already tightens loose boxes on plain backgrounds)"),
                ("missed_target", "labels with no detection at all: an empty / misplaced box OR a hard but correct "
                                  "example - look at missed_target.jpg"),
                ("wrong_class", "confident detection of another class on a label: model or label wrong - look at "
                                "wrong_class.jpg")]
    lines = ["# Suggested by tools/audit_labels.py. Lines starting with '#' are ignored by the builder.",
             "# Active (uncommented) lines are excluded from the next build; enable others only after review."]
    n_active = 0
    for flag, text in sections:
        rs = sorted({f"{r['source']}/{Path(r['path']).stem}": r for r in by_flag.get(flag, [])}.items(),
                    key=lambda kv: -kv[1]["score"])
        if not rs:
            continue
        on = flag in active
        lines += ["", f"# --- {flag} ({len(rs)}): {text}" + ("" if on else " - commented out, review first")]
        for key, r in rs:
            lines.append(f"{'' if on else '# '}{key}    # {flag} {names[r['cls']]} {r['score']:.2f}")
            n_active += on
    (out_dir / "exclude_suggested.txt").write_text("\n".join(lines) + "\n")
    print(f"\n{'source':18s} {'split':6s} {'images':>7s} {'flagged':>8s}  " + "  ".join(
        f"{f:>13s}" for f in ("missed_target", "missing_label", "wrong_class", "loose_box")))
    for (src, split), c in sorted(per_src.items()):
        print(f"{src:18s} {split:6s} {c['images']:7d} {c['flagged_images']:8d}  " + "  ".join(
            f"{c[f]:13d}" for f in ("missed_target", "missing_label", "wrong_class", "loose_box")))
    print(f"\nreview sheets + audit.csv in {out_dir}; exclude_suggested.txt: {n_active} active lines "
          f"(+ commented-out candidates to review)")
    return 0


if __name__ == "__main__":
    from aimbot import runlog

    sys.exit(runlog.run("audit", main))
