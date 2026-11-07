"""Image and label quality checks used by the dataset builder.

The builder does not "enhance" images (denoise, sharpen, super-resolve): the detector must learn
from frames that look like what the Blackfly delivers at run time, and enhancement would have to
be repeated on the Jetson for every frame. Instead it *removes* frames that would teach the wrong
thing, and records why (build report + previews/<source>_rejected.jpg):

  label checks   degenerate boxes (extreme aspect ratio, box covering most of the image),
                 targets too small to learn, single-frame label glitches in video labels
                 (a box that jumps away from where its neighbouring frames put it), and on plain
                 backgrounds the fit of every box to the object measured in the pixels: labels
                 much bigger than the object (loose_box) or not on it (misplaced_box)
  image checks   nearly black or washed-out frames, near-identical consecutive frames (hovering
                 drone on a static camera); optional and off by default because they also remove
                 good hard examples: frames much blurrier than the rest of their clip
                 (sharpness_rel) and boxes with nothing visible in them (invisible_target)

All thresholds live in datasets/sources.yaml (quality:), per source overridable.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

DEFAULTS = {
    "enabled": True,
    "min_target_px": 4,        # a target smaller than this (native pixels) is unlearnable: drop the frame
    "max_aspect": 8.0,         # box w/h or h/w above this: label error
    "max_box_frac": 0.8,       # box covering more than this fraction of the image: label error
    "temporal_jump": 2.0,      # video labels: box centre this many box sizes away from its neighbours' ...
    "temporal_area_ratio": 3.0,  # ... or area this many times off: single-frame label glitch
    "sharpness_rel": 0.0,      # > 0: drop frames less sharp than this fraction of their clip's median.
                               # Off by default: a pan from trees to plain sky also lowers sharpness, and
                               # motion blur is realistic for a gimbal camera (training adds it on purpose)
    "min_clip_frames": 8,      # clips with fewer kept frames skip the relative-sharpness check
    "dark_mean": 8.0,          # mean grey level below this: black frame
    "bright_frac": 0.6,        # more than this fraction of saturated pixels: washed out
    "exposure_min_contrast": 24,   # ... both kept when a labelled target stands out by this many grey levels
    "sequence_min_near": 0.8,  # label jumps only where consecutive boxes agree like in a video (Halmstad >= 0.85)
    "max_jump_frac": 0.05,     # ... and not in a group where more than 5 % of the frames would be "jumps"
    "invisible_target": False, # drop frames with a labelled box in which nothing stands out from a plain
                               # background. Off by default: on real data (Halmstad) most hits were faint but
                               # real distant targets - exactly the hard examples. tools/audit_labels.py finds
                               # empty / misplaced boxes far more reliably. Consider it for sets with known
                               # "target present" flag errors.
    "invisible_min_dev": 6.0,  # ...nothing in the box deviates more than this (grey levels) or 1.5x the ring noise
    "plain_ring_std": 6.0,     # ...only judged on plain backgrounds (sky, walls), never in clutter
    "box_fit": "tighten",      # measure the object inside each box (plain backgrounds only):
                               #   tighten: shrink loose labels to the object (+ margin), keep the frame
                               #   drop:    drop frames with loose labels (Halmstad: 58 % of visible frames!)
                               #   off:     no box-fit check
                               # tested on Halmstad: 6,706 loose boxes tightened, all checked samples correct
    "tighten_margin": 0.1,     # tightened box = measured object + 10 % of its size on each side ...
    "tighten_min_margin_px": 2,  # ... but at least 2 px, so tiny targets keep their blurred edge
    "loose_ratio": 2.0,        # loose: label side >= 2x the object's measured side ...
    "loose_min_excess_px": 10, # ... and > 10 px wider/taller than it (up to 5 px of slack per side is normal labelling)
    "misplaced_check": False,  # drop frames whose label is next to (not on) the measured object. Off: on
                               # Halmstad nearly every hit was a contrail, heat trail or cloud edge beside a
                               # correct label. tools/audit_labels.py (missing_label) finds real cases better
    "misplaced_inside": 0.6,   # misplaced: object fully measured, its centre outside the label, < 60 % inside
    "fit_seed_sigma": 4.0,     # object pixels: seeds > 4 sigma (and > fit_min_dev) from the background ...
    "fit_grow_sigma": 2.0,     # ... grown into connected pixels > 2 sigma, so faint edges count as object
    "fit_min_dev": 8.0,        # grey levels; also the floor for the seed threshold
    "fit_edge_frac": 0.3,      # the object's edge: pixels >= 30 % of its peak contrast (not halo / ringing)
    "dedup_hamming": 2,        # consecutive kept frames this similar (dHash bits) with the same boxes: keep one
    "exclude_list": "",        # file of "<source>/<image stem>" lines (wildcards ok), e.g. from tools/audit_labels.py
}

REASONS = {
    "degenerate_box": "box with no size, an extreme aspect ratio or covering most of the image",
    "tiny_target": "target smaller than min_target_px",
    "label_jump": "box jumps away from where the neighbouring frames put it",
    "dark": "nearly black frame",
    "washed_out": "mostly saturated frame",
    "blurry": "much blurrier than the rest of its clip",
    "invisible_target": "labelled target not visible against a plain background",
    "loose_box": "label much bigger than the object measured inside it (box_fit: drop)",
    "misplaced_box": "label not on the object (most of the measured object lies outside it)",
    "duplicate": "near-identical to the previous kept frame",
    "excluded": "listed in exclude_list",
}


def settings(over: Optional[dict]) -> dict:
    s = dict(DEFAULTS)
    s.update(over or {})
    return s


def small_gray(img: np.ndarray, width: int = 320) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    if w > width:
        g = cv2.resize(g, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    return g


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def target_visible(img: np.ndarray, boxes: list, min_contrast: float) -> bool:
    """Does any labelled box (+25 % margin) contain something that stands out (1st to 99th percentile of grey)?"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    for _, x1, y1, x2, y2 in boxes:
        mx, my = 0.25 * (x2 - x1) + 2, 0.25 * (y2 - y1) + 2
        a, b, c, d = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, x2 + mx)), int(min(h, y2 + my))
        if c - a < 2 or d - b < 2:
            continue
        lo, hi = np.percentile(g[b:d, a:c], (1, 99))
        if hi - lo >= min_contrast:
            return True
    return False


def exposure_reason(gray: np.ndarray, q: dict, img: Optional[np.ndarray] = None,
                    boxes: Optional[list] = None) -> Optional[str]:
    """'dark' / 'washed_out' for a nearly black / mostly saturated frame, unless a labelled target can be seen
    in it: drone lights at night (Anti-UAV) or a drone against a blown-out sky are hard examples, not bad frames."""
    reason = None
    if float(gray.mean()) < float(q["dark_mean"]):
        reason = "dark"
    elif float((gray >= 250).mean()) > float(q["bright_frac"]):
        reason = "washed_out"
    if reason and img is not None and boxes and target_visible(img, boxes, float(q.get("exposure_min_contrast", 24))):
        return None
    return reason


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash of a grey image."""
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def same_layout(b1: list, b2: list, thr: float = 0.85) -> bool:
    if len(b1) != len(b2):
        return False
    used = set()
    for c, *bb in b1:
        best = max(((iou(bb, o[1:]), j) for j, o in enumerate(b2) if o[0] == c and j not in used), default=(0.0, -1))
        if best[0] < thr:
            return False
        used.add(best[1])
    return True


def label_reason(boxes: list, w: int, h: int, q: dict) -> Optional[str]:
    """boxes: [(cls, x1, y1, x2, y2)] in pixels of a w x h image."""
    area = float(w * h)
    for _, x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:              # an empty box (e.g. Drone Hunter "0 0.48 0.02 0 0" on a frame that
            return "degenerate_box"         # shows the drone): the frame's labels cannot be trusted
        if max(bw / bh, bh / bw) > float(q["max_aspect"]) or (bw * bh) / area > float(q["max_box_frac"]):
            return "degenerate_box"
        if max(bw, bh) < float(q["min_target_px"]):
            return "tiny_target"
    return None


def invisible_target(gray_full: np.ndarray, boxes: list, q: dict) -> Optional[int]:
    """Index of a labelled box inside which nothing stands out from the plain background, else None.

    Labels are often loose (a 4 px drone in a 40 px box), so the test is not on box averages but on
    the strongest deviation inside the box (after a 3x3 blur against single-pixel noise) compared
    with the background's own noise in a ring around the box. Only judged where the ring is plain
    (sky, walls): in clutter a low-contrast drone is a hard *positive*, not a label error.
    """
    h, w = gray_full.shape[:2]
    g = gray_full.astype(np.float32)
    for k, (_, x1, y1, x2, y2) in enumerate(boxes):
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        ix1, iy1, ix2, iy2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
        if ix2 - ix1 < 3 or iy2 - iy1 < 3:
            continue
        mx, my = max(3, int(bw * 0.5)), max(3, int(bh * 0.5))
        ox1, oy1, ox2, oy2 = max(0, ix1 - mx), max(0, iy1 - my), min(w, ix2 + mx), min(h, iy2 + my)
        outer = g[oy1:oy2, ox1:ox2]
        mask = np.ones(outer.shape, bool)
        mask[iy1 - oy1:iy2 - oy1, ix1 - ox1:ix2 - ox1] = False
        ring = outer[mask]
        if ring.size < 16 or float(ring.std()) > float(q["plain_ring_std"]):
            continue                                   # cluttered surroundings: do not judge
        ref = float(np.median(ring))
        ring_dev = float(np.percentile(np.abs(ring - ref), 99))
        inner = cv2.blur(g[iy1:iy2, ix1:ix2], (3, 3))
        if float(np.abs(inner - ref).max()) <= max(1.5 * ring_dev, float(q["invisible_min_dev"])):
            return k
    return None


def measure_object(gray_full: np.ndarray, box, q: dict):
    """Measure the object inside a labelled box.

    Returns ((x1, y1, x2, y2), truncated) for the object's pixels, or None if it cannot be judged.
    Looks at the box plus a margin of 3/4 of its size. A plane fitted to the ring around the box
    removes sky gradients; the ring must then be plain (sky, wall, uniform IR sky), otherwise nothing
    is measured. Object pixels are seeds clearly off the background (> fit_seed_sigma, at least
    fit_min_dev grey levels) grown into connected pixels > fit_grow_sigma, so faint edges and thin
    arms count. Only components touching the labelled box count. ``truncated`` means the object runs
    out of the examined region (then only "the label is not on it" can be judged, not looseness).
    Returns None when no object pixel is found (the invisible_target case).
    """
    h, w = gray_full.shape[:2]
    _, x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    mx, my = max(5, int(round(bw * 0.75))), max(5, int(round(bh * 0.75)))
    ox1, oy1 = max(0, int(x1) - mx), max(0, int(y1) - my)
    ox2, oy2 = min(w, int(math.ceil(x2)) + mx), min(h, int(math.ceil(y2)) + my)
    ix1, iy1 = max(0, int(x1)) - ox1, max(0, int(y1)) - oy1
    ix2, iy2 = min(w, int(math.ceil(x2))) - ox1, min(h, int(math.ceil(y2))) - oy1
    if ix2 - ix1 < 3 or iy2 - iy1 < 3:
        return None
    raw = gray_full[oy1:oy2, ox1:ox2].astype(np.float32)
    region = cv2.blur(raw, (3, 3))                    # blurred copy: background fit and seed pixels
    rh, rw = region.shape
    ring_mask = np.ones(region.shape, bool)
    ring_mask[iy1:iy2, ix1:ix2] = False
    if int(ring_mask.sum()) < 24:
        return None
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    A = np.stack([xx[ring_mask], yy[ring_mask], np.ones(int(ring_mask.sum()), np.float32)], 1)
    vals = region[ring_mask]
    use = np.ones(vals.shape, bool)
    for _ in range(3):                                # robust plane: refit without outliers (e.g. the object
        coef, *_ = np.linalg.lstsq(A[use], vals[use], rcond=None)     # itself when the label is off it)
        r = vals - A @ coef
        med = float(np.median(r[use]))
        mad = 1.4826 * float(np.median(np.abs(r[use] - med))) + 0.5
        new_use = np.abs(r - med) <= 3.0 * mad
        if new_use.sum() < 24 or np.array_equal(new_use, use):
            break
        use = new_use
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    resid = region - plane                            # background gradient removed
    resid_raw = raw - plane                           # unblurred: the object's extent is measured on this
    ring = resid[ring_mask][use]
    ref = float(np.median(ring))
    sigma = 1.4826 * float(np.median(np.abs(ring - ref))) + 0.5
    if float(use.mean()) < 0.8:
        return None                                   # > 20 % of the surroundings are not background: clutter
    if sigma > float(q.get("plain_ring_std", 6.0)):
        return None                                   # cluttered surroundings: cannot measure reliably
    dev = np.abs(resid - ref)
    raw_ring = resid_raw[ring_mask][use]
    raw_ref = float(np.median(raw_ring))
    raw_sigma = 1.4826 * float(np.median(np.abs(raw_ring - raw_ref))) + 0.5
    seed_thr = max(float(q.get("fit_seed_sigma", 4.0)) * sigma, float(q.get("fit_min_dev", 8.0)))
    grow_thr = max(float(q.get("fit_grow_sigma", 2.0)) * raw_sigma, 0.5 * float(q.get("fit_min_dev", 8.0)))
    grow = (np.abs(resid_raw - raw_ref) > grow_thr) | (dev > seed_thr)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grow.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    seeds = dev > seed_thr
    keep, keep_ids, truncated = [], [], False
    for k in range(1, n):
        cx, cy, cw, ch, area = stats[k]
        if area < 2:
            continue
        if cx + cw <= ix1 or cx >= ix2 or cy + ch <= iy1 or cy >= iy2:
            continue                                  # does not touch the labelled box
        if not seeds[lab == k].any():
            continue
        if cx == 0 or cy == 0 or cx + cw >= rw or cy + ch >= rh:
            truncated = True
        keep.append((cx, cy, cx + cw, cy + ch))
        keep_ids.append(k)
    if not keep:
        return None
    # extent: pixels of those components that reach fit_edge_frac of the object's peak contrast (its visible
    # edge), so compression ringing and halos around high-contrast objects do not inflate the measurement
    obj = np.isin(lab, [k for k in keep_ids])
    dev_raw = np.abs(resid_raw - raw_ref)
    peak = float(dev_raw[obj].max())
    strong = obj & (dev_raw >= max(grow_thr, float(q.get("fit_edge_frac", 0.3)) * peak))
    ys, xs = np.nonzero(strong)
    if xs.size == 0:
        return None
    t = (float(xs.min() + ox1), float(ys.min() + oy1), float(xs.max() + 1 + ox1), float(ys.max() + 1 + oy1))
    return t, truncated


def fit_boxes(gray_full: np.ndarray, boxes: list, q: dict) -> list:
    """[(box index, 'loose_box' | 'misplaced_box', measured object box)] for every badly fitting box.

    misplaced_box only when the object was measured completely (not truncated) and its centre lies
    outside the label: on real data, objects that touch the label but run off (towers, chimneys,
    contrails, rotor blur) are other structures, not a misplaced label.
    """
    ratio, excess = float(q.get("loose_ratio", 2.0)), float(q.get("loose_min_excess_px", 10))
    inside_min = float(q.get("misplaced_inside", 0.6))
    out = []
    for k, b in enumerate(boxes):
        m = measure_object(gray_full, b, q)
        if m is None:
            continue
        t, truncated = m
        if truncated:
            continue                                  # object bigger than the view: nothing reliable to say
        _, x1, y1, x2, y2 = b
        tw, th = t[2] - t[0], t[3] - t[1]
        lw, lh = x2 - x1, y2 - y1
        if tw <= 0 or th <= 0:
            continue
        iw = max(0.0, min(x2, t[2]) - max(x1, t[0]))
        ih = max(0.0, min(y2, t[3]) - max(y1, t[1]))
        cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
        if (iw * ih) / (tw * th) < inside_min and not (x1 <= cx <= x2 and y1 <= cy <= y2):
            if q.get("misplaced_check", False):
                out.append((k, "misplaced_box", t))
            continue                                  # measured something beside the label: do not tighten to it
        if (lw >= ratio * tw and lw - tw > excess) or (lh >= ratio * th and lh - th > excess):
            out.append((k, "loose_box", t))
    return out


def box_fit(gray_full: np.ndarray, boxes: list, q: dict):
    """(reason, box index, measured box) of the first badly fitting box (misplaced first), else None."""
    fits = fit_boxes(gray_full, boxes, q)
    fits.sort(key=lambda f: f[1] != "misplaced_box")
    return (fits[0][1], fits[0][0], fits[0][2]) if fits else None


def tighten_box(box, measured, q: dict):
    """Loose label -> measured object + margin, never larger than the original label."""
    c, x1, y1, x2, y2 = box
    t1, u1, t2, u2 = measured
    m = max(float(q.get("tighten_min_margin_px", 2)), float(q.get("tighten_margin", 0.1)) * max(t2 - t1, u2 - u1))
    nx1, ny1 = max(x1, t1 - m), max(y1, u1 - m)
    nx2, ny2 = min(x2, t2 + m), min(y2, u2 + m)
    if nx2 - nx1 < 3 or ny2 - ny1 < 3:
        return box
    return (c, nx1, ny1, nx2, ny2)


def temporal_outliers(items: list, q: dict) -> set:
    """Indices of video frames whose single box of a class jumps away from both neighbours.

    The neighbours (nearest labelled frames within 2 frames on each side, also with exactly one
    box of that class) are linearly interpolated to the frame; a smooth fast-moving target is
    therefore not flagged, a one-frame glitch is. Boxes must be in one consistent unit per group.
    """
    jump, ratio = float(q["temporal_jump"]), float(q["temporal_area_ratio"])
    by_frame = {it.frame: i for i, it in enumerate(items) if it.frame is not None and it.frame >= 0}
    if len(by_frame) < 3:
        return set()

    def single(i: int, c: str):
        bs = [b for b in items[i].boxes if b[0] == c]
        return bs[0] if len(bs) == 1 else None

    def as_xyxy(b, norm: bool):
        if not norm:
            return b[1:]
        _, cx, cy, bw, bh = b
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

    # Only for real sequences. In a video almost every pair of consecutive frames has its box in nearly the same
    # place; in a folder of independent photos numbered 1, 2, 3 ... (DUT Anti-UAV: 10 % of pairs) it does not,
    # and every photo would look like a "jump".
    near_next = {}                          # frame f -> is f+1's box near f's box
    for f, i in by_frame.items():
        j = by_frame.get(f + 1)
        if j is None or not items[i].boxes or not items[j].boxes:
            continue
        a, b = as_xyxy(items[i].boxes[0], items[i].norm), as_xyxy(items[j].boxes[0], items[j].norm)
        size = max(a[2] - a[0], a[3] - a[1], 1e-9)
        near_next[f] = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) / size < 1.0
    # a frame agrees with the clip if its box is near the previous or the next frame's (a glitch does not count
    # against its neighbours, so a short clip with one glitch still counts as a video)
    judged = [any(near_next[g] for g in (f - 1, f) if g in near_next) for f in by_frame if (f - 1) in near_next or f in near_next]
    if len(judged) >= 5 and sum(judged) / len(judged) < float(q.get("sequence_min_near", 0.8)):
        return set()

    out = set()
    for i, it in enumerate(items):
        if it.frame is None or it.frame < 0 or not it.boxes:
            continue
        for c in {b[0] for b in it.boxes}:
            cur = single(i, c)
            if cur is None:
                continue
            prev = nxt = None
            for d in (1, 2):
                if prev is None and (it.frame - d) in by_frame:
                    p = single(by_frame[it.frame - d], c)
                    prev = (d, p) if p is not None else prev
                if nxt is None and (it.frame + d) in by_frame:
                    n = single(by_frame[it.frame + d], c)
                    nxt = (d, n) if n is not None else nxt
            if prev is None or nxt is None:
                continue
            (dp, pb), (dn, nb) = prev, nxt
            a = as_xyxy(pb, it.norm)
            b = as_xyxy(nb, it.norm)
            t = dp / (dp + dn)
            ip = [a[k] + (b[k] - a[k]) * t for k in range(4)]
            cb = as_xyxy(cur, it.norm)
            size = max(ip[2] - ip[0], ip[3] - ip[1], 1e-9)
            dc = math.hypot((cb[0] + cb[2] - ip[0] - ip[2]) / 2, (cb[1] + cb[3] - ip[1] - ip[3]) / 2) / size
            a_cur = max(1e-12, (cb[2] - cb[0]) * (cb[3] - cb[1]))
            a_int = max(1e-12, (ip[2] - ip[0]) * (ip[3] - ip[1]))
            # the neighbours must agree with each other, otherwise the motion is just fast/erratic
            na = max(a[2] - a[0], a[3] - a[1], 1e-9)
            n_jump = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) / na
            if n_jump <= jump * (dp + dn) / 2 and (dc > jump or max(a_cur / a_int, a_int / a_cur) > ratio):
                out.add(i)
    # label glitches are rare in a labelled video (Halmstad: 10 in 203k frames); flagging many frames of one
    # group means its frames are not one continuous shot, so judge none of them
    if len(out) > max(2, float(q.get("max_jump_frac", 0.05)) * len(judged)):
        return set()
    return out
