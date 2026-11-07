"""Readers for the datasets in the team README (and any extra YOLO / VOC / COCO folder).

Every reader yields :class:`Group` objects. A group is one video, sequence or folder of
consecutive frames, and it is the unit of the train/val split, so near-identical frames
of one clip never end up on both sides of the split.

Box convention inside this package: ``(cls, x1, y1, x2, y2)`` in pixels, 0-based pixel
edges (the left edge of pixel column 0 is x = 0), or YOLO-normalised
``(cls, cx, cy, w, h)`` when ``Item.norm`` is True.

Datasets with an exact parser:
    halmstad         Video_IR / Video_V *.mp4 + *_LABELS.mat (MATLAB groundTruth)
    mav_vid          {train,val}/img/<seq>_<frame>.jpg + YOLO .txt
    drone_vs_bird    <clip>.txt annotations (framenum n x y w h cls ...) + <clip>.<video>
    anti_uav         <seq>/{infrared,visible}.{mp4,json} or IR/RGB(_label.json), "exist" + "gt_rect"
    roboflow_indoor  Roboflow "YOLOv8" export: {train,valid,test}/{images,labels} + data.yaml
Datasets read by the auto-detecting reader (their label formats are not documented):
    dut_anti_uav, det_fly, drone_hunter, team, and any other folder (VOC XML, COCO JSON,
    LabelMe JSON, YOLO txt, CSV).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .schema import CLASSES, normalize_label

log = logging.getLogger(__name__)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv", ".mts"}
_SPLIT_TOKENS = {"train": "train", "training": "train", "trainval": "train",
                 "val": "val", "valid": "val", "validation": "val", "dev": "val",
                 "test": "test", "testing": "test"}


@dataclass
class Item:
    boxes: list                       # [(cls, x1, y1, x2, y2)] px, or [(cls, cx, cy, w, h)] normalised if norm
    image: Optional[Path] = None      # still image (None -> frame of the group's video)
    frame: int = -1                   # frame index in the group's video / sequence
    norm: bool = False
    name: str = ""                    # output file stem, unique within the group


@dataclass
class Group:
    source: str
    name: str
    items: list
    video: Optional[Path] = None      # items are frames of this video
    split: Optional[str] = None       # 'train' | 'val' | 'test' when the dataset defines it
    modality: str = "rgb"             # rgb | ir
    notes: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers
def split_from_path(rel: Path) -> Optional[str]:
    """Official split from folder names (shallowest match wins): .../train/..., .../valid/..."""
    for part in rel.parts[:-1] if rel.suffix else rel.parts:
        low = part.lower()
        if low in _SPLIT_TOKENS:
            return _SPLIT_TOKENS[low]
        for tok in re.split(r"[^a-z0-9]+", low):
            if tok in _SPLIT_TOKENS and tok != "dev":
                return _SPLIT_TOKENS[tok]
    return None


def iter_files(root: Path, exts: set) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts and p.is_file() and not any(s.startswith(".") for s in p.relative_to(root).parts):
            yield p


def _num_suffix(stem: str) -> tuple[str, Optional[int]]:
    m = re.match(r"^(.*?)(\d+)$", stem)
    if not m:
        return stem, None
    return m.group(1), int(m.group(2))


def sequence_key(path: Path, root: Path, block: int = 100) -> str:
    """Group still images that are probably consecutive frames of one clip.

    'clipA_000123.jpg' -> 'clipA_'; bare numbers ('000123.jpg') -> '<folder>/#1' (blocks of 100).
    """
    stem = re.sub(r"\.rf\.[0-9a-f]{8,}$", "", path.stem)            # Roboflow export suffix
    stem = re.sub(r"_(jpg|jpeg|png|bmp)$", "", stem, flags=re.I)
    # copies a dataset made of its own frames ('GOPR3121_10_1002_blur' in Drone Hunter) belong to the
    # original's clip: otherwise each copy is a one-frame "clip" and lands in val next to its original
    stem = re.sub(r"[_-](blur|blurred|motionblur|aug|augmented|noise|noisy|flip|flipped|mirror)$", "", stem, flags=re.I)
    prefix, num = _num_suffix(stem)
    prefix = prefix.rstrip("_- .")
    folder = path.parent.relative_to(root).as_posix() if path.parent != root else ""
    if prefix:
        return f"{folder}/{prefix}" if folder else prefix
    return f"{folder}/#{(num or 0) // block}"


def guess_box_format(rects: Iterable) -> str:
    """'xywh' or 'xyxy' for a list of 4-number boxes (Anti-UAV's README and paper disagree)."""
    vals = [r for r in rects if r and len(r) == 4 and sum(abs(v) for v in r) > 0]
    if not vals:
        return "xywh"
    if any(r[2] <= r[0] or r[3] <= r[1] for r in vals):
        return "xywh"
    return "xyxy"


def to_xyxy(r, fmt: str) -> tuple[float, float, float, float]:
    x, y, a, b = (float(v) for v in r[:4])
    return (x, y, a, b) if fmt == "xyxy" else (x, y, x + a, y + b)


def read_names_file(folder: Path) -> Optional[list[str]]:
    """Class names from data.yaml / dataset.yaml / classes.txt / obj.names near a YOLO dataset."""
    for cand in ("data.yaml", "dataset.yaml", "data.yml"):
        p = folder / cand
        if p.exists():
            try:
                import yaml

                d = yaml.safe_load(p.read_text(encoding="utf-8", errors="replace")) or {}
                names = d.get("names")
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
                if isinstance(names, list):
                    return [str(n) for n in names]
            except Exception:
                pass
    for cand in ("classes.txt", "obj.names", "labels.txt", "_darknet.labels"):
        p = folder / cand
        if p.exists():
            return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    return None


def find_names(start: Path, root: Path) -> Optional[list[str]]:
    p = start
    while True:
        n = read_names_file(p)
        if n:
            return n
        if p == root or p.parent == p or root not in p.parents and p != root:
            return None
        p = p.parent


def read_yolo_txt(path: Path, names: Optional[list], cmap: dict, default: Optional[str]) -> Optional[list]:
    """YOLO label file -> normalised boxes; None if the file is not YOLO-normalised."""
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for ln in text.splitlines():
        t = ln.replace(",", " ").split()
        if not t:
            continue
        try:
            vals = [float(v) for v in t]
        except ValueError:
            return None
        if len(vals) == 5:
            k, cx, cy, w, h = vals
        elif len(vals) >= 7 and len(vals) % 2 == 1:            # segmentation polygon -> box
            k, xs, ys = vals[0], vals[1::2], vals[2::2]
            cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        else:
            return None
        if not (0 <= cx <= 1.001 and 0 <= cy <= 1.001 and 0 <= w <= 1.001 and 0 <= h <= 1.001):
            return None
        raw = names[int(k)] if names and 0 <= int(k) < len(names) else str(int(k))
        cls = normalize_label(raw, cmap, default)
        if cls:
            out.append((cls, cx, cy, w, h))
    return out


def read_pixel_txt(path: Path, fmt: str, cmap: dict, default: Optional[str]) -> Optional[list]:
    """Per-image txt with pixel boxes, one per line: [class] x y w h (fmt xywh) or x1 y1 x2 y2 (fmt xyxy).

    The class may be a leading integer id / name or a trailing name; separators can be spaces or commas.
    Only used when a source sets ``txt_format`` (the layout cannot be told apart from YOLO reliably).
    """
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for ln in text.splitlines():
        t = ln.replace(",", " ").replace("\t", " ").split()
        if not t:
            continue
        nums, words = [], []
        for v in t:
            try:
                nums.append(float(v))
            except ValueError:
                words.append(v)
        label = words[0] if words else None
        if len(nums) >= 5 and not words and float(nums[0]).is_integer():
            label, nums = str(int(nums[0])), nums[1:]
        if len(nums) < 4:
            return None
        a, b, c, d = nums[:4]
        x1, y1, x2, y2 = (a, b, a + c, b + d) if fmt == "xywh" else (a, b, c, d)
        cls = normalize_label(label, cmap, default)
        if cls and x2 > x1 and y2 > y1:
            out.append((cls, x1, y1, x2, y2))
    return out


class Source:
    key = ""
    title = ""
    homepage = ""
    license = ""
    access = ""          # how the raw data is obtained (shown by tools/fetch_datasets.py --list)
    reader = "exact"

    def present(self, root: Path) -> bool:
        return root.exists() and any(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT | VID_EXT)

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- Halmstad
class Halmstad(Source):
    key = "halmstad"
    title = "Halmstad multi-sensor drone dataset (IR + visible, with bird / airplane / helicopter)"
    homepage = "https://github.com/DroneDetectionThesis/Drone-detection-dataset"
    license = "CC0 1.0"
    access = "auto (GitHub)"

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        from .mat5 import load_groundtruth

        mods = set(opts.get("modalities") or ["rgb", "ir"])
        cmap = opts.get("class_map") or {}
        for mat in iter_files(root, {".mat"}):
            if not mat.name.upper().endswith("_LABELS.MAT"):
                continue
            stem = mat.name[: -len("_LABELS.mat")]
            modality = "ir" if stem.upper().startswith("IR_") else "rgb"
            if modality not in mods:
                continue
            video = next((mat.with_name(stem + e) for e in (".mp4", ".MP4", ".avi") if mat.with_name(stem + e).exists()), None)
            if video is None:
                log.warning("halmstad: no video for %s", mat.name)
                continue
            try:
                gt = load_groundtruth(mat)
            except Exception as exc:
                log.warning("halmstad: cannot read %s (%s)", mat.name, exc)
                continue
            items = []
            for i in range(len(gt.frames)):
                boxes = [(c, x1, y1, x2, y2) for lab, x1, y1, x2, y2 in gt.boxes_xyxy(i)
                         if (c := normalize_label(lab, cmap))]
                items.append(Item(boxes, frame=i))
            yield Group(self.key, stem, items, video=video, modality=modality,
                        notes={"target": stem.split("_")[1].lower() if "_" in stem else ""})


# ----------------------------------------------------------------------------- MAV-VID
class MavVid(Source):
    key = "mav_vid"
    title = "MAV-VID (Multirotor Aerial Vehicle VID)"
    homepage = "https://bitbucket.org/alejodosr/mav-vid-dataset"
    license = "see dataset page"
    access = "auto (Bitbucket mirror) or manual"

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        cmap = opts.get("class_map") or {}
        default = opts.get("default_class", "drone")
        groups: dict = defaultdict(list)
        names_cache: dict = {}
        for img in iter_files(root, IMG_EXT):
            txt = img.with_suffix(".txt")
            if not txt.exists():
                continue
            if img.parent not in names_cache:
                names_cache[img.parent] = find_names(img.parent, root)
            boxes = read_yolo_txt(txt, names_cache[img.parent], cmap, default)
            if boxes is None:
                continue
            rel = img.relative_to(root)
            prefix, num = (img.stem.rsplit("_", 1) + [""])[:2] if "_" in img.stem else (img.stem, "")
            frame = int(num) if num.isdigit() else -1
            groups[(split_from_path(rel), img.parent, prefix)].append(Item(boxes, image=img, frame=frame, norm=True,
                                                                          name=img.stem))
        for (split, parent, prefix), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2])):
            items.sort(key=lambda it: (it.frame, it.name))
            yield Group(self.key, prefix, items, split=split)


# ----------------------------------------------------------------------------- Drone-vs-Bird
class DroneVsBird(Source):
    key = "drone_vs_bird"
    title = "Drone-vs-Bird detection challenge (WOSDETC)"
    homepage = "https://github.com/wosdetc/challenge"
    license = "data usage agreement (research)"
    access = "annotations auto (GitHub); videos by e-mail request"

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        cmap = opts.get("class_map") or {}
        videos = {}
        for v in iter_files(root, VID_EXT):
            videos.setdefault(v.stem.lower(), v)
        missing = []
        for txt in iter_files(root, {".txt"}):
            items = self._parse(txt, cmap)
            if items is None:
                continue
            video = videos.get(txt.stem.lower())
            if video is None:
                missing.append(txt.stem)
                continue
            yield Group(self.key, txt.stem, items, video=video)
        if missing:
            log.warning("drone_vs_bird: %d annotation files have no matching video yet (videos come by request "
                        "from wosdetc@googlegroups.com): %s%s", len(missing), ", ".join(missing[:5]),
                        " ..." if len(missing) > 5 else "")

    @staticmethod
    def _parse(txt: Path, cmap: dict) -> Optional[list]:
        frames: dict[int, list] = {}
        for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            t = ln.split()
            if not t:
                continue
            try:
                fr, n = int(t[0]), int(t[1])
            except (ValueError, IndexError):
                return None
            boxes = []
            for k in range(n):
                f = t[2 + 5 * k: 7 + 5 * k]
                if len(f) < 4:
                    return None
                x, y, w, h = (float(v) for v in f[:4])
                c = normalize_label(f[4] if len(f) > 4 else "drone", cmap, "drone")
                if c and w > 0 and h > 0:
                    boxes.append((c, x, y, x + w, y + h))
            frames[fr] = boxes
        if not frames:
            return None
        return [Item(frames[k], frame=k) for k in sorted(frames)]


# ----------------------------------------------------------------------------- Anti-UAV
_ANTI_UAV_LABELS = {"infrared.json": ("ir", ("infrared", "ir")), "visible.json": ("rgb", ("visible", "rgb")),
                    "ir_label.json": ("ir", ("ir", "infrared")), "rgb_label.json": ("rgb", ("rgb", "visible"))}


class AntiUav(Source):
    key = "anti_uav"
    title = "Anti-UAV (RGB + thermal IR videos; Anti-UAV300 / CVPR20 / 410)"
    homepage = "https://github.com/ucas-vg/Anti-UAV"
    license = "research use (see project page)"
    access = "auto (Google Drive via gdown) or manual"

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        mods = set(opts.get("modalities") or ["rgb", "ir"])
        forced = opts.get("box_format")
        for js in iter_files(root, {".json"}):
            spec = _ANTI_UAV_LABELS.get(js.name.lower())
            if spec is None:
                continue
            modality, stems = spec
            if modality not in mods:
                continue
            try:
                data = json.loads(js.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                log.warning("anti_uav: %s unreadable (%s)", js, exc)
                continue
            rects = data.get("gt_rect") or data.get("gt_rects") or []
            exist = data.get("exist") or []
            seq = js.parent
            video, frames = self._media(seq, stems)
            if video is None and not frames:
                log.warning("anti_uav: no video or frames next to %s", js)
                continue
            fmt = forced or guess_box_format(rects)
            items = []
            n = len(rects) if video is not None else min(len(rects), len(frames))
            for i in range(n):
                r = rects[i]
                absent = i < len(exist) and not exist[i]
                boxes = []
                if not absent and r and len(r) == 4 and sum(abs(float(v)) for v in r) > 0:
                    x1, y1, x2, y2 = to_xyxy(r, fmt)
                    if x2 > x1 and y2 > y1:
                        boxes = [("drone", x1, y1, x2, y2)]
                items.append(Item(boxes, image=None if video is not None else frames[i], frame=i))
            yield Group(self.key, f"{seq.name}_{modality}", items, video=video,
                        split=split_from_path(js.relative_to(root)), modality=modality, notes={"box_format": fmt})

    @staticmethod
    def _media(seq: Path, stems) -> tuple[Optional[Path], list]:
        for p in seq.iterdir():
            if p.is_file() and p.suffix.lower() in VID_EXT and p.stem.lower() in stems:
                return p, []
        for sub in [seq / s for s in stems] + [seq / s.upper() for s in stems] + [seq / "img", seq]:
            if sub.is_dir():
                imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMG_EXT)
                if imgs:
                    return None, imgs
        return None, []


# ----------------------------------------------------------------------------- YOLO-format datasets
class YoloFolder(Source):
    """A YOLO dataset (Roboflow export, the team's own labelled frames, ...)."""

    key = "yolo"
    title = "YOLO-format folder"
    access = "manual"

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        cmap = opts.get("class_map") or {}
        default = opts.get("default_class")
        names = opts.get("names") or find_names(root, root)
        groups: dict = defaultdict(list)
        n_unl = 0
        for img in iter_files(root, IMG_EXT):
            lab = self._label_path(img)
            if lab is None:
                n_unl += 1
                if not opts.get("unlabeled_as_negative", False):
                    continue
                boxes: list = []
            else:
                nm = names or find_names(img.parent, root)
                boxes = read_yolo_txt(lab, nm, cmap, default)
                if boxes is None:
                    log.warning("%s: %s is not a YOLO label file, skipped", self.key, lab)
                    continue
            rel = img.relative_to(root)
            key = sequence_key(img, root)
            prefix, num = _num_suffix(re.sub(r"\.rf\.[0-9a-f]{8,}$", "", img.stem))
            groups[(split_from_path(rel), key)].append(Item(boxes, image=img, frame=num if num is not None else -1,
                                                            norm=True, name=img.stem))
        if n_unl:
            log.info("%s: %d images without a label file %s", self.key, n_unl,
                     "kept as negatives" if opts.get("unlabeled_as_negative") else "skipped")
        for (split, key), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            items.sort(key=lambda it: (it.frame, it.name))
            yield Group(self.key, key.replace("/", "_").strip("_#") or "root", items, split=split)

    @staticmethod
    def _label_path(img: Path) -> Optional[Path]:
        cands = [img.with_suffix(".txt")]
        parts = list(img.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].lower() == "images":
                cands.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
                break
        for c in cands:
            if c.exists():
                return c
        return None


class RoboflowIndoor(YoloFolder):
    key = "roboflow_indoor"
    title = "Roboflow Universe: Indoor-drones (drones-l7ol3/indoor-drones-2ughv)"
    homepage = "https://universe.roboflow.com/drones-l7ol3/indoor-drones-2ughv"
    license = "CC BY 4.0"
    access = "auto with a free Roboflow API key"


class Team(YoloFolder):
    key = "team"
    title = "Team-recorded frames (Blackfly / webcam), labelled in YOLO format"
    homepage = "datasets/aimbot.yaml (class ids)"
    license = "team"
    access = "record + label yourselves (tools/export_frames.py)"

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        o = dict(opts)
        o.setdefault("names", CLASSES)
        yield from super().groups(root, o)


# ----------------------------------------------------------------------------- auto-detecting reader
class Generic(Source):
    """Reads VOC XML, COCO JSON, LabelMe JSON, CSV and YOLO txt annotations, whatever the folder layout."""

    key = "generic"
    title = "Any folder of annotated images"
    reader = "auto-detect"
    access = "manual"

    def groups(self, root: Path, opts: dict) -> Iterator[Group]:
        cmap = opts.get("class_map") or {}
        default = opts.get("default_class")
        images = list(iter_files(root, IMG_EXT))
        by_stem: dict[str, list[Path]] = defaultdict(list)
        by_rel = {}
        for p in images:
            by_stem[p.stem.lower()].append(p)
            by_rel[p.relative_to(root).as_posix().lower()] = p
        ann: dict[Path, list] = {}
        norm: dict[Path, bool] = {}
        fmt_counts: dict[str, int] = defaultdict(int)

        def resolve(name: str, near: Path) -> Optional[Path]:
            if not name:
                return None
            name = name.replace("\\", "/")
            for cand in (near / name, root / name):
                if cand.exists() and cand.suffix.lower() in IMG_EXT:
                    return cand
            hits = by_stem.get(Path(name).stem.lower(), [])
            if len(hits) <= 1:
                return hits[0] if hits else None

            def shared(h: Path) -> int:                     # folders in common with the annotation file
                n = 0
                for a, b in zip(h.parent.parts, near.parts):
                    if a != b:
                        break
                    n += 1
                return n

            # e.g. train/xml/00001.xml -> train/img/00001.jpg, not val/img/00001.jpg
            return max(hits, key=lambda h: (shared(h), h.parent.name == near.name))

        def add(img: Optional[Path], boxes: list, is_norm: bool, fmt: str):
            if img is None:
                return
            ann.setdefault(img, []).extend(boxes)
            norm[img] = is_norm
            fmt_counts[fmt] += 1

        # COCO / LabelMe JSON
        for js in iter_files(root, {".json"}):
            try:
                d = json.loads(js.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if isinstance(d, dict) and "images" in d and "annotations" in d:
                cats = {c.get("id"): c.get("name") for c in d.get("categories", [])}
                imgs = {im.get("id"): im for im in d.get("images", [])}
                per: dict = defaultdict(list)
                for a in d.get("annotations", []):
                    if a.get("iscrowd"):
                        continue
                    bb = a.get("bbox") or []
                    c = normalize_label(cats.get(a.get("category_id"), a.get("category_id")), cmap, default)
                    if c and len(bb) == 4 and bb[2] > 0 and bb[3] > 0:
                        per[a.get("image_id")].append((c, bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
                for iid, im in imgs.items():
                    add(resolve(str(im.get("file_name", "")), js.parent), per.get(iid, []), False, "coco")
            elif isinstance(d, dict) and "shapes" in d and ("imagePath" in d or "imageData" in d):
                boxes = []
                for s in d.get("shapes", []):
                    pts = s.get("points") or []
                    c = normalize_label(s.get("label"), cmap, default)
                    if c and len(pts) >= 2:
                        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                        boxes.append((c, min(xs), min(ys), max(xs), max(ys)))
                img = resolve(str(d.get("imagePath", "")), js.parent) or resolve(js.stem, js.parent)
                add(img, boxes, False, "labelme")
        # Pascal VOC XML
        for xp in iter_files(root, {".xml"}):
            try:
                r = ET.parse(xp).getroot()
            except Exception:
                continue
            if r.tag != "annotation":
                continue
            boxes = []
            for o in r.findall("object"):
                bb = o.find("bndbox")
                c = normalize_label((o.findtext("name") or "").strip(), cmap, default)
                if bb is None or not c:
                    continue
                try:
                    x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                except (TypeError, ValueError):
                    continue
                boxes.append((c, x1, y1, x2, y2))
            img = resolve(xp.stem, xp.parent) or resolve(r.findtext("filename") or "", xp.parent)
            add(img, boxes, False, "voc")
        # CSV (one row per box)
        for cp in iter_files(root, {".csv"}):
            for img, boxes in self._csv(cp, resolve, cmap, default).items():
                add(img, boxes, False, "csv")
        # YOLO txt for anything still unlabelled
        names_cache: dict = {}
        for img in images:
            if img in ann:
                continue
            lab = YoloFolder._label_path(img)
            if lab is None:
                continue
            if img.parent not in names_cache:
                names_cache[img.parent] = opts.get("names") or find_names(img.parent, root)
            boxes = read_yolo_txt(lab, names_cache[img.parent], cmap, default)
            if boxes is not None and not opts.get("txt_format"):
                add(img, boxes, True, "yolo")
            elif opts.get("txt_format") in ("xywh", "xyxy"):
                pb = read_pixel_txt(lab, str(opts["txt_format"]), cmap, default)
                if pb is not None:
                    add(img, pb, False, f"txt_{opts['txt_format']}")
        if opts.get("unlabeled_as_negative"):
            for img in images:
                if img not in ann:
                    add(img, [], False, "negative")
        log.info("%s: %d of %d images annotated (%s)", self.key, len(ann), len(images),
                 ", ".join(f"{k} {v}" for k, v in sorted(fmt_counts.items())) or "no annotations recognised")
        groups: dict = defaultdict(list)
        use_folders = opts.get("official_split", True)   # false: folder names only look like a split
        for img, boxes in ann.items():
            rel = img.relative_to(root)
            _, num = _num_suffix(img.stem)
            groups[(split_from_path(rel) if use_folders else None, sequence_key(img, root))].append(
                Item(boxes, image=img, frame=num if num is not None else -1, norm=norm.get(img, False), name=img.stem))
        for (split, key), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            items.sort(key=lambda it: (it.frame, it.name))
            yield Group(self.key, key.replace("/", "_").strip("_#") or "root", items, split=split,
                        notes={"formats": dict(fmt_counts)})

    @staticmethod
    def _csv(cp: Path, resolve, cmap, default) -> dict:
        out: dict = defaultdict(list)
        try:
            with cp.open(newline="", encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return out
        if not rows:
            return out
        cols = {c.lower().strip(): c for c in rows[0].keys() if c}
        fcol = next((cols[k] for k in ("filename", "file_name", "image", "image_path", "path", "img", "name") if k in cols), None)
        ccol = next((cols[k] for k in ("class", "label", "category", "class_name", "name") if k in cols and cols[k] != fcol), None)
        xyxy = all(k in cols for k in ("xmin", "ymin", "xmax", "ymax"))
        xywh = all(k in cols for k in ("x", "y", "w", "h")) or all(k in cols for k in ("x", "y", "width", "height"))
        if fcol is None or not (xyxy or xywh):
            return out
        for r in rows:
            try:
                if xyxy:
                    x1, y1, x2, y2 = (float(r[cols[k]]) for k in ("xmin", "ymin", "xmax", "ymax"))
                else:
                    wk = "w" if "w" in cols else "width"
                    hk = "h" if "h" in cols else "height"
                    x1, y1 = float(r[cols["x"]]), float(r[cols["y"]])
                    x2, y2 = x1 + float(r[cols[wk]]), y1 + float(r[cols[hk]])
            except (TypeError, ValueError):
                continue
            c = normalize_label(r.get(ccol) if ccol else None, cmap, default)
            img = resolve(str(r[fcol]), cp.parent)
            if img is not None and c:
                out[img].append((c, x1, y1, x2, y2))
        return out


class DutAntiUav(Generic):
    key = "dut_anti_uav"
    title = "DUT Anti-UAV detection set (train / val / test, 10,000 images)"
    homepage = "https://github.com/wangdongdut/DUT-Anti-UAV"
    license = "research use (see repository)"
    access = "auto (Google Drive via gdown) or manual (Baidu)"


class DetFly(Generic):
    key = "det_fly"
    title = "Det-Fly air-to-air micro-UAV dataset (13,271 images)"
    homepage = "https://github.com/Jake-WU/Det-Fly"
    license = "academic use (see repository)"
    access = "manual (OneDrive / Baidu links in the repository)"


class DroneHunter(Generic):
    key = "drone_hunter"
    title = "Autonomous drone hunter dataset (Wyder et al., PLOS ONE 2019; indoor + outdoor, 58,647 images)"
    homepage = "https://osf.io/n9q78/"
    license = "see OSF project"
    access = "auto (OSF API)"


SOURCES: dict[str, Source] = {s.key: s for s in (
    Halmstad(), MavVid(), DroneHunter(), DutAntiUav(), RoboflowIndoor(), DetFly(), AntiUav(), DroneVsBird(), Team(),
)}
# extra folders listed in datasets/sources.yaml with "reader: yolo|generic|anti_uav|..."
READERS: dict[str, type] = {"generic": Generic, "yolo": YoloFolder, "halmstad": Halmstad, "mav_vid": MavVid,
                            "anti_uav": AntiUav, "drone_vs_bird": DroneVsBird}


def get_source(key: str, opts: dict) -> Source:
    if key in SOURCES and not opts.get("reader"):
        return SOURCES[key]
    reader = str(opts.get("reader") or "generic")
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r} for source {key!r} (choose {', '.join(READERS)})")
    src = READERS[reader]()
    src.key = key
    src.title = opts.get("title", key)
    return src


def box_stats(sizes: list) -> dict:
    if not sizes:
        return {}
    ws = sorted(s[0] for s in sizes)
    hs = sorted(s[1] for s in sizes)
    return {"median_w": round(statistics.median(ws), 1), "median_h": round(statistics.median(hs), 1),
            "p10_w": round(ws[len(ws) // 10], 1), "p90_w": round(ws[(len(ws) * 9) // 10], 1)}
