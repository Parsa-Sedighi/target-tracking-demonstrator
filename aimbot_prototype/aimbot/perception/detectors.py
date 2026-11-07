"""Macro detectors (FR-01): YOLO26 (Ultralytics, .pt / TensorRT .engine / .onnx), a colour
blob detector for tennis-ball tests without ML, and a simulator 'truth' detector.

All detectors return full-frame pixel boxes and map the detector's own class name
to the operator's *logical* class (e.g. COCO "sports ball" -> "ball").

YOLO setup used by the configs:
  * ``model``: the AimBot YOLO26 model trained on the team's drone datasets
    (tools/train_yolo26.py -> models/aimbot_yolo26n.pt): drone, bird, airplane, helicopter.
  * ``fallback_model``: stock COCO yolo26n, used while the trained model does not exist yet.
    COCO has no drone class, so for that model only, airplane / bird / kite count as "drone".
  * ``assist_model`` + ``assist_classes``: optional second (COCO) model that adds the classes
    the main model does not have, e.g. "sports ball" for the ball-drop test. While tracking
    with a region of interest, only the model that can produce the locked class is run.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import PACKAGE_ROOT
from ..types import Detection, Frame

log = logging.getLogger(__name__)

# Default mapping from COCO / custom-model names to the operator's logical classes.
DEFAULT_CLASS_MAP = {
    "sports ball": "ball", "orange": "ball", "apple": "ball", "tennis ball": "ball", "ball": "ball",
    "drone": "drone", "quadcopter": "drone", "uav": "drone",
    "airplane": "airplane", "bird": "bird", "helicopter": "helicopter",
    "card": "card", "index card": "card", "book": "card",
    "hand": "hand", "person": "person", "cell phone": "phone",
}
# Only for models WITHOUT a drone class (stock COCO): a drone is usually detected as one of these.
COCO_DRONE_PROXIES = {"airplane": "drone", "kite": "drone", "bird": "drone"}
NATIVE_DRONE_NAMES = {"drone", "quadcopter", "uav", "multirotor"}


def _lower_keys(cmap: dict) -> dict:
    return {str(k).strip().lower(): v for k, v in cmap.items()}


def map_class(cmap: dict, lower: dict, raw: str) -> str:
    """Logical class of a model's class name. An exact class_map entry wins; otherwise the name is
    matched ignoring case and spaces at the ends, so a model trained with "UAV" or "Drone" still
    gives "drone". Names that are not in the map are kept as they are."""
    if raw in cmap:
        return cmap[raw]
    return lower.get(str(raw).strip().lower(), raw)


class DetectorBase:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        cmap = dict(DEFAULT_CLASS_MAP)
        cmap.update(self.cfg.get("class_map", {}) or {})
        self.class_map = cmap
        self._class_map_lower = _lower_keys(cmap)
        self.last_infer_ms = 0.0
        self.focus: Optional[str] = None   # logical class of the locked target while an ROI is used

    def logical(self, raw: str) -> str:
        return map_class(self.class_map, self._class_map_lower, raw)

    def detect(self, frame: Frame, roi=None) -> list[Detection]:
        raise NotImplementedError

    def warmup(self, shape=(480, 640, 3)) -> None:
        pass

    def info(self) -> dict:
        return {"name": self.name, "infer_ms": round(self.last_infer_ms, 2)}


def _crop(img: np.ndarray, roi):
    if roi is None:
        return img, 0, 0
    x1, y1, x2, y2 = (int(round(v)) for v in roi)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return np.ascontiguousarray(img[y1:y2, x1:x2]), x1, y1


def resolve_model_path(name: str) -> tuple[Path, bool]:
    """(path, found). Looks in the working directory, the project root and models/."""
    p = Path(os.path.expanduser(str(name)))
    if p.is_absolute():
        return p, p.exists()
    for cand in (Path.cwd() / p, PACKAGE_ROOT / p, PACKAGE_ROOT / "models" / p.name):
        if cand.exists():
            return cand, True
    return PACKAGE_ROOT / "models" / p.name, False


def _as_list(v) -> list:
    if not v:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _downloadable(name: str) -> bool:
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        return Path(name).name in GITHUB_ASSETS_NAMES
    except Exception:
        return False


class _YoloModel:
    """One Ultralytics model plus its class filter and name -> logical-class map."""

    def __init__(self, path: Path, device: str, prec: dict, base_map: dict, keep_raw=None, exclude_logical=None,
                 drone_proxies: bool = True, user_keys=()):
        from ultralytics import YOLO

        self.path = path
        self.yolo = YOLO(str(path), task="detect")
        names = self.yolo.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        raw_lower = {str(n).lower() for n in self.names.values()}
        self.native_drone = bool(raw_lower & NATIVE_DRONE_NAMES)
        cmap = dict(base_map)
        if not self.native_drone and drone_proxies:    # stock COCO: a drone shows up as airplane / kite / bird
            for k, v in COCO_DRONE_PROXIES.items():
                if k not in user_keys:                 # an explicit detector.class_map entry wins
                    cmap[k] = v
        self.class_map = cmap
        self._class_map_lower = _lower_keys(cmap)
        ids = list(self.names)
        if keep_raw:
            want = {str(n) for n in keep_raw}
            ids = [i for i in ids if self.names[i] in want]
        if exclude_logical:
            ids = [i for i in ids if self.logical(self.names[i]) not in exclude_logical]
        self.class_ids = None if len(ids) == len(self.names) else ids
        self.logical_set = {self.logical(self.names[i]) for i in ids}
        self.device = device
        self.prec = prec if path.suffix == ".pt" else {}

    def logical(self, raw: str) -> str:
        return map_class(self.class_map, self._class_map_lower, raw)

    def predict(self, img: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int):
        return self.yolo.predict(img, imgsz=imgsz, conf=conf, iou=iou, classes=self.class_ids, device=self.device,
                                 verbose=False, max_det=max_det, **self.prec)


class YoloDetector(DetectorBase):
    """YOLO26 (or any Ultralytics detector) with optional COCO fallback and assist models."""

    name = "yolo"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        try:
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed (pip install ultralytics)") from exc
        dev = str(self.cfg.get("device", "auto"))
        self.cuda = False
        try:
            import torch

            self.cuda = bool(torch.cuda.is_available())
        except Exception:
            pass
        if dev == "auto":
            dev = "0" if self.cuda else "cpu"
        self.device = dev
        want_half = bool(self.cfg.get("half", True)) and self.cuda and dev != "cpu"
        # FP16 switch: Ultralytics >= 8.4 renamed predict(half=True) to quantize=16; older versions only know half
        prec: dict = {}
        if want_half:
            try:
                from ultralytics.cfg import DEFAULT_CFG_DICT

                prec = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
            except Exception:
                prec = {"half": True}
        self.imgsz = int(self.cfg.get("imgsz", 640))
        self.roi_imgsz = int(self.cfg.get("roi_imgsz", 320))
        self.conf = float(self.cfg.get("conf", 0.25))
        self.iou = float(self.cfg.get("iou", 0.5))
        self.max_det = int(self.cfg.get("max_det", 50))
        user_map = dict(DEFAULT_CLASS_MAP)
        user_map.update(self.cfg.get("class_map", {}) or {})
        mk = dict(drone_proxies=bool(self.cfg.get("coco_drone_proxies", True)),
                  user_keys=set((self.cfg.get("class_map", {}) or {}).keys()))

        self.fallback_used = ""
        main_name = str(self.cfg.get("model", "yolo26n.pt"))
        chain = [main_name] + [str(f) for f in _as_list(self.cfg.get("fallback_model")) if str(f) != main_name]
        path, self.main, problems = None, None, []
        for name in chain:
            cand, found = resolve_model_path(name)
            if not (found or _downloadable(name)):
                problems.append(f"{Path(name).name} not found")
                continue
            if not found:
                (PACKAGE_ROOT / "models").mkdir(exist_ok=True)       # Ultralytics downloads official weights here
            try:
                self.main = _YoloModel(cand, dev, prec, user_map, keep_raw=self.cfg.get("classes"), **mk)
            except Exception as exc:          # e.g. an engine built for another TensorRT version, a truncated file
                problems.append(f"{cand.name} failed to load ({exc})")
                log.warning("detector model %s failed to load: %s", cand, exc)
                continue
            path = cand
            break
        if self.main is None:
            raise FileNotFoundError(f"no detector model could be loaded: {'; '.join(problems)}")
        if problems:
            hint = (" - train it with tools/train_yolo26.py" if not main_name.endswith(".engine")
                    else " - build it with tools/export_tensorrt.py")
            self.fallback_used = f"{problems[0]}{hint}; using {path.name}"
            log.warning(self.fallback_used)
        self.models = [self.main]
        self.assist = None
        assist_classes = self.cfg.get("assist_classes") or []
        for name in (_as_list(self.cfg.get("assist_model")) if assist_classes else []):
            apath, afound = resolve_model_path(str(name))
            if not (afound or _downloadable(str(name))):
                continue
            if apath.resolve() == path.resolve():
                break                                   # the main model already is the COCO model
            try:
                if not afound:
                    (PACKAGE_ROOT / "models").mkdir(exist_ok=True)
                m = _YoloModel(apath, dev, prec, user_map, keep_raw=assist_classes,
                               exclude_logical=self.main.logical_set, **mk)
            except Exception as exc:                    # the assist model is optional
                log.warning("assist model %s unavailable: %s", apath.name, exc)
                continue
            if m.class_ids:
                self.assist = m
                self.models.append(m)
            break
        self.model = self.main.yolo                     # compatibility
        self.model_path = str(path)
        self.meta: dict = {}                            # written by tools/train_yolo26.py next to the weights
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                import json

                self.meta = json.loads(meta_path.read_text())
            except Exception:
                self.meta = {}
        self.names = self.main.names
        self.half = bool(prec)
        self.class_ids = self.main.class_ids
        log.info("YOLO %s on %s (fp16=%s), imgsz %d, classes %s%s", path.name, self.device, self.half, self.imgsz,
                 sorted(self.main.logical_set),
                 f"; assist {self.assist.path.name} for {sorted(self.assist.logical_set)}" if self.assist else "")

    def logical(self, raw: str) -> str:
        return self.main.logical(raw)

    def warmup(self, shape=(480, 640, 3)) -> None:
        dummy = np.zeros(shape, np.uint8)
        for m in self.models:
            for _ in range(2):
                m.predict(dummy, self.imgsz, self.conf, self.iou, self.max_det)

    def _predict(self, img: np.ndarray, imgsz: int):
        return self.main.predict(img, imgsz, self.conf, self.iou, self.max_det)

    def detect(self, frame: Frame, roi=None) -> list[Detection]:
        img, ox, oy = _crop(frame.image, roi)
        if img.size == 0:
            return []
        models = self.models
        if roi is not None and self.focus and len(self.models) > 1:
            # tracking with an ROI: only the model that can see the locked class
            models = [m for m in self.models if self.focus in m.logical_set] or self.models[:1]
        imgsz = self.roi_imgsz if roi is not None else self.imgsz
        t0 = time.perf_counter()
        out: list[Detection] = []
        for m in models:
            res = m.predict(img, imgsz, self.conf, self.iou, self.max_det)
            if not res:
                continue
            boxes = res[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                raw = m.names.get(int(k), str(k))
                out.append(Detection((float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy)),
                                     m.logical(raw), float(c), raw))
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def info(self) -> dict:
        d = super().info()
        d.update({"model": Path(self.model_path).name, "device": self.device, "half": self.half, "imgsz": self.imgsz,
                  "classes": sorted(self.main.logical_set)})
        if self.assist is not None:
            d["assist"] = f"{self.assist.path.name}: {', '.join(sorted(self.assist.logical_set))}"
        if self.meta.get("datasets"):
            d["trained_on"] = ", ".join(f"{k} ({v})" for k, v in self.meta["datasets"].items())
        if self.fallback_used:
            d["note"] = self.fallback_used
        return d


class ColorBlobDetector(DetectorBase):
    """HSV threshold + circularity. Default range = optic-yellow tennis ball."""

    name = "color"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = dict(self.cfg)
        c.update(self.cfg.get("color", {}) or {})
        self.lo = np.array(c.get("hsv_lo", [25, 90, 90]), np.uint8)
        self.hi = np.array(c.get("hsv_hi", [45, 255, 255]), np.uint8)
        self.min_area = float(c.get("min_area_px", 25))
        self.min_circ = float(c.get("min_circularity", 0.55))
        self.cls_name = str(c.get("class_name", "sports ball"))
        self.down = max(1, int(c.get("downscale", 2)))

    def detect(self, frame: Frame, roi=None) -> list[Detection]:
        t0 = time.perf_counter()
        img, ox, oy = _crop(frame.image, roi)
        if img.size == 0:
            return []
        s = self.down
        small = cv2.resize(img, (img.shape[1] // s, img.shape[0] // s), interpolation=cv2.INTER_AREA) if s > 1 else img
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c) * s * s
            if area < self.min_area:
                continue
            per = cv2.arcLength(c, True) * s
            circ = 4 * math.pi * area / (per * per) if per > 0 else 0
            if circ < self.min_circ:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection((x * s + ox, y * s + oy, (x + w) * s + ox, (y + h) * s + oy), self.logical(self.cls_name),
                                 float(min(0.99, 0.4 + 0.6 * circ)), self.cls_name))
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        return out


class SimTruthDetector(DetectorBase):
    """Uses the simulator's ground truth, degraded with box noise, misses, false alarms and latency.

    Lets the whole downstream chain (selection, micro tracking, Kalman, control,
    logging) be tested without a trained model. The micro tracker still works on
    the rendered pixels.
    """

    name = "sim_truth"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.noise_px = float(self.cfg.get("noise_px", 1.5))
        self.miss_prob = float(self.cfg.get("miss_prob", 0.03))
        self.fp_prob = float(self.cfg.get("false_positive_prob", 0.02))
        self.latency = float(self.cfg.get("latency_ms", 15.0)) / 1000.0
        self.min_visible = float(self.cfg.get("min_visible", 0.55))
        self.rng = random.Random(5)

    def detect(self, frame: Frame, roi=None) -> list[Detection]:
        t0 = time.perf_counter()
        out = []
        truth = frame.meta.get("truth", [])
        H, W = frame.height, frame.width
        for tr in truth:
            if tr["visible"] < self.min_visible or self.rng.random() < self.miss_prob:
                continue
            x1, y1, x2, y2 = tr["bbox"]
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            if roi is not None:
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                    continue
            n = self.noise_px
            box = (x1 + self.rng.gauss(0, n), y1 + self.rng.gauss(0, n), x2 + self.rng.gauss(0, n), y2 + self.rng.gauss(0, n))
            conf = min(0.99, 0.55 + 0.4 * tr["visible"] + self.rng.gauss(0, 0.03))
            out.append(Detection(box, self.logical(tr["cls"]), conf, tr["cls"]))
        if self.rng.random() < self.fp_prob:
            w = self.rng.uniform(15, 60)
            x, y = self.rng.uniform(0, W - w), self.rng.uniform(0, H - w)
            out.append(Detection((x, y, x + w, y + w), "drone", self.rng.uniform(0.25, 0.42), "drone"))
        remaining = self.latency - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)   # emulate inference time on the edge GPU
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        return out


def create_detector(cfg: dict) -> DetectorBase:
    kind = str((cfg or {}).get("backend", "yolo")).lower()
    if kind == "yolo":
        return YoloDetector(cfg)
    if kind in ("color", "colour", "blob"):
        return ColorBlobDetector(cfg)
    if kind in ("sim", "sim_truth", "truth"):
        return SimTruthDetector(cfg)
    raise ValueError(f"unknown detector backend {kind!r}")
