"""RF-DETR Nano drone detector: dataset checks, training, and inference.

The default dataset is the visible-light part of the UAV release
(``uav_dataset/uav_dataset_release/visible``, one class: ``0 = drone``).

Install (inside the project venv). The RTX 50-series needs a CUDA 12.8+ PyTorch build:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install "rfdetr[train]==1.11.2"

Command line (run from the repo root; S = ros2_ws/src/perception_stack/detection_tracking/rf_detr.py):
    # 1. Verify the dataset (counts images, boxes, negatives; checks every label line)
    python S check-data

    # 2. Optional: build a small hard-linked subset and do a quick smoke-test run first
    python S make-subset --train 2000 --val 400 --test 400
    python S train datasets/uav_subset --epochs 2

    # 3. Full training run
    python S train --epochs 20

    # 4. Run the trained model on an image
    python S predict runs/rfdetr_nano/checkpoint_best_total.pth some_image.jpg

Supported dataset layouts:
    YOLO (Ultralytics or Roboflow style): a ``data.yaml`` whose ``train``/``val``/``test`` entries
        point at image folders (e.g. ``images/train``); labels sit in the matching ``labels/``
        folder. Empty label files are negatives (images with no drone).
    COCO (Roboflow export): ``train/``, ``valid/``, ``test/``, each with ``_annotations.coco.json``.
``make-dummy-data`` still generates a tiny synthetic COCO dataset for pipeline testing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Paths are relative to the directory you run the script from (the repo root).
DEFAULT_DATASET_DIR = Path("uav_dataset/uav_dataset_release/visible")
DEFAULT_SUBSET_DIR = Path("datasets/uav_subset")
DEFAULT_DUMMY_DIR = Path("datasets/dummy_drones")
DEFAULT_OUTPUT_DIR = Path("runs/rfdetr_nano")
COCO_SPLITS = ("train", "valid", "test")
COCO_ANNOTATION_FILE = "_annotations.coco.json"
YOLO_SPLITS = ("train", "val", "test")
YOLO_YAML_FILES = ("data.yaml", "data.yml")
# Same extension set RF-DETR's YOLO loader accepts.
YOLO_IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpg", ".jpeg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device(requested: str | None = None) -> str:
    """Return the best available torch device: CUDA (Jetson/NVIDIA) > MPS (Apple) > CPU."""
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def check_dataset(dataset_dir: Path, scan_labels: bool = True) -> dict[str, Any]:
    """Validate the dataset layout and return a short summary.

    ``summary["dataset_file"]`` is the value RF-DETR's ``train(dataset_file=...)`` needs.
    ``scan_labels=False`` skips reading every YOLO label file (only folder structure and
    file counts are checked), which is much faster on a large dataset.

    Raises FileNotFoundError / ValueError with a readable message if the layout is wrong,
    so problems surface before a long training run starts rather than partway through.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    yolo_yaml = next((dataset_dir / n for n in YOLO_YAML_FILES if (dataset_dir / n).exists()), None)
    if yolo_yaml and not (dataset_dir / "train" / COCO_ANNOTATION_FILE).exists():
        return _check_yolo_dataset(dataset_dir, yolo_yaml, scan_labels)

    summary: dict[str, Any] = {"format": "coco", "dataset_file": "roboflow", "splits": {}}
    class_names: list[str] | None = None
    for split in COCO_SPLITS:
        ann_path = dataset_dir / split / COCO_ANNOTATION_FILE
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Missing {ann_path}. RF-DETR's COCO loader needs train/, valid/ and test/ "
                f"folders, each with a {COCO_ANNOTATION_FILE} file."
            )
        coco = json.loads(ann_path.read_text())
        images, annotations = coco.get("images", []), coco.get("annotations", [])
        missing = [im["file_name"] for im in images if not (ann_path.parent / im["file_name"]).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) listed in {ann_path} are missing, e.g. {missing[0]}")
        if not images:
            raise ValueError(f"{ann_path} contains no images")
        names = [c["name"] for c in sorted(coco.get("categories", []), key=lambda c: c["id"])]
        class_names = class_names or names
        summary["splits"][split] = {"images": len(images), "boxes": len(annotations)}
    summary["classes"] = class_names
    return summary


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_yolo_yaml(yaml_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Read data.yaml and return (raw mapping, class names as a 0..N-1 list)."""
    import yaml  # PyYAML ships with rfdetr's dependencies

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        ids = sorted(int(k) for k in names)
        if ids != list(range(len(ids))):
            raise ValueError(f"{yaml_path}: class ids must be contiguous 0..N-1, got {ids}")
        class_names = [str(names[k] if k in names else names[str(k)]) for k in ids]
    elif isinstance(names, list):
        class_names = [str(n) for n in names]
    else:
        raise ValueError(f"{yaml_path}: missing or invalid 'names' entry")
    if not class_names:
        raise ValueError(f"{yaml_path}: 'names' is empty")
    return data, class_names


def _yolo_split_dirs(root: Path, data: dict[str, Any], yaml_path: Path, split: str) -> tuple[Path, Path] | None:
    """Return (images_dir, labels_dir) for a split, resolved the same way RF-DETR does.

    Paths declared in data.yaml win (Ultralytics layout: ``val: images/val``); otherwise the
    Roboflow layout ``<split>/images`` is used. Returns None if the split is not present.
    """
    raw = data.get(split)
    if raw is None and split == "val":
        raw = data.get("valid")

    if raw is None:
        folder = root / ("valid" if split == "val" else split)
        if split == "val" and not folder.exists():
            folder = root / "val"
        images, labels = folder / "images", folder / "labels"
        return (images, labels) if images.is_dir() else None

    if not isinstance(raw, str):
        raise ValueError(f"{yaml_path}: '{split}' must be a single folder path, got {raw!r}")

    base = root
    declared_base = data.get("path")
    if declared_base:
        base = Path(declared_base) if Path(declared_base).is_absolute() else root / declared_base
        if not _is_within(base, root):
            # RF-DETR silently ignores declared paths outside the dataset root, then fails to
            # find the Roboflow fallback folders, so flag it clearly here.
            raise ValueError(
                f"{yaml_path}: 'path: {declared_base}' points outside the dataset folder. "
                "Delete the 'path:' line so split paths are read relative to data.yaml."
            )

    images = base / raw
    if not _is_within(images, root):
        raise ValueError(f"{yaml_path}: '{split}: {raw}' resolves outside the dataset folder")
    parts = images.parts
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")  # rightmost 'images' segment
        labels = Path(*parts[:idx], "labels", *parts[idx + 1 :])
    else:
        images, labels = images / "images", images / "labels"
    return images, labels


def _list_images(folder: Path) -> list[str]:
    with os.scandir(folder) as it:
        return sorted(
            e.name for e in it if e.is_file() and os.path.splitext(e.name)[1].lower() in YOLO_IMAGE_EXTENSIONS
        )


def _scan_yolo_label(label_path: Path, num_classes: int) -> tuple[int, list[str]]:
    """Return (box count, list of problems) for one YOLO label file."""
    boxes, problems = 0, []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        where = f"{label_path.name}:{line_no}"
        if len(fields) != 5:
            problems.append(f"{where}: expected 5 values, got {len(fields)}")
            continue
        try:
            cls_id = int(fields[0])
            xc, yc, w, h = (float(v) for v in fields[1:])
        except ValueError:
            problems.append(f"{where}: non-numeric value")
            continue
        if not 0 <= cls_id < num_classes:
            problems.append(f"{where}: class id {cls_id} not in 0..{num_classes - 1}")
        elif not (w > 0 and h > 0):
            problems.append(f"{where}: zero or negative box size")
        elif not all(-1e-3 <= v <= 1 + 1e-3 for v in (xc, yc, w, h)):
            problems.append(f"{where}: coordinates not normalized to [0, 1]")
        else:
            boxes += 1
    return boxes, problems


def _check_yolo_dataset(root: Path, yaml_path: Path, scan_labels: bool) -> dict[str, Any]:
    data, class_names = _load_yolo_yaml(yaml_path)
    summary: dict[str, Any] = {
        "format": "yolo",
        "dataset_file": "yolo",
        "data_file": str(yaml_path),
        "classes": class_names,
        "splits": {},
    }
    problems: list[str] = []
    for split in YOLO_SPLITS:
        dirs = _yolo_split_dirs(root, data, yaml_path, split)
        if dirs is None or not dirs[0].is_dir():
            if split == "test":
                continue  # RF-DETR treats the test split as optional
            raise FileNotFoundError(f"{split} images folder not found for {yaml_path}")
        images_dir, labels_dir = dirs
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

        images = _list_images(images_dir)
        if not images:
            raise ValueError(f"No images found in {images_dir}")
        label_names = {e.name for e in os.scandir(labels_dir) if e.name.endswith(".txt")}
        stems = [os.path.splitext(name)[0] for name in images]
        unlabeled = sum(1 for s in stems if f"{s}.txt" not in label_names)
        orphan_labels = len(label_names - {f"{s}.txt" for s in stems})

        stats: dict[str, Any] = {"images": len(images), "unlabeled_images": unlabeled}
        if orphan_labels:
            stats["labels_without_image"] = orphan_labels
        if scan_labels:
            boxes = negatives = 0
            for s in stems:
                label_path = labels_dir / f"{s}.txt"
                if not label_path.exists():
                    continue
                n, bad = _scan_yolo_label(label_path, len(class_names))
                boxes += n
                negatives += n == 0 and not bad
                problems.extend(f"{split}/{p}" for p in bad)
            stats.update(boxes=boxes, negative_images=negatives)
        summary["splits"][split] = stats

    if problems:
        shown = "\n  ".join(problems[:10])
        raise ValueError(f"{len(problems)} bad label line(s), e.g.:\n  {shown}")
    return summary


def make_subset(
    src_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_SUBSET_DIR,
    counts: tuple[int, int, int] = (2000, 400, 400),
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Create a small random YOLO subset of a YOLO dataset for quick test runs.

    Files are hard-linked (no extra disk space) when source and destination are on the same
    drive, and copied otherwise. The output uses the Ultralytics layout with its own data.yaml.
    """
    import yaml

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    yaml_path = next((src_dir / n for n in YOLO_YAML_FILES if (src_dir / n).exists()), None)
    if yaml_path is None:
        raise FileNotFoundError(f"No data.yaml in {src_dir}; make-subset only supports YOLO datasets")
    data, class_names = _load_yolo_yaml(yaml_path)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty (use --overwrite to replace it)")
        if not (out_dir / "data.yaml").exists():
            raise FileExistsError(f"Refusing to delete {out_dir}: it does not look like a subset folder")
        shutil.rmtree(out_dir)

    rng = random.Random(seed)
    for split, count in zip(YOLO_SPLITS, counts):
        dirs = _yolo_split_dirs(src_dir, data, yaml_path, split)
        if dirs is None or not dirs[0].is_dir():
            continue
        images = _list_images(dirs[0])
        picked = rng.sample(images, min(count, len(images)))
        img_out, lbl_out = out_dir / "images" / split, out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for name in picked:
            _link_or_copy(dirs[0] / name, img_out / name)
            label = dirs[1] / f"{os.path.splitext(name)[0]}.txt"
            if label.exists():
                _link_or_copy(label, lbl_out / label.name)

    subset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    if not (out_dir / "images" / "test").is_dir():
        del subset_yaml["test"]
    (out_dir / "data.yaml").write_text(yaml.safe_dump(subset_yaml, sort_keys=False), encoding="utf-8")
    return out_dir


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_dummy_dataset(
    out_dir: Path = DEFAULT_DUMMY_DIR,
    counts: tuple[int, int, int] = (40, 10, 10),
    image_size: tuple[int, int] = (640, 480),
    seed: int = 0,
) -> Path:
    """Write a tiny synthetic COCO dataset of "drones" (dark quadcopter shapes on a sky gradient).

    This is only for testing the pipeline; a model trained on it is not useful on real footage.
    """
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    width, height = image_size
    categories = [{"id": 1, "name": "drone", "supercategory": "none"}]

    for split, count in zip(COCO_SPLITS, counts):
        split_dir = Path(out_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for i in range(count):
            # Sky-like vertical gradient with a bit of noise.
            top = np.array([rng.randint(90, 150), rng.randint(140, 190), rng.randint(200, 255)])
            bottom = np.array([rng.randint(180, 230), rng.randint(200, 235), rng.randint(220, 255)])
            t = np.linspace(0, 1, height)[:, None, None]
            sky = (top * (1 - t) + bottom * t).repeat(width, axis=1)
            sky += np.random.default_rng(seed + i).normal(0, 4, sky.shape)
            img = Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(img)

            boxes = []
            for _ in range(rng.randint(1, 3)):
                size = rng.randint(20, 90)
                x0 = rng.randint(0, width - size)
                y0 = rng.randint(0, height - size)
                _draw_quadcopter(draw, x0, y0, size, shade=rng.randint(10, 60))
                boxes.append([x0, y0, size, size])

            file_name = f"{split}_{i:04d}.jpg"
            img.save(split_dir / file_name, quality=90)
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
            for box in boxes:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                    }
                )

        coco = {"images": images, "annotations": annotations, "categories": categories}
        (split_dir / COCO_ANNOTATION_FILE).write_text(json.dumps(coco))
    return Path(out_dir)


def _draw_quadcopter(draw: Any, x0: int, y0: int, size: int, shade: int) -> None:
    """Draw a simple top-down quadcopter (body, two arms, four rotors) inside a size x size box."""
    color = (shade, shade, shade)
    r = max(2, size // 6)  # rotor radius
    x1, y1 = x0 + size, y0 + size
    arm = max(1, size // 14)
    draw.line([(x0 + r, y0 + r), (x1 - r, y1 - r)], fill=color, width=arm)
    draw.line([(x1 - r, y0 + r), (x0 + r, y1 - r)], fill=color, width=arm)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=max(1, arm))
    c = size // 2
    b = max(2, size // 8)
    draw.rectangle([x0 + c - b, y0 + c - b, x0 + c + b, y0 + c + b], fill=color)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainSettings:
    """Training hyperparameters. Defaults are sized for a single 8 GB GPU (e.g. RTX 5060).

    The full visible set is ~63k training images, so one epoch is already ~15k steps at
    batch size 4: a 20-epoch cap with early stopping is plenty for a Nano fine-tune.
    """

    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 20
    batch_size: int = 4
    grad_accum_steps: int = 4  # effective batch = batch_size * grad_accum_steps = 16
    lr: float = 1e-4
    resolution: int | None = None  # None = Nano default (384); must be divisible by 32
    device: str | None = None  # None = auto (cuda > mps > cpu)
    num_workers: int = 4
    early_stopping: bool = True
    early_stopping_patience: int = 5  # epochs without mAP improvement before stopping
    checkpoint_interval: int = 1  # save a resumable checkpoint every N epochs
    run_test: bool = True  # evaluate on the test split once training ends
    resume: Path | None = None  # path to a checkpoint to continue from


def train(settings: TrainSettings) -> Path:
    """Fine-tune RF-DETR Nano (COCO-pretrained) on the drone dataset. Returns the output folder."""
    from rfdetr import RFDETRNano

    # Full check (~10 s on the 90k-image set) so a bad label fails now, not hours into training.
    summary = check_dataset(settings.dataset_dir)
    print(f"[rf_detr] dataset OK: {json.dumps(summary)}")

    device = pick_device(settings.device)
    print(f"[rf_detr] training on device: {device}")
    if device == "cpu":
        print("[rf_detr] WARNING: no GPU found; training on CPU will be extremely slow.")

    model = RFDETRNano()  # downloads the pretrained rf-detr-nano.pth on first use (~100 MB)
    kwargs: dict[str, Any] = dict(
        dataset_dir=str(settings.dataset_dir),
        # RF-DETR does not auto-detect the layout: "yolo" for data.yaml datasets, "roboflow" for COCO.
        dataset_file=summary["dataset_file"],
        output_dir=str(settings.output_dir),
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        grad_accum_steps=settings.grad_accum_steps,
        lr=settings.lr,
        device=device,
        num_workers=settings.num_workers,
        early_stopping=settings.early_stopping,
        early_stopping_patience=settings.early_stopping_patience,
        checkpoint_interval=settings.checkpoint_interval,
        run_test=settings.run_test and "test" in summary.get("splits", {}),
    )
    if settings.resolution:
        kwargs["resolution"] = settings.resolution
    if settings.resume:
        kwargs["resume"] = str(settings.resume)

    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    model.train(**kwargs)
    print(f"[rf_detr] done. Best weights: {Path(settings.output_dir) / 'checkpoint_best_total.pth'}")
    return Path(settings.output_dir)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class DroneDetector:
    """Runs a trained RF-DETR checkpoint on camera frames.

    ``detect_targets`` returns the same dict format as ``yolo_classifier.TargetClassifier``
    so the two detectors are interchangeable in the perception pipeline:
        {'class': str, 'bbox': (x, y, w, h), 'confidence': float}
    ``detect`` is the same call under the name the operator UI (ui_stack) expects.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        confidence_threshold: float = 0.5,
        optimize: bool = True,
    ) -> None:
        from rfdetr import RFDETR

        self.conf_thresh = confidence_threshold
        self.checkpoint_path = Path(checkpoint_path)
        # trust_checkpoint=True: we load our own locally trained file, not an untrusted download.
        self.model = RFDETR.from_checkpoint(str(checkpoint_path), trust_checkpoint=True)
        self.class_names: list[str] = list(getattr(self.model, "class_names", None) or [])
        if optimize:
            self._optimize()

    def _optimize(self) -> None:
        """Compile the model for faster per-frame inference (optional; predict works without it)."""
        try:
            if hasattr(self.model, "inference"):  # rfdetr >= 1.7
                self.model.inference()
            else:  # older rfdetr releases
                self.model.optimize_for_inference()
        except Exception as exc:
            print(f"[rf_detr] inference optimization skipped: {exc}")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect drones in one BGR (OpenCV) frame; the interface used by the operator UI."""
        return self.detect_targets(frame, bgr=True)

    def detect_targets(self, frame: np.ndarray, bgr: bool = True) -> list[dict[str, Any]]:
        """Detect drones in one frame. OpenCV frames are BGR, so they are flipped to RGB first."""
        rgb = frame[:, :, ::-1].copy() if bgr else frame
        dets = self.model.predict(rgb, threshold=self.conf_thresh, include_source_image=False)

        results = []
        for (x1, y1, x2, y2), conf, cls_id in zip(dets.xyxy, dets.confidence, dets.class_id):
            results.append(
                {
                    "class": self._class_name(int(cls_id)),
                    "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "confidence": float(conf),
                }
            )
        return results

    def _class_name(self, cls_id: int) -> str:
        if isinstance(self.class_names, dict):
            return str(self.class_names.get(cls_id, cls_id))
        return self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RF-DETR Nano drone detector")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-dummy-data", help="generate a small synthetic drone dataset")
    p.add_argument("--out", type=Path, default=DEFAULT_DUMMY_DIR)
    p.add_argument("--counts", type=int, nargs=3, default=(40, 10, 10), metavar=("TRAIN", "VALID", "TEST"))

    p = sub.add_parser("check-data", help="validate a dataset folder")
    p.add_argument("dataset_dir", type=Path, nargs="?", default=DEFAULT_DATASET_DIR)
    p.add_argument("--quick", action="store_true", help="only check folders and file counts, skip label scan")

    p = sub.add_parser("make-subset", help="hard-link a small random YOLO subset for quick test runs")
    p.add_argument("dataset_dir", type=Path, nargs="?", default=DEFAULT_DATASET_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_SUBSET_DIR)
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--val", type=int, default=400)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="replace an existing subset folder")

    p = sub.add_parser("train", help="fine-tune RF-DETR Nano")
    p.add_argument("dataset_dir", type=Path, nargs="?", default=DEFAULT_DATASET_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=TrainSettings.epochs)
    p.add_argument("--batch-size", type=int, default=TrainSettings.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=TrainSettings.grad_accum_steps)
    p.add_argument("--lr", type=float, default=TrainSettings.lr)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--num-workers", type=int, default=TrainSettings.num_workers)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--patience", type=int, default=TrainSettings.early_stopping_patience)
    p.add_argument("--checkpoint-interval", type=int, default=TrainSettings.checkpoint_interval)
    p.add_argument("--no-test", action="store_true", help="skip the final test-split evaluation")
    p.add_argument("--resume", type=Path, default=None)

    p = sub.add_parser("predict", help="run a checkpoint on an image and print detections")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("image", type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "make-dummy-data":
        out = make_dummy_dataset(args.out, tuple(args.counts))
        print(f"[rf_detr] wrote synthetic dataset to {out}")
        print(json.dumps(check_dataset(out), indent=2))

    elif args.command == "check-data":
        print(json.dumps(check_dataset(args.dataset_dir, scan_labels=not args.quick), indent=2))

    elif args.command == "make-subset":
        out = make_subset(
            args.dataset_dir, args.out, (args.train, args.val, args.test), args.seed, args.overwrite
        )
        print(f"[rf_detr] wrote subset to {out}")
        print(json.dumps(check_dataset(out), indent=2))

    elif args.command == "train":
        train(
            TrainSettings(
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.lr,
                resolution=args.resolution,
                device=args.device,
                num_workers=args.num_workers,
                early_stopping=not args.no_early_stopping,
                early_stopping_patience=args.patience,
                checkpoint_interval=args.checkpoint_interval,
                run_test=not args.no_test,
                resume=args.resume,
            )
        )

    elif args.command == "predict":
        from PIL import Image

        detector = DroneDetector(args.checkpoint, confidence_threshold=args.threshold, optimize=False)
        frame = np.array(Image.open(args.image).convert("RGB"))
        for det in detector.detect_targets(frame, bgr=False):
            print(det)

    return 0


if __name__ == "__main__":
    sys.exit(main())
