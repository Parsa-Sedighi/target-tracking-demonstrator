"""RF-DETR Nano drone detector: dataset checks, training, and inference.

This module is the skeleton for the project's drone detector. It is usable before
real data exists: ``make-dummy-data`` generates a small synthetic dataset so the
full train -> checkpoint -> predict pipeline can be smoke-tested end to end.

Install (inside the project venv):
    pip install "rfdetr[train]==1.11.2"

Command line (run from the repo root):
    # 1. Generate a throwaway synthetic dataset (until real data is available)
    python ros2_ws/src/perception_stack/detection_tracking/rf_detr.py make-dummy-data

    # 2. Verify a dataset folder has the layout RF-DETR expects
    python ros2_ws/src/perception_stack/detection_tracking/rf_detr.py check-data datasets/drones

    # 3. Train
    python ros2_ws/src/perception_stack/detection_tracking/rf_detr.py train datasets/drones --epochs 50

    # 4. Run the trained model on an image
    python ros2_ws/src/perception_stack/detection_tracking/rf_detr.py predict \
        runs/rfdetr_nano/checkpoint_best_total.pth some_image.jpg

Expected dataset layout (COCO, the default Roboflow export format):
    datasets/drones/
        train/  _annotations.coco.json  + images
        valid/  _annotations.coco.json  + images
        test/   _annotations.coco.json  + images
A YOLO-format export (``data.yaml`` + images/labels folders) is also accepted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Paths are relative to the directory you run the script from (the repo root).
DEFAULT_DATASET_DIR = Path("datasets/drones")
DEFAULT_DUMMY_DIR = Path("datasets/dummy_drones")
DEFAULT_OUTPUT_DIR = Path("runs/rfdetr_nano")
COCO_SPLITS = ("train", "valid", "test")
COCO_ANNOTATION_FILE = "_annotations.coco.json"


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
def check_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validate the dataset layout and return a short summary.

    Raises FileNotFoundError / ValueError with a readable message if the layout is wrong,
    so problems surface before a long training run starts rather than partway through.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    yolo_yaml = next((p for p in (dataset_dir / "data.yaml", dataset_dir / "data.yml") if p.exists()), None)
    if yolo_yaml and not (dataset_dir / "train" / COCO_ANNOTATION_FILE).exists():
        return {"format": "yolo", "data_file": str(yolo_yaml)}

    summary: dict[str, Any] = {"format": "coco", "splits": {}}
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
    """Training hyperparameters. Defaults are sized for a laptop / single small GPU."""

    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 50
    batch_size: int = 4
    grad_accum_steps: int = 4  # effective batch = batch_size * grad_accum_steps = 16
    lr: float = 1e-4
    resolution: int | None = None  # None = Nano default (384); must be divisible by 32
    device: str | None = None  # None = auto (cuda > mps > cpu)
    num_workers: int = 2
    early_stopping: bool = True
    resume: Path | None = None  # path to a checkpoint to continue from


def train(settings: TrainSettings) -> Path:
    """Fine-tune RF-DETR Nano (COCO-pretrained) on the drone dataset. Returns the output folder."""
    from rfdetr import RFDETRNano

    summary = check_dataset(settings.dataset_dir)
    print(f"[rf_detr] dataset OK: {summary}")

    device = pick_device(settings.device)
    print(f"[rf_detr] training on device: {device}")

    model = RFDETRNano()  # downloads the pretrained rf-detr-nano.pth on first use (~100 MB)
    kwargs: dict[str, Any] = dict(
        dataset_dir=str(settings.dataset_dir),
        output_dir=str(settings.output_dir),
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        grad_accum_steps=settings.grad_accum_steps,
        lr=settings.lr,
        device=device,
        num_workers=settings.num_workers,
        early_stopping=settings.early_stopping,
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
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        confidence_threshold: float = 0.5,
        optimize: bool = True,
    ) -> None:
        from rfdetr import RFDETR

        self.conf_thresh = confidence_threshold
        # trust_checkpoint=True: we load our own locally trained file, not an untrusted download.
        self.model = RFDETR.from_checkpoint(str(checkpoint_path), trust_checkpoint=True)
        self.class_names: list[str] = list(getattr(self.model, "class_names", None) or [])
        if optimize:
            try:
                self.model.optimize_for_inference()
            except Exception as exc:  # optional speed-up; plain predict still works without it
                print(f"[rf_detr] optimize_for_inference skipped: {exc}")

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
        print(json.dumps(check_dataset(args.dataset_dir), indent=2))

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
