"""Train the AimBot detector: YOLO26 (COCO-pretrained) fine-tuned on the merged drone datasets.

    python tools/train_yolo26.py                                  # yolo26n, 640 px, 60 epochs, GPU 0
    python tools/train_yolo26.py --model yolo26s.pt --epochs 100  # bigger model (slower on the Jetson)
    python tools/train_yolo26.py --imgsz 960                      # more pixels for tiny far-away drones
    python tools/train_yolo26.py --quick                          # 10 % of the data, 5 epochs: pipeline check
    python tools/train_yolo26.py --resume                         # continue the last interrupted run
                                  (without --resume you are asked when an unfinished run is found)
    python tools/train_yolo26.py --workers 4                      # fewer data-loader processes (less memory)
    python tools/train_yolo26.py --finalize runs/detect/aimbot_yolo26n   # ...or ship it as it is
    python tools/train_yolo26.py --eval-only --weights models/aimbot_yolo26n.pt   # per-dataset scores

Before: python tools/fetch_datasets.py && python tools/build_dataset.py (docs/TRAINING.md).
After:  models/aimbot_yolo26n.pt (+ .json with classes, datasets and scores) is what AimBot loads
(detector.model in config/*.yaml). On the Jetson build the TensorRT engine from it:
        python3 tools/export_tensorrt.py --model models/aimbot_yolo26n.pt

YOLO26 is NMS-free end to end and uses small-target-aware label assignment (STAL), which suits
small, distant drones; optimizer=auto picks MuSGD for long runs.

A run that stops making progress (no batch for --stall-minutes) is ended with exit code 75;
train_yolo26.bat then continues it from its last saved epoch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))           # data-loader workers (Windows spawn) import aimbot.training.augment
DEFAULT_DATA = ROOT / "datasets" / "aimbot_drones" / "aimbot_drones.yaml"
PRD_PRECISION, PRD_RECALL = 0.85, 0.80          # PRD user story 2
STALL_EXIT = 75                                 # "stalled, continue it": train_yolo26.bat resumes on this code


def memory_gb() -> float:
    """Physical memory of this PC in GB (0 if unknown)."""
    try:
        import psutil                          # comes with Ultralytics

        return psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return 0.0


def plan_workers(train: Optional[int], val: Optional[int], ram_gb: float, cpus: int,
                 windows: bool = os.name == "nt") -> tuple[int, int]:
    """Data-loader processes for (training, validation).

    Ultralytics keeps both loaders' processes alive for the whole run and gives validation twice as
    many as training. On Windows every one of them starts from scratch and holds about 2 GB: 8 + 16
    filled a 64 GB PC (error 1455, "the paging file is too small") and the run stalled. Defaults: up
    to 8 for training (they set the speed) and 4 for validation, fewer on PCs with less memory.
    Numbers given on the command line are kept, except that validation gets an even number (at
    least 2), because Ultralytics doubles the setting it is given.
    """
    per = 2.0 if windows else 1.0              # Linux forks the workers, which shares most of that
    t = train if train is not None else max(1, min(8, cpus))
    v = val if val is not None else 4
    if ram_gb > 0:
        room = max(3, int((ram_gb * 0.6 - 6) // per))  # 60 % of the memory, less ~6 GB for the trainer itself
        if val is None:
            v = min(v, max(2, room // 3))
        if train is None:
            t = min(t, max(1, room - v))
    if t <= 0:                                 # --workers 0: everything loads in the training process
        return 0, 0
    return t, max(2, v - v % 2)


def make_trainer(train_workers: int, val_workers: int):
    """Ultralytics' detection trainer, with its own data-loader process counts for training and validation."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class AimBotTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            keep = self.args.workers
            if not keep:                       # 0: CPU training (Ultralytics loads in-process there) or asked for
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            # Ultralytics uses args.workers processes for training and twice that for validation
            self.args.workers = train_workers if mode == "train" else max(1, val_workers // 2)
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = keep

    return AimBotTrainer


class Watchdog:
    """Calls on_stall when no training or validation batch has finished for `minutes`.

    A data-loader process that cannot hand over a batch (on Windows: memory ran out) leaves the
    run waiting for that batch forever, with the GPU idle. Watching starts with the first epoch
    and ends with the training; a PC that was asleep in between does not count as a stall.
    """

    EVENTS = ("on_train_epoch_start", "on_train_batch_end", "on_val_start", "on_val_batch_end",
              "on_fit_epoch_end", "on_model_save")

    def __init__(self, minutes: float, on_stall: Callable[[float], None], check_every: float = 30.0):
        self.limit = float(minutes) * 60.0
        self.on_stall = on_stall
        self.check_every = check_every
        self.last: Optional[float] = None      # time of the last progress; None = not watching
        self.thread: Optional[threading.Thread] = None

    def beat(self, *_) -> None:
        self.last = time.monotonic()

    def stop(self, *_) -> None:
        self.last = None

    def attach(self, model) -> "Watchdog":
        """Register on an Ultralytics model (before model.train) and start watching in the background."""
        for event in self.EVENTS:
            model.add_callback(event, self.beat)
        model.add_callback("on_train_end", self.stop)
        if self.limit > 0 and self.thread is None:
            self.thread = threading.Thread(target=self._watch, name="stall-watchdog", daemon=True)
            self.thread.start()
        return self

    def _watch(self) -> None:
        prev = time.monotonic()
        while True:
            time.sleep(self.check_every)
            now = time.monotonic()
            if now - prev > max(300.0, 4 * self.check_every) and self.last is not None:
                self.last = now                # this thread did not run for minutes: the PC was asleep
            prev = now
            last = self.last
            if last is not None and now - last > self.limit:
                self.on_stall(now - last)
                return


def stop_stalled_run(seconds: float) -> None:
    """End a stalled run: stop its data-loader processes and exit with STALL_EXIT."""
    print(f"\nNo training progress for {seconds / 60:.0f} minutes: a data-loader process stopped delivering "
          "images (on Windows usually because memory ran out). Ending this run so it can continue from its "
          "last saved epoch (train_yolo26.bat does that by itself; by hand: train_yolo26.py --resume).",
          file=sys.stderr, flush=True)
    try:
        import psutil

        kids = psutil.Process().children(recursive=True)
        for k in kids:
            try:
                k.kill()
            except Exception:
                pass
        psutil.wait_procs(kids, timeout=15)
    except Exception:
        pass
    try:
        from aimbot import runlog

        runlog.stop(STALL_EXIT)
    except Exception:
        pass
    os._exit(STALL_EXIT)


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def split_images(data_yaml: Path, split: str) -> list:
    """All images of one split of an Ultralytics data yaml, sorted (= by dataset, clip, frame)."""
    import yaml

    d = yaml.safe_load(data_yaml.read_text()) or {}
    base = Path(d.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = data_yaml.parent / base
    entry = d.get(split)
    if not entry:
        return []
    root = Path(entry) if Path(entry).is_absolute() else base / entry
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def even_sample(files: list, n: Optional[int]) -> list:
    """About n of the sorted files, evenly spaced: every dataset, clip and part of a clip keeps its share.

    (Ultralytics' own `fraction` keeps the first images in name order, i.e. only the first dataset.)"""
    if not n or n >= len(files):
        return list(files)
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def subset_data(data_yaml: Path, train_n: Optional[int], val_n: Optional[int], tag: str) -> tuple:
    """Write <dataset>/subsets/<tag>.yaml with evenly sampled train / val image lists.

    Returns (yaml path, {split: (used, total)}). A split without a number keeps all its images."""
    import yaml

    d = yaml.safe_load(data_yaml.read_text()) or {}
    base = Path(d.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = data_yaml.parent / base
    out_dir = data_yaml.parent / "subsets"
    out_dir.mkdir(exist_ok=True)
    new = dict(d)
    new["path"] = str(base)
    counts = {}
    for split, n in (("train", train_n), ("val", val_n)):
        files = split_images(data_yaml, split)
        if not files:
            raise FileNotFoundError(f"no {split} images found for {data_yaml}")
        pick = even_sample(files, n)
        counts[split] = (len(pick), len(files))
        if len(pick) == len(files):
            new[split] = str(Path(d[split]) if Path(d[split]).is_absolute() else base / d[split])
            continue
        lst = out_dir / f"{tag}_{split}.txt"
        lst.write_text("\n".join(str(p) for p in pick) + "\n", encoding="utf-8")
        new[split] = str(lst)
    if d.get("test"):
        new["test"] = str(Path(d["test"]) if Path(d["test"]).is_absolute() else base / d["test"])
    y = out_dir / f"{tag}.yaml"
    y.write_text(f"# {tag}: evenly sampled subset of {data_yaml.name} (written by tools/train_yolo26.py)\n"
                 + yaml.safe_dump(new, sort_keys=False))
    return y, counts


def per_source_eval(weights: Path, data_yaml: Path, split: str = "val", imgsz: int = 640, device=None,
                    batch: int = 16, workers: int = 4) -> dict:
    """Validate on each source's val (or test) folder separately; returns {source: metrics}."""
    import yaml
    from ultralytics import YOLO

    d = yaml.safe_load(data_yaml.read_text())
    base = Path(d.get("path", data_yaml.parent))
    names = d["names"]
    split_dir = base / "images" / split
    results = {}
    if not split_dir.exists():
        return results
    model = YOLO(str(weights))
    tmp = data_yaml.parent / f"_eval_{split}.yaml"
    for src_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if not any(src_dir.glob("*.jpg")):
            continue
        tmp.write_text(yaml.safe_dump({"path": str(base), "train": f"images/train", "val": f"images/{split}/{src_dir.name}",
                                       "names": names}))
        m = model.val(data=str(tmp), imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False,
                      workers=workers, project=str(ROOT / "runs" / "eval"), name=f"{src_dir.name}_{split}",
                      exist_ok=True)
        row = {"images": len(list(src_dir.glob("*.jpg"))), "P": float(m.box.mp), "R": float(m.box.mr),
               "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "per_class": {}}
        for j, k in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(j)
            row["per_class"][names[int(k)] if isinstance(names, dict) else names[int(k)]] = {
                "P": float(p), "R": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)}
        results[src_dir.name] = row
    tmp.unlink(missing_ok=True)
    return results


def print_table(res: dict, split: str) -> None:
    print(f"\nPer-dataset {split} scores (drone class where present, else all classes):")
    print(f"{'source':18s} {'images':>7s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s}  drone P/R")
    for k, r in res.items():
        dr = r["per_class"].get("drone")
        drs = f"{dr['P']:.2f}/{dr['R']:.2f}" if dr else "-"
        print(f"{k:18s} {r['images']:7d} {r['P']:6.3f} {r['R']:6.3f} {r['mAP50']:7.3f} {r['mAP50-95']:9.3f}  {drs}")


def find_unfinished(project: Path, name: str) -> Optional[Path]:
    """Newest run <name>, <name>2, ... if its last.pt can still be resumed (Ultralytics sets epoch -1 when done)."""
    if not project.is_dir():
        return None
    pat = re.compile(re.escape(name) + r"\d*")
    runs = [d for d in project.iterdir() if d.is_dir() and pat.fullmatch(d.name) and (d / "weights" / "last.pt").exists()]
    if not runs:
        return None
    newest = max(runs, key=lambda d: (d / "weights" / "last.pt").stat().st_mtime)
    try:
        import torch

        ck = torch.load(newest / "weights" / "last.pt", map_location="cpu", weights_only=False)
    except Exception:
        return None
    if int(ck.get("epoch", -1)) >= 0 and ck.get("optimizer") is not None:
        return newest
    return None                                # the newest run finished: start a new one


def ask_resume(run_dir: Path) -> bool:
    """An interrupted run was found: resume it (default) or start over? Only asks in a console."""
    done = epochs_done(run_dir)
    planned = "?"
    try:
        import yaml

        planned = (yaml.safe_load((run_dir / "args.yaml").read_text()) or {}).get("epochs", "?")
    except Exception:
        pass
    print(f"\nAn unfinished training run was found: {run_dir}  ({done} of {planned} epochs done).")
    if not sys.stdin or not sys.stdin.isatty():
        print("Not resuming it (no console to ask). Use --resume to continue it.")
        return False
    try:
        ans = input("Continue it? [Y = continue / n = start a new run]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("", "y", "yes", "j", "ja")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default="yolo26n.pt", help="starting weights: yolo26n/s/m.pt (COCO) or a previous .pt")
    ap.add_argument("--epochs", type=int, default=None, help="default 60 (--test-run 20, --quick 5)")
    ap.add_argument("--hours", type=float, default=None,
                    help="train for this long instead (overrides --epochs; the schedule is fitted to the time)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = AutoBatch (60%% of GPU memory); or e.g. 32")
    ap.add_argument("--device", default=None, help="0 | cpu | 0,1 (default: GPU if available)")
    ap.add_argument("--workers", type=int, default=None,
                    help="data-loader processes for training (default: up to 8, fewer on PCs with less memory)")
    ap.add_argument("--val-workers", type=int, default=None,
                    help="data-loader processes for validation (default: up to 4; an even number, at least 2)")
    ap.add_argument("--images", type=int, default=None,
                    help="train on about this many images, taken evenly from every dataset and clip (default: all)")
    ap.add_argument("--val-images", type=int, default=None,
                    help="check progress on about this many validation images (default: all; with --images a "
                         "quarter as many, at least 2000). The final per-dataset scores always use all of them.")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the images (like --images)")
    ap.add_argument("--test-run", action="store_true",
                    help="10,000 images, 20 epochs: a first model in about half an hour (models/aimbot_<model>_test.pt)")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--name", default=None, help="run name (default aimbot_<model stem>)")
    ap.add_argument("--out", default=None, help="where to copy best.pt (default models/aimbot_<model stem>.pt)")
    ap.add_argument("--resume", action="store_true", help="resume the last run with this name")
    ap.add_argument("--auto-resume", action="store_true",
                    help="continue the unfinished run if there is one, else start a new one (no question)")
    ap.add_argument("--stall-minutes", type=float, default=15.0,
                    help=f"end the run (exit code {STALL_EXIT}) when no batch finishes for this long; 0 = never")
    ap.add_argument("--quick", action="store_true", help="10%% of the images, 5 epochs: pipeline check")
    ap.add_argument("--eval-only", action="store_true", help="only run the per-dataset evaluation of --weights")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--cache", default=False, help="ram | disk | False")
    ap.add_argument("--warmup-epochs", type=float, default=None, help="default 3, less for short runs")
    ap.add_argument("--close-mosaic", type=int, default=None,
                    help="last N epochs without mosaic augmentation (default 10, less for short runs)")
    ap.add_argument("--threads", type=int, default=0, help="CPU training only: torch threads (default: all cores)")
    ap.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--camera-aug", type=float, default=0.15,
                    help="probability of motion blur and of sensor noise per training image (0 = off)")
    ap.add_argument("--finalize", metavar="RUN_DIR", default=None,
                    help="no training: turn an (interrupted) run's checkpoint into models/<name>.pt + .json")
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best", help="with --finalize: which weights")
    a = ap.parse_args()

    data = Path(a.data)
    if not data.exists():
        print(f"{data} not found. Build the dataset first:\n  python tools/fetch_datasets.py --list\n"
              "  python tools/build_dataset.py")
        return 1
    if a.threads or str(a.device).lower() == "cpu":
        # Ultralytics trains on CPU with cpu_count - 1 threads (1 on a 2-core machine); use them all
        import ultralytics.utils as _uu
        import ultralytics.utils.torch_utils as _ut

        n = a.threads or (os.cpu_count() or 1)
        os.environ["OMP_NUM_THREADS"] = str(n)
        _uu.NUM_THREADS = _ut.NUM_THREADS = n
    from ultralytics import YOLO, __version__ as ul_version

    stem = Path(a.model).stem
    suffix = "_quick" if a.quick else "_test" if a.test_run else ""   # never replaces the real model
    name = a.name or f"aimbot_{stem}{suffix}"
    out = Path(a.out) if a.out else ROOT / "models" / f"aimbot_{stem}{suffix}.pt"
    if a.epochs is None:
        a.epochs = 5 if a.quick else 20 if a.test_run else 60
    if a.warmup_epochs is None:
        a.warmup_epochs = round(min(3.0, max(0.3, 0.1 * a.epochs)), 2)
    if a.close_mosaic is None:
        a.close_mosaic = min(10, max(1, a.epochs // 6))
    if a.test_run and not a.images:
        a.images = 10000
    if a.quick and not a.images and a.fraction >= 1:
        a.fraction = 0.1
    if a.eval_only:
        w = Path(a.weights or out)
        for split in ("val", "test"):
            res = per_source_eval(w, data, split, a.imgsz, a.device)
            if res:
                print_table(res, split)
        return 0
    if a.finalize:
        run_dir = Path(a.finalize)
        return finalize(run_dir / "weights" / f"{a.checkpoint}.pt", out, data, a, ul_version, run_dir)
    train_data = data
    a.used_images = None
    if a.images or a.fraction < 1 or a.val_images:
        n_all = len(split_images(data, "train"))
        n_train = a.images or (max(1, round(n_all * a.fraction)) if a.fraction < 1 else None)
        n_val = a.val_images or (max(2000, n_train // 4) if n_train else None)
        train_data, counts = subset_data(data, n_train, n_val, name)
        a.used_images = {s: c[0] for s, c in counts.items()}
        print(f"Training on {counts['train'][0]:,} of {counts['train'][1]:,} images and checking progress on "
              f"{counts['val'][0]:,} of {counts['val'][1]:,} validation images, taken evenly from every dataset "
              f"and clip ({train_data})")
    project = Path(a.project)
    if a.camera_aug > 0:                 # also on --resume, so the rest of the run sees the same augmentation
        from aimbot.training.augment import install_camera_aug

        install_camera_aug(a.camera_aug, a.camera_aug)
    run = find_unfinished(project, name)
    if a.auto_resume and not a.resume:
        a.resume = run is not None
        if run is None:
            print("No saved epoch to continue from yet: starting the run again.")
    elif not a.resume and run is not None and ask_resume(run):
        a.resume = True
    ram, cpus = memory_gb(), os.cpu_count() or 1
    tw, vw = plan_workers(a.workers, a.val_workers, ram, cpus)
    if str(a.device).lower() != "cpu":
        print(f"Data loading: {tw} processes for training, {vw} for validation"
              + (f" (this PC has {ram:.0f} GB of memory)" if ram else ""))
        if ram and tw + vw > sum(plan_workers(None, None, ram, cpus)):
            print("  that is more than this PC's memory comfortably holds: if the run stalls, use fewer (--workers)")
    trainer = make_trainer(tw, vw)
    dog = Watchdog(a.stall_minutes, stop_stalled_run)
    if a.resume:
        if run is None:
            print(f"nothing to resume: no unfinished run {project / name}* (finished runs cannot be resumed)")
            return 1
        print(f"Resuming {run}")
        model = YOLO(str(run / "weights" / "last.pt"))
        dog.attach(model)
        model.train(resume=True, trainer=trainer, workers=tw)
    else:
        model = YOLO(a.model)
        dog.attach(model)
        model.train(
            trainer=trainer,
            data=str(train_data), epochs=a.epochs, imgsz=a.imgsz,
            batch=int(a.batch) if a.batch >= 1 or a.batch == -1 else a.batch,
            device=a.device, workers=tw, fraction=1.0, patience=a.patience,
            project=str(project), name=name, exist_ok=False, cache=a.cache, optimizer="auto", seed=0,
            warmup_epochs=a.warmup_epochs, plots=not a.no_plots, **({"time": a.hours} if a.hours else {}),
            # augmentation for this project: lighting changes indoors / IR, slight camera roll on the
            # gimbal, small far targets (scale), no vertical flips (sky is up)
            hsv_v=0.5, degrees=5.0, scale=0.5, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=a.close_mosaic,
        )
    dog.stop()
    save_dir = Path(model.trainer.save_dir)
    return finalize(save_dir / "weights" / "best.pt", out, data, a, ul_version, save_dir)


def epochs_done(run_dir: Path) -> int:
    rc = run_dir / "results.csv"
    if not rc.exists():
        return 0
    return max(0, len([ln for ln in rc.read_text().splitlines()[1:] if ln.strip()]))


def finalize(src: Path, out: Path, data: Path, a, ul_version: str, run_dir: Path) -> int:
    """Checkpoint -> deployable model: strip the optimizer, evaluate per dataset, write <model>.json."""
    import yaml
    from ultralytics.utils.torch_utils import strip_optimizer

    if not src.exists():
        print(f"no checkpoint at {src}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != out.resolve():
        shutil.copy2(src, out)
    strip_optimizer(out)                       # inference-only weights (EMA, FP16), ~5 MB for yolo26n
    print(f"\nweights ({src.parent.parent.name}/{src.parent.name}/{src.name}) -> {out}")

    res = {}
    for split in ("val", "test"):
        r = per_source_eval(out, data, split, a.imgsz, a.device)
        if r:
            res[split] = r
            print_table(r, split)
    report = {}
    rep_path = data.parent / "build_report.json"
    if rep_path.exists():
        report = json.loads(rep_path.read_text())
    names = yaml.safe_load(data.read_text())["names"]
    args = {}
    if (run_dir / "args.yaml").exists():
        args = yaml.safe_load((run_dir / "args.yaml").read_text()) or {}
    meta = {
        "model": out.name, "base": str(Path(str(args.get("model", a.model))).name), "architecture": "YOLO26",
        "classes": names, "trained": dt.datetime.now().isoformat(timespec="seconds"),
        "epochs_completed": epochs_done(run_dir), "epochs_planned": args.get("epochs", a.epochs),
        "checkpoint": src.name, "imgsz": args.get("imgsz", a.imgsz), "device": str(args.get("device", a.device)),
        "fraction": args.get("fraction", a.fraction), "ultralytics": ul_version, "run": run_dir.name,
        "datasets": {s["source"]: s["images_total"] for s in report.get("sources", [])},
        "scores": res,
    }
    if getattr(a, "used_images", None):        # trained on an evenly sampled subset (--images / --test-run)
        meta["trained_on_images"] = a.used_images.get("train")
        meta["validated_during_training_on"] = a.used_images.get("val")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    drone = [r["per_class"]["drone"] for r in res.get("val", {}).values() if "drone" in r["per_class"]]
    if drone:
        p = sum(d["P"] for d in drone) / len(drone)
        rr = sum(d["R"] for d in drone) / len(drone)
        verdict = "meets" if (p >= PRD_PRECISION and rr >= PRD_RECALL) else "does NOT meet yet"
        print(f"\nDrone class, mean over datasets: precision {p:.2f}, recall {rr:.2f} -> {verdict} the PRD target "
              f"(P >= {PRD_PRECISION}, R >= {PRD_RECALL})")
    if out.name.endswith(("_test.pt", "_quick.pt")):
        print(f"\nThis is a test model: AimBot keeps using models/aimbot_{Path(a.model).stem}.pt. To try this one:\n"
              f"  run_webcam.bat --set detector.model=models/{out.name}")
    else:
        print(f"\nAimBot uses {out} automatically (detector.model). On the Jetson:\n"
              f"  python3 tools/export_tensorrt.py --model models/{out.name}")
    return 0

if __name__ == "__main__":
    from aimbot import runlog

    sys.exit(runlog.run("train", main))
