"""Launch the operator UI.

    python ui_stack/run_ui.py                      # use the trained RF-DETR model if it exists
    python ui_stack/run_ui.py --model path/to.pth  # use a specific checkpoint
    python ui_stack/run_ui.py --demo               # fake detections only, no model / GPU needed

The model runs on Webcam and Video file sources; the Demo source always uses fake detections.
"""
import argparse
import os
import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
REPO_ROOT = UI_DIR.parent
DETECTION_DIR = REPO_ROOT / "ros2_ws" / "src" / "perception_stack" / "detection_tracking"
DEFAULT_MODEL = REPO_ROOT / "runs" / "rfdetr_nano" / "checkpoint_best_total.pth"

# Let Python find main_window.py, demo_detector.py and video_canvas.py.
sys.path.insert(0, str(UI_DIR / "src" / "display_gui"))

# On some Macs Qt starts with an empty plugin search list and aborts with
# 'Could not find the Qt platform plugin "cocoa"'. Point it at PySide6's own plugins.
try:
    import PySide6

    _platforms = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins" / "platforms"
    if _platforms.is_dir():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_platforms))
except ImportError:
    pass  # main_window will report the missing package with a clear error

from main_window import main  # noqa: E402


def load_detector(args):
    """Return (detector, display name), or (None, None) to run with fake detections only."""
    if args.demo:
        return None, None
    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    if not model_path.exists():
        if args.model:
            sys.exit(f"Model checkpoint not found: {model_path}")
        print(f"[run_ui] No trained model at {model_path}; using fake demo detections.")
        return None, None

    sys.path.insert(0, str(DETECTION_DIR))
    from rf_detr import DroneDetector  # imported lazily: needs torch + rfdetr

    print(f"[run_ui] Loading RF-DETR model {model_path} ...")
    detector = DroneDetector(model_path, confidence_threshold=args.threshold, optimize=not args.no_optimize)
    return detector, f"RF-DETR ({model_path.name}, conf >= {args.threshold:.2f})"


def parse_args():
    parser = argparse.ArgumentParser(description="Target Tracking Demonstrator operator UI")
    parser.add_argument("--model", help=f"RF-DETR checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection confidence")
    parser.add_argument("--demo", action="store_true", help="fake detections only; do not load a model")
    parser.add_argument("--no-optimize", action="store_true", help="skip model compilation (slower, starts faster)")
    return parser.parse_args()


if __name__ == "__main__":
    main(*load_detector(parse_args()))
