"""Launch the operator UI:  python run_ui.py   (run from inside the ui_stack folder)."""
import os
import sys
from pathlib import Path

# Let Python find main_window.py, demo_detector.py and video_canvas.py.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "display_gui"))

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

if __name__ == "__main__":
    main()
