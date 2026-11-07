"""Console log for the long-running tools (fetch, build, train, audit).

    sys.exit(runlog.run("train", main))

Everything the tool prints still goes to the console as before (progress bars included) and is
also appended to logs/<name>.log, so a run can be checked afterwards or from another machine.
A progress bar that redraws itself with carriage returns is written once, in its final state,
and colour codes are left out of the file. On Windows the PC is kept from going to sleep while
the tool runs (closing a laptop lid still follows the lid setting).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class LineSink:
    """Collects written text and writes complete lines to a file; '\\r' restarts the pending line."""

    def __init__(self, f):
        self.f = f
        self.buf = ""

    def feed(self, s: str) -> None:
        if not s:
            return
        self.buf += s
        while True:
            i = self.buf.find("\n")
            if i < 0:
                break
            line, self.buf = self.buf[:i], self.buf[i + 1:]
            line = line.rstrip("\r")
            line = line[line.rfind("\r") + 1:]
            self._write(line)
        # a bar that keeps redrawing without a newline: keep only its latest state in memory
        j = self.buf.rfind("\r", 0, max(0, len(self.buf) - 1))
        if j >= 0:
            self.buf = self.buf[j + 1:]

    def _write(self, line: str) -> None:
        if "\b" in line:                    # progress redrawn with backspaces (7-Zip): keep the last state
            line = line.rstrip("\b").rsplit("\b", 1)[-1]
        try:
            self.f.write(_ANSI.sub("", line) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        if self.buf.strip("\r"):
            self._write(self.buf[self.buf.rstrip("\r").rfind("\r") + 1:].rstrip("\r"))
        self.buf = ""
        try:
            self.f.flush()
        except Exception:
            pass


class Tee:
    """sys.stdout / sys.stderr replacement: the console stream plus a LineSink."""

    def __init__(self, stream, sink: LineSink):
        self._stream = stream
        self._sink = sink

    def write(self, s):
        n = len(s)
        try:
            n = self._stream.write(s)
        except UnicodeEncodeError:          # e.g. output redirected to a cp1252 file
            enc = getattr(self._stream, "encoding", None) or "ascii"
            try:
                n = self._stream.write(s.encode(enc, "replace").decode(enc, "replace"))
            except Exception:
                pass
        except Exception:
            pass
        self._sink.feed(s)
        return n

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._sink.f.flush()
        except Exception:
            pass

    def isatty(self):                       # progress bars stay interactive in the console
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


_state: dict = {}


def keep_awake(on: bool = True) -> bool:
    """Windows: stop the PC from going to sleep while this process runs (no admin rights needed)."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


def start(name: str, log_dir: Optional[Path] = None, awake: bool = True) -> Optional[Path]:
    """Start copying stdout and stderr to <log_dir>/<name>.log (appended). Returns the path."""
    if _state:
        return _state.get("path")
    log_dir = Path(log_dir) if log_dir else LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{name}.log"
        f = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    except Exception:
        return None                          # read-only folder etc.: run without a log
    sink = LineSink(f)
    sink._write("")
    sink._write(f"===== {dt.datetime.now().isoformat(sep=' ', timespec='seconds')}  "
                f"{' '.join([Path(sys.argv[0]).name] + sys.argv[1:])}")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(orig_out, sink), Tee(orig_err, sink)
    # loggers created before this point hold the old streams
    loggers = [logging.getLogger()] + [lg for lg in logging.Logger.manager.loggerDict.values()
                                       if isinstance(lg, logging.Logger)]
    for lg in loggers:
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                if h.stream is orig_out:
                    h.setStream(sys.stdout)
                elif h.stream is orig_err:
                    h.setStream(sys.stderr)
    _state.update(path=path, sink=sink, out=orig_out, err=orig_err, awake=awake and keep_awake(True))
    return path


def stop(code: Optional[int] = None) -> None:
    if not _state:
        return
    sink: LineSink = _state["sink"]
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sink.close()
    when = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    sink._write(f"===== {when}  finished, exit code {code}" if code is not None else f"===== {when}  finished")
    for lg in [logging.getLogger()] + [lg for lg in logging.Logger.manager.loggerDict.values()
                                       if isinstance(lg, logging.Logger)]:
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and isinstance(getattr(h, "stream", None), Tee):
                h.setStream(_state["out"] if h.stream is sys.stdout else _state["err"])
    sys.stdout, sys.stderr = _state["out"], _state["err"]
    try:
        sink.f.close()
    except Exception:
        pass
    if _state.get("awake"):
        keep_awake(False)
    _state.clear()


def run(name: str, main: Callable[[], Optional[int]], log_dir: Optional[Path] = None) -> int:
    """Run main() with its output also going to logs/<name>.log; returns the exit code."""
    path = start(name, log_dir)
    code = 1
    try:
        code = main() or 0
    except SystemExit as e:                  # argparse errors / --help
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            code = 1
        else:
            code = int(e.code or 0)
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
        code = 130
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        stop(code)
    if path and code not in (0, 130):
        print(f"(the full output is in {path})")
    return code
