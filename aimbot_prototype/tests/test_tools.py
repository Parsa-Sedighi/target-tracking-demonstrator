"""Console logs of the long-running tools and the interrupted-training detection."""

import io
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from aimbot import runlog  # noqa: E402


def sink_lines(*chunks):
    f = io.StringIO()
    s = runlog.LineSink(f)
    for c in chunks:
        s.feed(c)
    s.close()
    return f.getvalue().splitlines()


def test_progress_bar_redraws_are_logged_once_in_final_state():
    bar = "".join(f"\r\x1b[K  epoch 1: {i:3d}%" for i in range(0, 101, 10))
    assert sink_lines("start\n", bar, "\n", "done\r\n") == ["start", "  epoch 1: 100%", "done"]


def test_line_split_across_writes_and_crlf():
    assert sink_lines("ab", "c\r", "\n", "x\r", "y\r", "\n") == ["abc", "y"]


def test_pending_text_written_on_close_and_memory_bounded():
    f = io.StringIO()
    s = runlog.LineSink(f)
    for i in range(10000):
        s.feed(f"\rdownloading {i}")
    assert len(s.buf) < 100
    s.close()
    assert f.getvalue().splitlines() == ["downloading 9999"]


def test_run_logs_output_logging_and_exit_code(tmp_path):
    lg = logging.getLogger("aimbot_test_runlog")
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)          # created before the log starts, like Ultralytics' LOGGER
    lg.addHandler(h)
    try:
        def main():
            print("hello")
            lg.info("from a logger")
            print("oops", file=sys.stderr)
            return 3

        assert runlog.run("t", main, log_dir=tmp_path) == 3
    finally:
        lg.removeHandler(h)
    text = (tmp_path / "t.log").read_text(encoding="utf-8")
    assert "hello" in text and "from a logger" in text and "oops" in text
    assert "finished, exit code 3" in text
    assert not isinstance(sys.stdout, runlog.Tee) and not isinstance(sys.stderr, runlog.Tee)
    assert h.stream is sys.stdout


def test_run_catches_crash_and_argparse_exit(tmp_path):
    def crash():
        raise ValueError("bad thing")

    assert runlog.run("c", crash, log_dir=tmp_path) == 1
    text = (tmp_path / "c.log").read_text(encoding="utf-8")
    assert "ValueError: bad thing" in text and "Traceback" in text

    def help_exit():
        raise SystemExit(0)

    assert runlog.run("c", help_exit, log_dir=tmp_path) == 0
    assert (tmp_path / "c.log").read_text(encoding="utf-8").count("=====") == 4   # appended: 2 runs


def test_find_unfinished_training_run(tmp_path):
    torch = pytest.importorskip("torch")
    import os
    import train_yolo26 as t

    def ckpt(run, epoch, optimizer, mtime):
        w = tmp_path / run / "weights"
        w.mkdir(parents=True)
        torch.save({"epoch": epoch, "optimizer": optimizer}, w / "last.pt")
        os.utime(w / "last.pt", (mtime, mtime))

    assert t.find_unfinished(tmp_path, "aimbot_yolo26n") is None
    ckpt("aimbot_yolo26n", -1, None, 1000)                 # finished
    assert t.find_unfinished(tmp_path, "aimbot_yolo26n") is None
    ckpt("aimbot_yolo26n2", 7, {"state": 1}, 2000)         # interrupted at epoch 8
    assert t.find_unfinished(tmp_path, "aimbot_yolo26n") == tmp_path / "aimbot_yolo26n2"
    ckpt("aimbot_yolo26n_quick", 2, {"state": 1}, 3000)    # a --quick run is a different name
    assert t.find_unfinished(tmp_path, "aimbot_yolo26n") == tmp_path / "aimbot_yolo26n2"
    assert t.find_unfinished(tmp_path, "aimbot_yolo26n_quick") == tmp_path / "aimbot_yolo26n_quick"
    ckpt("aimbot_yolo26n3", -1, None, 4000)                # newest finished: nothing to resume
    assert t.find_unfinished(tmp_path, "aimbot_yolo26n") is None


def test_run_logged_streams_output_and_keeps_the_last_lines():
    from aimbot.training.fetch import run_logged

    ok, tail = run_logged([sys.executable, "-c",
                           "import sys; [sys.stderr.write(f'\\r{i}%') for i in range(0, 101, 25)]; print(); print('done')"])
    assert ok and tail.splitlines()[-1] == "done" and "100%" in tail
    ok, tail = run_logged([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    assert not ok and tail == "boom"
    assert run_logged(["no-such-program-xyz"]) == (False, "no-such-program-xyz is not installed")


def test_fetch_git_clones_a_repository_and_updates_it(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not installed")
    from aimbot.training.fetch import fetch_git

    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "train").mkdir()
    (repo / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.com", "PATH": __import__("os").environ["PATH"]}
    for cmd in (["git", "init", "-q"], ["git", "add", "."], ["git", "commit", "-qm", "data"]):
        subprocess.run(cmd, cwd=repo, check=True, env=env)
    msgs = []
    dest = tmp_path / "raw" / "mav_vid"
    assert fetch_git({"url": repo.as_uri()}, dest, msgs.append)
    assert (dest / "train" / "a.txt").exists() and not (tmp_path / "raw" / "mav_vid_clone").exists()
    assert fetch_git({"url": repo.as_uri()}, dest, msgs.append) and "already cloned" in msgs[-1]
    msgs.clear()
    assert not fetch_git({"url": (tmp_path / "nothing-here").as_uri()}, tmp_path / "raw" / "x", msgs.append)
    assert any("git clone failed" in m for m in msgs)


def test_google_drive_quota_falls_back_to_the_mirror(tmp_path, monkeypatch):
    import types

    from aimbot.training import fetch as F

    def refuse(**kw):
        raise RuntimeError("Failed to retrieve file url:\n\n\tToo many users have viewed or downloaded this file recently.")

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=refuse, download_folder=lambda **kw: None))
    got = []

    def fake_download(url, dest, log=print, timeout=120.0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PK")
        got.append((url, dest.name))
        return dest

    monkeypatch.setattr(F, "download", fake_download)
    spec = {"ids": {"train": "a", "val": "b"}, "mirror": {"train": "https://hf.example/x/train.zip",
                                                          "val": "https://hf.example/x/val.zip"}}
    msgs = []
    assert F.fetch_gdrive(spec, tmp_path / "dut", msgs.append)
    assert got == [("https://hf.example/x/train.zip", "train.zip"), ("https://hf.example/x/val.zip", "val.zip")]
    assert any("quota exceeded" in m for m in msgs)
    assert (tmp_path / "dut" / "train" / "train.zip").exists()


def test_download_resumes_after_a_dropped_connection(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from aimbot.training import fetch as F

    data = bytes(range(256)) * 4000
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            calls.append(rng)
            start = int(rng.split("=")[1].split("-")[0]) if rng else 0
            if len(calls) == 1:                   # first try: the connection drops after 300 kB
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data[:300_000])
                self.close_connection = True
                return
            body = data[start:]
            if rng:
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/mav_vid_dataset.zip"
        msgs = []
        out = F.download(url, tmp_path / "mav_vid_dataset.zip", msgs.append)
        assert out.read_bytes() == data and not (tmp_path / "mav_vid_dataset.zip.part").exists()
        assert calls == [None, "bytes=300000-"] and any("continuing at" in m for m in msgs)
        F.download(url, out, msgs.append)                       # complete file: not downloaded again
        assert msgs[-1].endswith("already downloaded") and calls[-1] is None
    finally:
        srv.shutdown()


def test_rmtree_removes_read_only_files(tmp_path):
    import os
    import stat

    from aimbot.training.fetch import rmtree

    d = tmp_path / "mav_vid_clone" / ".git" / "objects" / "pack"
    d.mkdir(parents=True)
    f = d / "tmp_pack_x"
    f.write_bytes(b"x")
    os.chmod(f, stat.S_IREAD)
    rmtree(tmp_path / "mav_vid_clone")
    assert not (tmp_path / "mav_vid_clone").exists()
    rmtree(tmp_path / "missing")                      # no error for a folder that is not there


def _git_http_server(root):
    """A local smart-HTTP git server: git http-backend behind a tiny CGI bridge."""
    import os
    import shutil
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    if shutil.which("git") is None:
        pytest.skip("git not installed")
    exec_path = subprocess.run(["git", "--exec-path"], capture_output=True, text=True).stdout.strip()
    backend = Path(exec_path) / ("git-http-backend.exe" if os.name == "nt" else "git-http-backend")
    if not backend.exists():
        pytest.skip("git http-backend not available")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _cgi(self, method):
            path, _, query = self.path.partition("?")
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else b""
            env = {**os.environ, "GIT_PROJECT_ROOT": str(root), "GIT_HTTP_EXPORT_ALL": "1", "PATH_INFO": path,
                   "QUERY_STRING": query, "REQUEST_METHOD": method, "REMOTE_ADDR": "127.0.0.1",
                   "CONTENT_TYPE": self.headers.get("Content-Type", ""), "CONTENT_LENGTH": str(len(body))}
            out = subprocess.run([str(backend)], input=body, env=env, capture_output=True).stdout
            sep = b"\r\n\r\n" if b"\r\n\r\n" in out else b"\n\n"
            head, _, payload = out.partition(sep)
            status, headers = 200, []
            for line in head.replace(b"\r\n", b"\n").split(b"\n"):
                k, _, v = line.decode().partition(":")
                if k.lower() == "status":
                    status = int(v.split()[0])
                elif k:
                    headers.append((k, v.strip()))
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._cgi("GET")

        def do_POST(self):
            self._cgi("POST")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_gitpack_downloads_files_like_git_clone(tmp_path, monkeypatch):
    import os
    import subprocess

    from aimbot.training import gitpack

    root = tmp_path / "srv"
    repo = root / "data.git"
    work = tmp_path / "work"
    (work / "train" / "img").mkdir(parents=True)
    big = os.urandom(300_000)                                   # "big": goes through a file on disk
    (work / "mav_vid_dataset.zip").write_bytes(big)
    (work / "train" / "img" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (work / "README.md").write_text("MAV-VID\n" * 50)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    for cmd in (["git", "init", "-q"], ["git", "add", "."], ["git", "commit", "-qm", "v1"]):
        subprocess.run(cmd, cwd=work, check=True, env=env)
    (work / "README.md").write_text("MAV-VID v2\n" * 50)        # a second commit: only the newest is fetched
    subprocess.run(["git", "commit", "-qam", "v2"], cwd=work, check=True, env=env)
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(repo)], check=True, env=env)
    monkeypatch.setattr(gitpack, "MEM_LIMIT", 100_000)
    srv = _git_http_server(root)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/data.git"
        msgs = []
        files = gitpack.fetch_repo_files(url, tmp_path / "raw" / "mav_vid", msgs.append)
    finally:
        srv.shutdown()
    dest = tmp_path / "raw" / "mav_vid"
    assert sorted(p.relative_to(dest.resolve()).as_posix() for p in files) == [
        "README.md", "mav_vid_dataset.zip", "train/img/a.txt"]
    assert (dest / "mav_vid_dataset.zip").read_bytes() == big
    assert (dest / "README.md").read_text().startswith("MAV-VID v2")
    assert not (dest / "_gitpack_tmp").exists() and any("complete" in m for m in msgs)


def test_seven_zip_is_found_and_used_for_split_archives(tmp_path, monkeypatch):
    from aimbot.training import fetch as F

    monkeypatch.setattr(F.shutil, "which", lambda name: None)
    prog = tmp_path / "Program Files"
    monkeypatch.setenv("ProgramFiles", str(prog))
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    assert F.sevenzip_exe() is None
    (prog / "7-Zip").mkdir(parents=True)
    (prog / "7-Zip" / "7z.exe").write_bytes(b"")
    exe = F.sevenzip_exe()
    assert exe == str(prog / "7-Zip" / "7z.exe")
    calls = []

    def fake_run(cmd, env=None):
        calls.append(cmd)
        out = Path(next(c for c in cmd if c.startswith("-o"))[2:])
        (out / "imgs").mkdir(parents=True, exist_ok=True)
        (out / "imgs" / "a.jpg").write_bytes(b"jpg")
        return True, "Everything is Ok"

    monkeypatch.setattr(F, "run_logged", fake_run)
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (1, 2):
        (raw / f"train.7z.{i:03d}").write_bytes(b"x")
    assert F.extract_all(raw, log=lambda *a: None) == 1
    assert calls[0][:3] == [exe, "x", "-y"] and calls[0][-1].endswith("train.7z.001")
    assert (raw / "train" / "imgs" / "a.jpg").exists() and (raw / ".train.7z.001.extracted").exists()


def test_log_keeps_the_last_state_of_a_backspace_progress_line():
    assert sink_lines("  5%\b\b\b\b 50%\b\b\b\b100%\n", "Everything is Ok\n") == ["100%", "Everything is Ok"]


def test_loader_processes_fit_the_pc_memory():
    import train_yolo26 as t

    assert t.plan_workers(None, None, 63.7, 32, windows=True) == (8, 4)    # 64 GB: the RTX 4080 SUPER PC
    assert t.plan_workers(None, None, 47.8, 16, windows=True) == (8, 2)
    assert t.plan_workers(None, None, 31.8, 16, windows=True) == (4, 2)
    assert t.plan_workers(None, None, 15.8, 16, windows=True) == (1, 2)
    assert t.plan_workers(None, None, 0, 4, windows=True) == (4, 4)         # memory unknown: by CPU count
    assert t.plan_workers(12, 6, 15.8, 16, windows=True) == (12, 6)         # numbers given are kept ...
    assert t.plan_workers(None, 3, 63.7, 32, windows=True) == (8, 2)        # ... validation even (doubled inside)
    assert t.plan_workers(0, None, 63.7, 32, windows=True) == (0, 0)
    assert t.plan_workers(None, None, 7.5, 2, windows=False) == (1, 2)


class _CallbackModel:
    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def fire(self, event):
        for fn in self.callbacks.get(event, []):
            fn(None)


def test_watchdog_reports_a_stall_once_and_never_after_training():
    import time
    import train_yolo26 as t

    stalls = []
    m = _CallbackModel()
    t.Watchdog(0.5 / 60, stalls.append, check_every=0.05).attach(m)
    assert set(t.Watchdog.EVENTS) <= set(m.callbacks) and "on_train_end" in m.callbacks
    time.sleep(0.8)
    assert stalls == []                        # nothing is watched before the first epoch starts
    m.fire("on_train_epoch_start")
    for _ in range(20):                        # steady progress
        m.fire("on_train_batch_end")
        time.sleep(0.05)
    assert stalls == []
    time.sleep(1.2)                            # progress stops
    assert len(stalls) == 1 and stalls[0] >= 0.5

    done = []
    m2 = _CallbackModel()
    t.Watchdog(0.3 / 60, done.append, check_every=0.05).attach(m2)
    m2.fire("on_val_batch_end")
    m2.fire("on_train_end")                    # training finished: the evaluation afterwards is not watched
    time.sleep(0.8)
    assert done == []


def test_stalled_run_ends_its_loader_processes_with_exit_code_75(tmp_path):
    psutil = pytest.importorskip("psutil")
    import subprocess
    import textwrap
    import train_yolo26 as t

    script = tmp_path / "stall.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        sys.path.insert(0, {str(ROOT / "tools")!r})
        sys.path.insert(0, {str(ROOT)!r})
        import train_yolo26 as t
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print(child.pid, flush=True)
        class Model:
            def add_callback(self, event, fn):
                if event == "on_train_batch_end":
                    self.beat = fn
        m = Model()
        t.Watchdog(0.01, t.stop_stalled_run, check_every=0.1).attach(m)
        m.beat(None)
        time.sleep(60)                         # hung: the watchdog has to end this process
        print("still running", flush=True)
    """))
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert p.returncode == t.STALL_EXIT, p.stderr
    assert "No training progress" in p.stderr and "still running" not in p.stdout
    pid = int(p.stdout.split()[0])
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_trainer_gives_validation_its_own_loader_count(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    import os
    import cv2
    import numpy as np
    import train_yolo26 as t

    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        for i in range(12):
            cv2.imwrite(str(tmp_path / "images" / split / f"{i}.jpg"), np.full((64, 64, 3), 20 * i, np.uint8))
            (tmp_path / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    data = tmp_path / "d.yaml"
    data.write_text(f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    tr = t.make_trainer(6, 2)(overrides={"model": "yolo26n.yaml", "data": str(data), "device": "cpu", "imgsz": 64,
                                         "batch": 2, "epochs": 1, "workers": 8, "plots": False,
                                         "project": str(tmp_path / "runs"), "name": "w"})
    tr.model = tr.get_model(cfg="yolo26n.yaml", verbose=False)
    train_dir, val_dir = str(tmp_path / "images" / "train"), str(tmp_path / "images" / "val")
    assert tr.args.workers == 0                # CPU: Ultralytics loads in the training process ...
    assert tr.get_dataloader(train_dir, 2, -1, "train").num_workers == 0      # ... and so does this trainer
    tr.args.workers = 8                        # as on a GPU
    assert tr.get_dataloader(train_dir, 2, -1, "train").num_workers == 6
    assert tr.get_dataloader(val_dir, 2, -1, "val").num_workers == 2          # Ultralytics alone: 6 (2 x 8, max 6 batches)
    assert tr.args.workers == 8


def test_subset_takes_every_dataset_and_clip_evenly(tmp_path):
    import train_yolo26 as t
    import yaml

    for split, n in (("train", 60), ("val", 20)):
        for src, share in (("anti_uav", 3), ("halmstad", 1)):
            d = tmp_path / "images" / split / src
            d.mkdir(parents=True)
            for i in range(n * share // 4):
                (d / f"{src}_clip{i // 10:02d}_{i:05d}.jpg").write_bytes(b"x")
    data = tmp_path / "aimbot_drones.yaml"
    data.write_text(f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
    y, counts = t.subset_data(data, 20, None, "aimbot_yolo26n_test")
    assert counts == {"train": (20, 60), "val": (20, 20)}
    picked = (tmp_path / "subsets" / "aimbot_yolo26n_test_train.txt").read_text().split()
    assert sum("anti_uav" in p for p in picked) == 15 and sum("halmstad" in p for p in picked) == 5
    assert len({Path(p).name.split("_")[-2] for p in picked if "anti_uav" in p}) == 5      # all 5 clips
    d = yaml.safe_load(y.read_text())
    assert d["names"] == {0: "drone"} and Path(d["val"]) == tmp_path / "images" / "val"  # all val images kept
    assert t.even_sample(list(range(10)), None) == list(range(10)) and t.even_sample(list(range(10)), 3) == [0, 3, 6]


def test_video_file_plays_through_the_webcam_config(tmp_path):
    import subprocess
    import cv2
    import numpy as np

    clip = tmp_path / "flight #2 (test).mp4"            # characters a YAML override would mangle
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(60):
        f = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(f, (40 + 4 * i, 120), 12, (0, 128, 255), -1)
        w.write(f)
    w.release()
    cmd = [sys.executable, "-m", "aimbot", "-c", "webcam", "--video", str(clip), "--headless", "--no-gateway",
           "--duration", "3", "--set", "detector.backend=color", "--set", "logging.auto_record=false",
           "--log-level", "WARNING"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    rates = [float(line.split(" cam ")[1].split()[0]) for line in p.stdout.splitlines() if " cam " in line]
    assert rates and max(rates) > 20                      # frames arrive at the clip's 30 fps
    missing = subprocess.run(cmd[:6] + [str(tmp_path / "nope.mp4")] + cmd[7:], cwd=ROOT, capture_output=True,
                             text=True, timeout=120)
    assert missing.returncode == 2 and "video not found" in missing.stderr
