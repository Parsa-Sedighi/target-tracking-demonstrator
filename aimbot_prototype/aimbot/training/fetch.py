"""Download the team README datasets into datasets/raw/<key>/.

Automatic: Halmstad (GitHub), Drone-vs-Bird annotations (GitHub), Drone Hunter (OSF API),
DUT Anti-UAV and Anti-UAV300 (Google Drive, needs ``pip install gdown``), Roboflow
Indoor-drones (free API key), MAV-VID (Bitbucket mirror, if it clones).
Manual: Det-Fly (OneDrive/Baidu), Drone-vs-Bird videos (by e-mail request), anything
whose automatic download fails (the message says what to put where).

Archives (.zip/.tar*/.7z/.rar, and 7-Zip split archives name.7z.001, .002, ...) found in a dataset
folder are extracted in place.
"""

from __future__ import annotations

import bisect
import fnmatch
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from .sources import IMG_EXT, SOURCES, VID_EXT

UA = "AimBot-dataset-fetch/1.0"

DATASETS: dict[str, dict] = {
    "halmstad": {
        "method": "git_sparse", "url": "https://github.com/DroneDetectionThesis/Drone-detection-dataset.git",
        "sparse": ["Data/Video_IR", "Data/Video_V"],
        "zip": "https://github.com/DroneDetectionThesis/Drone-detection-dataset/archive/refs/heads/master.zip",
        "size": "~200 MB (videos + labels; the audio part is skipped)",
    },
    "mav_vid": {
        # The Bitbucket repository holds one file, mav_vid_dataset.zip (~11 GB). Git for Windows cannot unpack a
        # single file over 4 GB ("pack has bad object ... inflate returned 1"), so it is fetched over git's HTTP
        # protocol by aimbot/training/gitpack.py (same request as git clone --depth 1, checked by SHA-1).
        "method": "gitpack", "url": "https://bitbucket.org/alejodosr/mav-vid-dataset.git", "size": "~11 GB (one zip)",
        "manual": ("Kaggle page is offline; the mirror is https://bitbucket.org/alejodosr/mav-vid-dataset "
                   "(linked from github.com/alejodosr/adaptive-inattention). Download it there and unzip into "
                   "{dest} so that it contains train/img/*.jpg+.txt and val/img/*.jpg+.txt."),
    },
    "drone_hunter": {
        "method": "osf", "node": "n9q78", "size": "~58k images (7-Zip volumes) + flight videos",
        "skip": ["*-master.zip"],             # snapshots of the authors' code repositories, no data
        "manual": "Open https://osf.io/n9q78/ , download the files (Files tab) and unzip them into {dest}.",
    },
    "dut_anti_uav": {
        "method": "gdrive", "size": "10,000 images",
        "ids": {"train": "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj", "val": "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n",
                "test": "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0"},
        # the same three archives re-uploaded to Hugging Face (public), used when Google Drive refuses
        # ("Too many users have viewed or downloaded this file recently")
        "mirror": {s: f"https://huggingface.co/datasets/DestinyLove/DUT-Anti-UAV/resolve/main/DUT-Anti-UAV/{s}.zip"
                   for s in ("train", "val", "test")},
        "manual": ("Download train / val / test from the Google Drive or Baidu links in "
                   "https://github.com/wangdongdut/DUT-Anti-UAV and put the archives (or folders) in {dest}/train, "
                   "{dest}/val and {dest}/test."),
    },
    "roboflow_indoor": {
        "method": "roboflow", "workspace": "drones-l7ol3", "project": "indoor-drones-2ughv", "format": "yolov8",
        "size": "small",
        "manual": ("Create a free account at roboflow.com, copy your Private API Key (Settings -> API Keys), then "
                   "double-click fetch_roboflow.bat and paste it (or run python tools/fetch_datasets.py "
                   "roboflow_indoor --roboflow-key <KEY>, or set ROBOFLOW_API_KEY). Or download the YOLOv8 export from "
                   "https://universe.roboflow.com/drones-l7ol3/indoor-drones-2ughv and unzip it into {dest}."),
    },
    "det_fly": {
        "method": "manual", "size": "13,271 4K images (~15 GB)",
        "manual": ("Det-Fly is on OneDrive / Baidu only (links in https://github.com/Jake-WU/Det-Fly): download "
                   "'JPEGImages' and 'Annotations' and unzip both into {dest}."),
    },
    "anti_uav": {
        "method": "gdrive", "size": "~7 GB (318 RGB + IR video pairs)",
        "ids": {"Anti-UAV300": "1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr"},
        "manual": ("Download Anti-UAV300 from the Google Drive / Baidu link in https://github.com/ucas-vg/Anti-UAV "
                   "(or https://github.com/ZhaoJ9014/Anti-UAV) and unzip it into {dest}."),
    },
    "drone_vs_bird": {
        "method": "git", "url": "https://github.com/wosdetc/challenge.git", "size": "annotations ~2 MB; videos by request",
        "manual": ("The annotations are downloaded automatically. The videos come after signing the data usage "
                   "agreement: e-mail wosdetc@googlegroups.com, then put the video files (any folder layout) in "
                   "{dest}/videos. Each video must have the same name as its annotation .txt."),
    },
    "team": {
        "method": "manual", "size": "yours",
        "manual": ("Record with the real camera (logging.frames.mode: png), export frames (tools/export_frames.py), "
                   "label them in YOLO format with the class ids of datasets/aimbot.yaml and put them in "
                   "{dest}/images and {dest}/labels."),
    },
}


# ----------------------------------------------------------------------------- utilities
def http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_json(url: str, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def download(url: str, dest: Path, log: Callable = print, timeout: float = 120.0, retries: int = 4) -> Path:
    """Stream a URL to a file. A complete file of the same size is skipped. A dropped connection is
    retried, and an interrupted download (<name>.part) continues where it stopped if the server allows."""
    import http.client

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = tmp.stat().st_size if (tmp.exists() and not dest.exists()) else 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                partial = r.status == 206
                length = int(r.headers.get("Content-Length") or 0)
                m = re.search(r"/(\d+)\s*$", r.headers.get("Content-Range") or "")
                total = int(m.group(1)) if (partial and m) else (have + length if partial else length)
                if not partial:
                    have = 0
                    if dest.exists() and total and dest.stat().st_size == total:
                        log(f"  {dest.name}: already downloaded")
                        return dest
                elif have:
                    log(f"  {dest.name}: continuing at {have / 1e6:.0f} of {total / 1e6:.0f} MB")
                done, t_last = have, 0.0
                with tmp.open("ab" if partial else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - t_last > 5:
                            t_last = time.time()
                            log(f"  {dest.name}: {done / 1e6:.0f}" + (f" / {total / 1e6:.0f} MB" if total else " MB"))
            if total and done < total:
                raise IOError(f"the connection closed after {done / 1e6:.0f} of {total / 1e6:.0f} MB")
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and tmp.exists():      # the .part does not fit the file any more: start over
                tmp.unlink()
                continue
            raise
        except (OSError, http.client.HTTPException) as exc:
            if attempt == retries - 1:
                raise
            log(f"  {dest.name}: {exc or exc.__class__.__name__}; trying again in 10 s")
            time.sleep(10)
            continue
        tmp.replace(dest)
        log(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")
        return dest
    raise IOError(f"{dest.name}: download failed")


def _safe_members(names, base: Path):
    base = base.resolve()
    for n in names:
        target = (base / n).resolve()
        if base == target or base in target.parents:
            yield n


ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".7z", ".rar")
_VOLUME = re.compile(r"(?P<base>.+\.(?P<kind>7z|zip))\.(?P<num>\d{3})", re.IGNORECASE)


def _volume_info(p: Path) -> Optional[tuple[str, str, int]]:
    """('name.7z', '7z', 2) for 'name.7z.002' (a 7-Zip split archive), else None."""
    m = _VOLUME.fullmatch(p.name)
    return (m["base"], m["kind"].lower(), int(m["num"])) if m else None


def is_archive(p: Path) -> bool:
    if not p.is_file() or p.name.endswith(".part"):
        return False
    vol = _volume_info(p)
    if vol:
        return vol[2] == 1                    # a split archive is extracted through its first volume
    return p.name.lower().endswith(ARCHIVE_EXT)


def volume_set(first: Path) -> tuple[list[Path], str]:
    """The volumes name.7z.001, .002, ... in order, or ([], why the set cannot be extracted yet)."""
    base = _volume_info(first)[0]
    prefix = base.lower() + "."
    nums: dict[int, Path] = {}
    for q in first.parent.iterdir():
        low = q.name.lower()
        if not low.startswith(prefix):
            continue
        rest = low[len(prefix):]
        if rest.endswith(".part"):
            return [], f"{q.name} did not finish downloading (run the download again)"
        if len(rest) == 3 and rest.isdigit():
            nums[int(rest)] = q
    missing = [f"{base}.{i:03d}" for i in range(1, max(nums, default=0) + 1) if i not in nums]
    if missing or not nums:
        return [], "missing " + (", ".join(missing) or base + ".001")
    return [nums[i] for i in sorted(nums)], ""


class JoinedVolumes(io.RawIOBase):
    """The volumes of a split archive read as one seekable file (no joined copy on disk)."""

    def __init__(self, paths: list):
        self._files = [open(p, "rb") for p in paths]
        self._sizes = [os.path.getsize(p) for p in paths]
        self._starts = [0]
        for size in self._sizes[:-1]:
            self._starts.append(self._starts[-1] + size)
        self._total = sum(self._sizes)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        start = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._total}[whence]
        self._pos = max(0, start + offset)
        return self._pos

    def readinto(self, b) -> int:
        mv = memoryview(b).cast("B")
        n = 0
        while n < len(mv) and self._pos < self._total:
            i = bisect.bisect_right(self._starts, self._pos) - 1
            off = self._pos - self._starts[i]
            f = self._files[i]
            f.seek(off)
            k = f.readinto(mv[n:n + min(len(mv) - n, self._sizes[i] - off)])
            if not k:
                break
            n += k
            self._pos += k
        return n

    def close(self) -> None:
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        super().close()


def _skipped(p: Path, root: Optional[Path], patterns) -> bool:
    if not patterns:
        return False
    rel = p.relative_to(root).as_posix() if root else p.name
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in patterns)


def sevenzip_exe() -> Optional[str]:
    """7-Zip's command-line program if it is installed: far faster than py7zr on big archives."""
    for name in ("7z", "7za", "7zz"):
        exe = shutil.which(name)
        if exe:
            return exe
    for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if base and (Path(base) / "7-Zip" / "7z.exe").exists():
            return str(Path(base) / "7-Zip" / "7z.exe")
    return None


def _sevenzip_extract(exe: str, archive: Path, out: Path, log: Callable) -> bool:
    """7z x (split archives: give it the .001 volume). Progress shows in the console."""
    ok, tail = run_logged([exe, "x", "-y", "-bsp1", f"-o{out}", str(archive)])
    if not ok:
        log("  7-Zip failed: " + (tail.strip().splitlines() or ["no output"])[-1])
    return ok


def extract(archive: Path, log: Callable = print) -> Optional[Path]:
    """Extract next to the archive into <stem>/ (once; a marker file records it). Rejects path traversal."""
    name = archive.name
    vol = _volume_info(archive)
    stem = vol[0] if vol else name
    for e in sorted(ARCHIVE_EXT, key=len, reverse=True):
        if stem.lower().endswith(e):
            stem = stem[: -len(e)]
            break
    out = archive.parent / stem
    marker = archive.parent / f".{name}.extracted"
    if marker.exists():
        return out
    parts: list = []
    if vol:
        parts, problem = volume_set(archive)
        if problem:
            log(f"  not extracting {stem} yet: {problem}")
            return None
        log(f"  extracting {stem} ({len(parts)} volumes, {sum(q.stat().st_size for q in parts) / 1e9:.1f} GB) ...")
    else:
        log(f"  extracting {name} ...")
    out.mkdir(parents=True, exist_ok=True)
    low = name.lower()
    try:
        if (vol and vol[1] == "7z") or (not vol and low.endswith(".7z")):
            exe = sevenzip_exe()
            if exe:
                log(f"  with 7-Zip ({exe})")
                if not _sevenzip_extract(exe, archive, out, log):
                    return None
            else:
                try:
                    import py7zr
                except ImportError:
                    log(f"  cannot extract {stem}: install 7-Zip (7-zip.org) or pip install py7zr, then run again")
                    return None
                log("  with py7zr (slow on big archives: installing 7-Zip from 7-zip.org makes this much faster)")
                if vol:
                    with io.BufferedReader(JoinedVolumes(parts), 1 << 20) as fp, py7zr.SevenZipFile(fp) as z:
                        z.extractall(out)
                else:
                    with py7zr.SevenZipFile(archive) as z:
                        z.extractall(out)
        elif vol:                                  # split .zip
            with io.BufferedReader(JoinedVolumes(parts), 1 << 20) as fp, zipfile.ZipFile(fp) as z:
                for m in _safe_members(z.namelist(), out):
                    z.extract(m, out)
        elif low.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                ok = list(_safe_members(z.namelist(), out))
                for m in ok:
                    z.extract(m, out)
        elif low.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")):
            with tarfile.open(archive) as t:
                allm = t.getmembers()
                ok = set(_safe_members([m.name for m in allm], out))
                members = [m for m in allm if m.name in ok and not (m.issym() or m.islnk())]
                try:
                    t.extractall(out, members=members, filter="data")     # Python >= 3.10.12 / 3.11.4
                except TypeError:
                    t.extractall(out, members=members)
        elif low.endswith(".rar"):
            exe = sevenzip_exe()
            if not ((exe and _sevenzip_extract(exe, archive, out, log)) or
                    _run(["unrar", "x", "-o+", str(archive), str(out)])):
                log(f"  cannot extract {name}: install 7-Zip (Windows) or unrar and re-run, or extract it by hand into {out}")
                return None
    except Exception as exc:
        log(f"  extracting {name} failed: {exc}")
        return None
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return out


def extract_all(folder: Path, log: Callable = print, skip=()) -> int:
    n = 0
    for _ in range(3):                        # archives inside archives
        found = [p for p in folder.rglob("*") if is_archive(p) and not (p.parent / f".{p.name}.extracted").exists()
                 and not _skipped(p, folder, skip)]
        if not found:
            break
        for p in found:
            if extract(p, log) is not None:
                n += 1
    return n


def _run(cmd: list, cwd: Optional[Path] = None) -> bool:
    if shutil.which(cmd[0]) is None:
        return False
    try:
        return subprocess.run(cmd, cwd=cwd, check=False).returncode == 0
    except OSError:
        return False


def count_media(folder: Path) -> tuple[int, int]:
    if not folder.exists():
        return 0, 0
    n_img = n_vid = 0
    for p in folder.rglob("*"):
        s = p.suffix.lower()
        if s in IMG_EXT:
            n_img += 1
        elif s in VID_EXT:
            n_vid += 1
    return n_img, n_vid


# ----------------------------------------------------------------------------- methods
def rmtree(path: Path) -> None:
    """shutil.rmtree that also removes read-only files (git marks its pack files read-only on Windows)."""
    import sys

    def retry(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    if Path(path).exists():
        shutil.rmtree(path, **({"onexc": retry} if sys.version_info >= (3, 12) else {"onerror": retry}))


# git never asks for a password here: public repositories need none, and a sign-in window would only
# hide the real error (repository moved, network) from the log
GIT = ["git", "-c", "credential.helper=", "-c", "core.longpaths=true"]
GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


def run_logged(cmd: list, env: Optional[dict] = None) -> tuple[bool, str]:
    """Run a program, show its output (progress included) on the console and in the log; (ok, last lines)."""
    if shutil.which(cmd[0]) is None:
        return False, f"{cmd[0]} is not installed"
    import sys

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env={**os.environ, **(env or {})})
    except OSError as exc:
        return False, str(exc)
    tail: list = []
    pending = ""
    while True:
        chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
        if not chunk:
            break
        text = chunk.decode("utf-8", "replace")
        try:
            sys.stderr.write(text)
            sys.stderr.flush()
        except Exception:
            pass
        pending += text
        *lines, pending = re.split(r"[\r\n]", pending)
        tail = (tail + [ln for ln in lines if ln.strip()])[-12:]
    if pending.strip():
        tail = (tail + [pending])[-12:]
    return proc.wait() == 0, "\n".join(tail)


def fetch_git(spec: dict, dest: Path, log: Callable) -> bool:
    if (dest / ".git").exists():
        log(f"  {dest} already cloned; updating")
        run_logged(GIT + ["-C", str(dest), "pull", "--ff-only"], GIT_ENV)
        return True
    if shutil.which("git") is None:
        log("  git is not installed")
        return fetch_archive(spec, dest, log)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + "_clone")
    ok, why = False, ""
    for attempt in range(2):                  # big repositories over HTTPS sometimes drop once
        rmtree(tmp)
        if spec.get("sparse"):
            ok, why = run_logged(GIT + ["clone", "--progress", "--depth", "1", "--filter=blob:none", "--sparse",
                                        spec["url"], str(tmp)], GIT_ENV)
            if ok:
                ok, why = run_logged(GIT + ["-C", str(tmp), "sparse-checkout", "set", *spec["sparse"]], GIT_ENV)
        else:
            ok, why = run_logged(GIT + ["clone", "--progress", "--depth", "1", spec["url"], str(tmp)], GIT_ENV)
        # a retry only helps a dropped connection; a missing repository or a file git cannot handle
        # ("pack has bad object": Git for Windows and single files over 4 GB) fails the same way again
        if ok or re.search(r"not found|could not read Username|Authentication failed|403|bad object|inflate returned",
                           why, re.I):
            break
        log("  git clone failed, trying once more ...")
    if not ok:
        rmtree(tmp)
        log("  git clone failed: " + (why.strip().splitlines() or ["no output"])[-1])
        return fetch_archive(spec, dest, log)
    if dest.exists():                         # keep files the user already put there (e.g. DvB videos)
        for p in tmp.iterdir():
            target = dest / p.name
            if not target.exists():
                shutil.move(str(p), str(target))
        rmtree(tmp)
    else:
        tmp.rename(dest)
    return True


def fetch_gitpack(spec: dict, dest: Path, log: Callable) -> bool:
    """A git repository's files without git (see gitpack.py); skipped when they are already there."""
    from .gitpack import fetch_repo_files

    stale = dest.with_name(dest.name + "_clone")     # left by an earlier git clone that failed
    if stale.exists():
        log(f"  removing {stale.name} (an earlier git clone that could not finish)")
        rmtree(stale)
    have = [p for p in dest.iterdir() if p.is_file() and not p.name.startswith(".")] if dest.exists() else []
    if have:
        log(f"  already downloaded: {', '.join(p.name for p in have)}")
        return True
    log(f"  downloading {spec['url']} over git's HTTP protocol")
    try:
        files = fetch_repo_files(spec["url"], dest, log)
    except Exception as exc:
        log(f"  download failed: {exc}. Run fetch_datasets.bat again to retry.")
        rmtree(dest / "_gitpack_tmp")
        return False
    log("  " + ", ".join(f"{p.name} ({p.stat().st_size / 1e9:.2f} GB)" for p in files))
    return bool(files)


def fetch_files(spec: dict, dest: Path, log: Callable) -> bool:
    """Download each URL in spec['files'] into dest (archives are extracted afterwards)."""
    ok = bool(spec.get("files"))
    for url in spec.get("files", []):
        name = urllib.parse.unquote(url.rstrip("/").rsplit("/", 1)[-1])
        try:
            download(url, dest / name, log, timeout=300)
        except Exception as exc:
            log(f"  {name}: download failed ({exc})")
            ok = False
    return ok


def fetch_archive(spec: dict, dest: Path, log: Callable) -> bool:
    """Fallback for a git source: download the repository as one archive and extract it into dest."""
    url = spec.get("archive")
    if not url:
        return False
    log(f"  downloading the repository as an archive instead: {url}")
    z = dest / "_download" / (url.rstrip("/").rsplit("/", 1)[-1] or "repo.zip")
    try:
        download(url, z, log, timeout=600)
    except Exception as exc:
        log(f"  archive download failed: {exc}")
        return False
    return extract(z, log) is not None


def fetch_github_zip(spec: dict, dest: Path, log: Callable) -> bool:
    url = spec.get("zip")
    if not url:
        return False
    z = dest / "_download" / "repo.zip"
    try:
        download(url, z, log)
    except Exception as exc:
        log(f"  zip download failed: {exc}")
        return False
    with zipfile.ZipFile(z) as zf:
        keep = [n for n in zf.namelist() if any(f"/{s}/" in n for s in spec.get("sparse", [""]))]
        for m in _safe_members(keep, dest):
            zf.extract(m, dest)
    rmtree(dest / "_download")
    return True


def fetch_osf(spec: dict, dest: Path, log: Callable) -> bool:
    """Recursively download every file of an OSF project (and its components) via the public API."""
    api = "https://api.osf.io/v2"
    n_files = 0
    failed: list = []
    skip = spec.get("skip", [])

    def walk_folder(url: str, rel: Path):
        nonlocal n_files
        while url:
            _, j = http_json(url)
            for item in j.get("data", []):
                at = item.get("attributes", {})
                name = at.get("name") or item.get("id")
                if at.get("kind") == "folder":
                    href = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                    if href:
                        walk_folder(href, rel / name)
                elif at.get("kind") == "file":
                    link = item.get("links", {}).get("download")
                    target = dest / rel / name
                    size = at.get("size") or 0
                    if _skipped(target, dest, skip):
                        continue
                    if target.exists() and size and target.stat().st_size == size:
                        continue
                    if link:
                        try:
                            download(link, target, log)
                            n_files += 1
                        except Exception as exc:     # retried on the next run; the other files go on
                            failed.append(name)
                            log(f"  {name}: download failed ({exc})")
            url = (j.get("links") or {}).get("next")

    def walk_node(node: str, rel: Path):
        _, prov = http_json(f"{api}/nodes/{node}/files/")
        for p in prov.get("data", []):
            href = p.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
            if href:
                walk_folder(href, rel)
        try:
            _, kids = http_json(f"{api}/nodes/{node}/children/")
            for k in kids.get("data", []):
                title = (k.get("attributes") or {}).get("title") or k.get("id")
                walk_node(k["id"], rel / "".join(c if c.isalnum() or c in " -_." else "_" for c in title))
        except urllib.error.HTTPError:
            pass

    try:
        walk_node(spec["node"], Path("."))
    except Exception as exc:
        log(f"  OSF download failed: {exc}")
        return False
    log(f"  {n_files} files downloaded from OSF")
    if failed:
        log(f"  {len(failed)} file(s) failed: {', '.join(failed)}. Run fetch_datasets.bat again to retry them.")
    return not failed


def fetch_gdrive(spec: dict, dest: Path, log: Callable) -> bool:
    try:
        import gdown
    except ImportError:
        log("  needs gdown: pip install gdown")
        return False
    ok_all = True
    for label, fid in spec["ids"].items():
        sub = dest / label
        sub.mkdir(parents=True, exist_ok=True)
        if any(p for p in sub.iterdir() if not p.name.startswith(".")) and count_media(sub) != (0, 0):
            log(f"  {label}: already present")
            continue
        log(f"  {label}: Google Drive id {fid}")
        out = None
        mirror = (spec.get("mirror") or {}).get(label)
        try:
            out = gdown.download(id=fid, output=str(sub) + os.sep, quiet=False, resume=True)
        except Exception as exc:
            why = "download quota exceeded for today" if "Too many users" in str(exc) else str(exc).strip().splitlines()[0][:200]
            log(f"  {label}: Google Drive refused the download ({why})")
        if not out and mirror:
            log(f"  {label}: downloading the copy at {mirror}")
            try:
                out = download(mirror, sub / mirror.rsplit("/", 1)[-1], log, timeout=300)
            except Exception as exc:
                log(f"  {label}: that download failed too ({exc})")
        elif not out:
            try:
                out = gdown.download_folder(id=fid, output=str(sub), quiet=False, resume=True)
            except Exception as exc:
                log(f"  {label}: folder download failed ({exc})")
        if not out:
            ok_all = False
    return ok_all


def fetch_roboflow(spec: dict, dest: Path, log: Callable, api_key: Optional[str], version: Optional[int] = None) -> bool:
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        n_img = count_media(dest)[0]
        if n_img:
            log(f"  already downloaded ({n_img} images)")
            return True
        log("  needs your free Roboflow API key: double-click fetch_roboflow.bat")
        return False
    api = "https://api.roboflow.com"
    ws, proj, fmt = spec["workspace"], spec["project"], spec["format"]
    q = urllib.parse.urlencode({"api_key": key})
    try:
        _, info = http_json(f"{api}/{ws}/{proj}?{q}")
        versions = [int(str(v.get("id", "")).rsplit("/", 1)[-1]) for v in info.get("versions", []) if v.get("id")]
        if not versions:
            log("  the project has no dataset versions")
            return False
        ver = version or max(versions)
        log(f"  {ws}/{proj}: versions {sorted(versions)}, downloading v{ver} as {fmt}")
        url = f"{api}/{ws}/{proj}/{ver}/{fmt}?{q}&nocache=true"
        for _ in range(600):                  # the export is generated on demand
            status, j = http_json(url)
            if status == 200 and "export" in j:
                link = j["export"]["link"]
                break
            log(f"  export in progress {float(j.get('progress', 0)) * 100:.0f} %")
            time.sleep(2)
        else:
            log("  export did not finish")
            return False
        z = download(link, dest / f"roboflow_v{ver}.zip", log)
        extract(z, log)
        return True
    except urllib.error.HTTPError as exc:
        hint = " (check the API key)" if exc.code in (401, 403) else ""
        log(f"  Roboflow API error {exc.code}{hint}: {exc.read()[:200]!r}".replace(key, "***"))
        return False
    except Exception as exc:
        log(f"  Roboflow download failed: {exc}".replace(key, "***"))
        return False


def fetch(key: str, raw_root: Path, log: Callable = print, roboflow_key: Optional[str] = None,
          roboflow_version: Optional[int] = None) -> bool:
    spec = DATASETS[key]
    dest = raw_root / key
    method = spec["method"]
    log(f"[{key}] {SOURCES[key].title if key in SOURCES else key} ({spec.get('size', '')})")
    ok = False
    if method == "git_sparse":
        ok = fetch_git(spec, dest, log) or fetch_github_zip(spec, dest, log)
    elif method == "git":
        ok = fetch_git(spec, dest, log)
    elif method == "files":
        ok = fetch_files(spec, dest, log) or (bool(spec.get("url")) and fetch_git(spec, dest, log))
    elif method == "gitpack":
        ok = fetch_gitpack(spec, dest, log)
    elif method == "osf":
        ok = fetch_osf(spec, dest, log)
    elif method == "gdrive":
        ok = fetch_gdrive(spec, dest, log)
    elif method == "roboflow":
        ok = fetch_roboflow(spec, dest, log, roboflow_key, roboflow_version)
    if dest.exists():
        n = extract_all(dest, log, spec.get("skip", ()))
        if n:
            log(f"  extracted {n} archive(s)")
    n_img, n_vid = count_media(dest)
    if not ok or (n_img + n_vid == 0):
        msg = spec.get("manual", "").format(dest=dest)
        if msg:
            log(f"  MANUAL STEP: {msg}")
    if key == "drone_vs_bird" and n_vid == 0:
        log(f"  annotations are in {dest}; the videos still have to be requested (see above)")
    log(f"  -> {dest}: {n_img} images, {n_vid} videos")
    return ok and (n_img + n_vid > 0 or key == "drone_vs_bird")


def status(raw_root: Path) -> list[dict]:
    rows = []
    for key, spec in DATASETS.items():
        dest = raw_root / key
        n_img, n_vid = count_media(dest)
        archives = [p.name for p in dest.rglob("*") if is_archive(p) and not (p.parent / f".{p.name}.extracted").exists()
                    and not _skipped(p, dest, spec.get("skip", ()))] if dest.exists() else []
        if key == "drone_vs_bird" and dest.exists() and n_vid == 0:
            state = "annotations only (videos by request)" if any(dest.rglob("*.txt")) else "missing"
        elif archives:
            state = "archives not extracted"
        elif n_img + n_vid:
            state = "ready"
        elif dest.exists() and any(dest.iterdir()):
            state = "annotations only" if key == "drone_vs_bird" else "no images/videos yet"
        else:
            state = "missing"
        src = SOURCES.get(key)
        rows.append({"key": key, "state": state, "images": n_img, "videos": n_vid, "path": str(dest),
                     "access": src.access if src else spec["method"], "license": src.license if src else "",
                     "homepage": src.homepage if src else ""})
    return rows
