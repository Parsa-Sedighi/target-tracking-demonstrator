"""Download the files of a git repository over git's smart HTTP protocol, without git.

Git for Windows cannot unpack a single file larger than 4 GB ("pack has bad object ... inflate
returned 1"), and that is exactly what the MAV-VID repository holds: one 11 GB zip. This module
makes the same request as ``git clone --depth 1``, inflates the pack as it arrives and writes the
files straight to disk. Every file is checked against its SHA-1 from the repository tree, and the
pack against its own checksum.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Callable, Optional

UA = "git/2.45.0 (AimBot-dataset-fetch)"
MEM_LIMIT = 64 << 20                # objects up to this size stay in memory (commits, trees, small files)
TYPES = {1: b"commit", 2: b"tree", 3: b"blob", 4: b"tag"}


def pkt(line: bytes) -> bytes:
    return b"%04x" % (len(line) + 4) + line


class Packets:
    """pkt-line reader over a binary stream; read() returns None for a flush-pkt."""

    def __init__(self, fp):
        self.fp = fp

    def _exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.fp.read(n - len(data))
            if not chunk:
                raise IOError("the server closed the connection")
            data += chunk
        return data

    def read(self) -> Optional[bytes]:
        n = int(self._exact(4), 16)
        return None if n == 0 else self._exact(n - 4)


class PackStream:
    """The pack bytes (band 1) of a side-band response; band 2 is progress, band 3 an error."""

    def __init__(self, packets: Packets, first: bytes):
        self.packets = packets
        self.pending: Optional[bytes] = first
        self.buf = b""
        self.done = False

    def _fill(self) -> bool:
        while not self.done:
            p = self.pending if self.pending is not None else self.packets.read()
            self.pending = None
            if p is None:
                self.done = True
                return False
            band, payload = p[0], p[1:]
            if band == 1:
                self.buf += payload
                return True
            if band == 3:
                raise IOError("server: " + payload.decode("utf-8", "replace").strip())
        return False

    def read(self, n: int) -> bytes:
        while len(self.buf) < n and self._fill():
            pass
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def read_some(self, n: int = 1 << 20) -> bytes:
        if not self.buf:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def unread(self, data: bytes) -> None:
        self.buf = data + self.buf


def apply_delta(base: bytes, delta: bytes) -> bytes:
    def varint(i):
        v = shift = 0
        while True:
            b = delta[i]
            i += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return v, i

    _, i = varint(0)
    size, i = varint(i)
    out = bytearray()
    while i < len(delta):
        op = delta[i]
        i += 1
        if op & 0x80:                       # copy from the base
            off = ln = 0
            for k in range(4):
                if op & (1 << k):
                    off |= delta[i] << (8 * k)
                    i += 1
            for k in range(3):
                if op & (1 << (4 + k)):
                    ln |= delta[i] << (8 * k)
                    i += 1
            out += base[off:off + (ln or 0x10000)]
        elif op:                            # insert literal bytes
            out += delta[i:i + op]
            i += op
        else:
            raise IOError("bad delta instruction")
    if len(out) != size:
        raise IOError("delta result has the wrong size")
    return bytes(out)


def advertised_head(url: str, timeout: float = 120.0) -> tuple[str, set]:
    req = urllib.request.Request(f"{url}/info/refs?service=git-upload-pack", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        packets = Packets(r)
        refs: dict = {}
        caps: set = set()
        while True:
            line = packets.read()
            if line is None:
                if refs:
                    break
                continue                    # the flush after "# service=git-upload-pack"
            if line.startswith(b"#"):
                continue
            if b"\0" in line:
                line, c = line.split(b"\0", 1)
                caps = set(c.decode("utf-8", "replace").split())
            sha, name = line.strip().split(b" ", 1)
            refs[name.decode()] = sha.decode()
    head = refs.get("HEAD") or refs.get("refs/heads/master") or refs.get("refs/heads/main")
    if not head:
        raise IOError("the repository advertises no default branch")
    return head, caps


def fetch_repo_files(url: str, dest: Path, log: Callable = print, timeout: float = 300.0) -> list:
    """Write every file of the default branch of `url` (https://host/owner/repo.git) into dest."""
    url = url.rstrip("/")
    head, caps = advertised_head(url, timeout)
    opts = ["side-band-64k" if "side-band-64k" in caps else "side-band", "no-progress", "agent=aimbot"]
    if "ofs-delta" in caps:
        opts.append("ofs-delta")
    body = pkt(f"want {head} {' '.join(opts)}\n".encode()) + pkt(b"deepen 1\n") + b"0000" + pkt(b"done\n")
    req = urllib.request.Request(f"{url}/git-upload-pack", data=body, headers={
        "User-Agent": UA, "Content-Type": "application/x-git-upload-pack-request",
        "Accept": "application/x-git-upload-pack-result"})
    dest.mkdir(parents=True, exist_ok=True)
    tmpdir = dest / "_gitpack_tmp"
    tmpdir.mkdir(exist_ok=True)
    objects: dict = {}                      # sha -> (type, bytes | Path)
    sha_at: dict = {}                       # pack offset -> sha (for OFS_DELTA bases)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        packets = Packets(r)
        while True:                         # shallow lines, flush, NAK; then the side-band pack
            p = packets.read()
            if p is None or p.startswith((b"shallow ", b"unshallow ", b"NAK", b"ACK")):
                continue
            if p.startswith(b"ERR "):
                raise IOError("server: " + p[4:].decode("utf-8", "replace").strip())
            break
        s = PackStream(packets, p)
        psha = hashlib.sha1()               # the pack's own checksum covers every byte before the trailer

        def take(n: int) -> bytes:
            b = s.read(n)
            if len(b) < n:
                raise IOError("the download ended early")
            psha.update(b)
            return b

        sig, _ver, count = struct.unpack(">4sII", take(12))
        if sig != b"PACK":
            raise IOError("the server did not send a git pack")
        offset = 12
        for index in range(count):
            start = offset
            c = take(1)[0]
            used = 1
            typ, size, shift = (c >> 4) & 7, c & 15, 4
            while c & 0x80:
                c = take(1)[0]
                used += 1
                size |= (c & 0x7F) << shift
                shift += 7
            base_sha = None
            if typ == 6:                    # OFS_DELTA: base given as a backwards offset
                c = take(1)[0]
                used += 1
                rel = c & 0x7F
                while c & 0x80:
                    c = take(1)[0]
                    used += 1
                    rel = ((rel + 1) << 7) | (c & 0x7F)
                base_sha = sha_at.get(start - rel)
            elif typ == 7:                  # REF_DELTA: base given by name
                base_sha = take(20).hex()
                used += 20
            offset += used
            whole = typ in TYPES
            h = hashlib.sha1(TYPES[typ] + b" %d\0" % size) if whole else None
            to_disk = whole and typ == 3 and size > MEM_LIMIT
            part = tmpdir / f"obj{index}"
            sink = open(part, "wb") if to_disk else None
            mem = bytearray()
            d = zlib.decompressobj()
            got, t_last = 0, time.time()
            try:
                while not d.eof:
                    chunk = s.read_some(1 << 20)
                    if not chunk:
                        raise IOError("the download ended early")
                    out = d.decompress(chunk)
                    if d.eof and d.unused_data:
                        s.unread(d.unused_data)
                        chunk = chunk[:len(chunk) - len(d.unused_data)]
                    psha.update(chunk)
                    offset += len(chunk)
                    got += len(out)
                    if h:
                        h.update(out)
                    if sink:
                        sink.write(out)
                        if time.time() - t_last > 5:
                            t_last = time.time()
                            log(f"  {size / 1e6:.0f} MB file: {got / 1e6:.0f} MB")
                    else:
                        mem += out
            finally:
                if sink:
                    sink.close()
            if whole:
                if got != size:
                    raise IOError(f"object {index}: {got} bytes instead of {size}")
                sha = h.hexdigest()
                if sink:
                    final = tmpdir / sha
                    os.replace(part, final)
                    objects[sha] = (TYPES[typ], final)
                    log(f"  {size / 1e6:.0f} MB file: complete")
                else:
                    objects[sha] = (TYPES[typ], bytes(mem))
            else:
                btyp, bdata = objects.get(base_sha, (None, None))
                if not isinstance(bdata, bytes):
                    raise IOError("a delta whose base is not in memory (not supported)")
                data = apply_delta(bdata, bytes(mem))
                sha = hashlib.sha1(btyp + b" %d\0" % len(data) + data).hexdigest()
                objects[sha] = (btyp, data)
            sha_at[start] = sha
        if s.read(20) != psha.digest():
            raise IOError("the pack checksum does not match")
    if head not in objects:
        raise IOError("the pack does not contain the commit")
    tree_sha = objects[head][1].split(b"\n", 1)[0].split()[1].decode()
    root = dest.resolve()
    written: list = []

    def walk(tsha: str, rel: Path) -> None:
        data = objects[tsha][1]
        i = 0
        while i < len(data):
            sp = data.index(b" ", i)
            nul = data.index(b"\0", sp)
            mode, name = data[i:sp], data[sp + 1:nul].decode("utf-8", "replace")
            sha = data[nul + 1:nul + 21].hex()
            i = nul + 21
            target = (dest / rel / name).resolve()
            if root not in target.parents:
                continue                    # never write outside dest
            if mode == b"40000":
                walk(sha, rel / name)
            elif mode != b"160000" and sha in objects:
                target.parent.mkdir(parents=True, exist_ok=True)
                blob = objects[sha][1]
                if isinstance(blob, Path):
                    os.replace(blob, target)
                else:
                    target.write_bytes(blob)
                written.append(target)

    walk(tree_sha, Path("."))
    for leftover in tmpdir.iterdir():
        leftover.unlink()
    tmpdir.rmdir()
    return written
