"""Read MATLAB Video Labeler ``groundTruth`` label files (.mat v5) without MATLAB or scipy.

The Halmstad drone dataset stores its boxes as a Computer Vision Toolbox ``groundTruth``
object. MATLAB writes objects as MCOS data in the file's hidden *subsystem* element, which
``scipy.io.loadmat`` returns as an opaque blob. Inside that blob MATLAB keeps the struct
that ``groundTruth.saveobj`` produced::

    struct  fields: DataSource, LabelDefinitions, LabelData, Version
      LabelDefinitions  struct (N x 1)  fields: Name, Type, Group, Description
      LabelData         struct (F x 1)  fields: Time, <label 1>, <label 2>, ...
                        each label cell = M x 4 double [x y w h] (M boxes) or empty

This module parses the MAT5 element tree properly (array flags, dimensions, struct field
names, cells, zlib-compressed elements), finds that struct and returns every box of every
label in every frame. It does not rely on byte-size heuristics, so frames with several boxes
or several labels are read correctly.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

_NUM = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}
_CLASS = {1: "cell", 2: "struct", 3: "object", 4: "char", 5: "sparse", 6: "double", 7: "single", 8: "int8",
          9: "uint8", 10: "int16", 11: "uint16", 12: "int32", 13: "uint32", 14: "int64", 15: "uint64",
          16: "function", 17: "opaque"}
_NUMERIC_CLASSES = {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}


@dataclass
class Node:
    cls: str
    dims: tuple = ()
    name: str = ""
    value: object = None                 # ndarray for numeric, str for char
    fields: list = field(default_factory=list)
    children: list = field(default_factory=list)
    classname: str = ""

    def field_values(self, fname: str) -> list["Node"]:
        """For a struct (array): the values of one field, one per element (column-major)."""
        if self.cls not in ("struct", "object") or fname not in self.fields:
            return []
        k, nf = self.fields.index(fname), len(self.fields)
        return self.children[k::nf]

    def text(self) -> str:
        return self.value if isinstance(self.value, str) else ""


def _elements(buf: bytes, off: int = 0) -> Iterator[tuple[int, bytes]]:
    end = len(buf)
    while off + 8 <= end:
        t, sz = struct.unpack_from("<HH", buf, off)
        if sz != 0 and t < 20:                       # "small data element": 4-byte tag + <=4 bytes
            pay, nxt = off + 4, off + 8
        else:
            t, sz = struct.unpack_from("<II", buf, off)
            pay = off + 8
            nxt = pay + sz + ((8 - (sz % 8)) % 8) if t != 15 else pay + sz   # compressed: no padding
        if t == 0 or t > 20 or pay + sz > end:
            return
        data = buf[pay:pay + sz]
        if t == 15:                                  # miCOMPRESSED: zlib stream holding one element
            yield from _elements(zlib.decompress(data))
        else:
            yield t, data
        off = nxt


def _parse_matrix(payload: bytes, depth: int = 0) -> Node:
    if depth > 40:
        return Node("too_deep")
    subs = list(_elements(payload))
    if not subs:
        return Node("empty", (0, 0))
    flags = np.frombuffer(subs[0][1][:8], "<u4")
    cls = _CLASS.get(int(flags[0] & 0xFF), str(int(flags[0] & 0xFF)))
    if cls == "opaque":                              # name, type system, class name, then data matrix
        node = Node(cls, name=_txt(subs[1][1]) if len(subs) > 1 else "",
                    classname=_txt(subs[3][1]) if len(subs) > 3 else "")
        node.children = [_parse_matrix(p, depth + 1) for t, p in subs[4:] if t == 14]
        return node
    if len(subs) < 3:
        return Node(cls)
    dims = tuple(int(d) for d in np.frombuffer(subs[1][1], "<i4"))
    node = Node(cls, dims, _txt(subs[2][1]))
    rest = subs[3:]
    if cls == "cell":
        node.children = [_parse_matrix(p, depth + 1) for t, p in rest if t == 14]
    elif cls in ("struct", "object"):
        i = 0
        if cls == "object" and rest:
            node.classname = _txt(rest[0][1])
            i = 1
        if len(rest) >= i + 2:
            flen = int(np.frombuffer(rest[i][1][:4], "<i4")[0])
            raw = rest[i + 1][1]
            node.fields = [raw[k:k + flen].split(b"\0")[0].decode("latin1") for k in range(0, len(raw), flen)] if flen else []
            node.children = [_parse_matrix(p, depth + 1) for t, p in rest[i + 2:] if t == 14]
    elif cls == "char":
        if rest:
            t, p = rest[0]
            node.value = p.decode("utf-16-le", "replace") if t in (3, 4, 17) else p.decode("utf-8", "replace")
        else:
            node.value = ""
    elif cls in _NUMERIC_CLASSES and rest:
        t, p = rest[0]
        if t in _NUM:
            arr = np.frombuffer(p, _NUM[t])
            n = int(np.prod(dims)) if dims else arr.size
            if arr.size == n and dims:
                arr = arr.reshape(dims, order="F")
            node.value = arr
    return node


def _txt(b: bytes) -> str:
    return b.split(b"\0")[0].decode("latin1")


def read_mat5(path: str | Path) -> tuple[list[tuple[str, Node]], list[Node]]:
    """Return (variables, subsystem_nodes). Variables are (name, Node); opaque objects stay opaque."""
    data = Path(path).read_bytes()
    if len(data) < 128 or data[124:126] not in (b"\x00\x01", b"\x01\x00"):
        raise ValueError(f"{path}: not a MATLAB v5 MAT-file (v7.3/HDF5 files need h5py)")
    if data[126:128] != b"IM":
        raise ValueError(f"{path}: big-endian MAT-files are not supported")
    ssoff = struct.unpack_from("<Q", data, 116)[0]
    variables, subsystem = [], []
    off = 128
    while off + 8 <= len(data):
        t, sz = struct.unpack_from("<II", data, off)
        pay = off + 8
        body = data[pay:pay + sz]
        is_ss = ssoff not in (0, 0x2020202020202020) and off == ssoff
        nxt = pay + sz + (0 if t == 15 else (8 - (sz % 8)) % 8)
        elems = [(t, body)] if t != 15 else list(_elements(zlib.decompress(body)))
        for et, ep in elems:
            if et != 14:
                continue
            node = _parse_matrix(ep)
            if is_ss or (node.cls == "uint8" and node.name == "" and isinstance(node.value, np.ndarray) and off == ssoff):
                fw = node.value.tobytes() if isinstance(node.value, np.ndarray) else b""
                subsystem = [_parse_matrix(p) for tt, p in _elements(fw, 8) if tt == 14]
            else:
                variables.append((node.name, node))
        off = nxt
    return variables, subsystem


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for c in node.children:
        yield from _walk(c)


@dataclass
class GroundTruth:
    source: str                                   # DataSource path recorded by MATLAB (the video file)
    labels: list[str]                             # rectangle label names, in LabelData field order
    frames: list[dict[str, list[tuple[float, float, float, float]]]]   # per row: label -> [(x, y, w, h)]

    def boxes_xyxy(self, i: int, matlab_spatial: bool = True) -> list[tuple[str, float, float, float, float]]:
        """Boxes of row *i* as (label, x1, y1, x2, y2) in 0-based pixel-edge coordinates.

        MATLAB's spatial system puts pixel (1,1) between 0.5 and 1.5, so the 0-based edge
        coordinate is the MATLAB value - 0.5.
        """
        sh = 0.5 if matlab_spatial else 1.0
        out = []
        for lab, boxes in self.frames[i].items():
            for x, y, w, h in boxes:
                if w > 0 and h > 0:
                    out.append((lab, x - sh, y - sh, x - sh + w, y - sh + h))
        return out


def load_groundtruth(path: str | Path) -> GroundTruth:
    """Decode a Video Labeler ``groundTruth`` .mat file (all labels, all boxes per frame)."""
    _, subsystem = read_mat5(path)
    save = None
    for top in subsystem:
        for n in _walk(top):
            if n.cls == "struct" and "LabelData" in n.fields and "LabelDefinitions" in n.fields:
                save = n
                break
        if save is not None:
            break
    if save is None:
        raise ValueError(f"{path}: no groundTruth LabelData found")
    src = ""
    ds = save.field_values("DataSource")
    if ds:
        for n in _walk(ds[0]):
            if n.cls == "char" and n.text():
                src = n.text()
                break
    ld = save.field_values("LabelData")[0]
    defs = save.field_values("LabelDefinitions")[0]
    def_names = [n.text() for n in defs.field_values("Name")]
    labels = [f for f in ld.fields if f != "Time" and (not def_names or f in def_names)]
    n_rows = int(np.prod(ld.dims)) if ld.dims else 0
    frames: list[dict] = []
    for r in range(n_rows):
        row: dict[str, list] = {}
        for lab in labels:
            k = ld.fields.index(lab)
            node = ld.children[r * len(ld.fields) + k] if r * len(ld.fields) + k < len(ld.children) else None
            boxes = []
            if node is not None and isinstance(node.value, np.ndarray) and node.value.size >= 4:
                arr = np.asarray(node.value, dtype=float).reshape(-1, 4) if node.value.ndim == 1 else np.asarray(node.value, float)
                if arr.ndim == 2 and arr.shape[1] == 4:
                    boxes = [tuple(float(v) for v in b) for b in arr]
            if boxes:
                row[lab] = boxes
        frames.append(row)
    return GroundTruth(src, labels, frames)
