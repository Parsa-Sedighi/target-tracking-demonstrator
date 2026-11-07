"""Dataset pipeline tests: label decoders, readers for each dataset format, the builder, the
fetch helpers and the YOLO26 detector's class handling. Synthetic fixtures are generated in
tmp folders; the two Halmstad label files in tests/data are real (CC0)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from aimbot.training import build as B
from aimbot.training.fetch import extract, status
from aimbot.training.mat5 import load_groundtruth
from aimbot.training.schema import normalize_label
from aimbot.training.sources import (AntiUav, DroneVsBird, Generic, Halmstad, MavVid, RoboflowIndoor,
                                     guess_box_format, sequence_key, split_from_path)

DATA = Path(__file__).parent / "data"


# ----------------------------------------------------------------------------- helpers
def _img(path: Path, w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return img


def _video(path: Path, n=12, w=320, h=240):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.putText(f, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        vw.write(f)
    vw.release()
    assert path.exists()


def _read_labels(out: Path, split: str, source: str) -> dict:
    d = out / "labels" / split / source
    return {p.stem: p.read_text().split("\n") for p in d.glob("*.txt")} if d.exists() else {}


# ----------------------------------------------------------------------------- Halmstad / MAT5
def test_halmstad_groundtruth_drone_and_bird_in_one_frame():
    gt = load_groundtruth(DATA / "IR_DRONE_104_LABELS.mat")
    assert gt.labels == ["AIRPLANE", "BIRD", "DRONE", "HELICOPTER"]
    assert gt.source.endswith("IR_DRONE_104.mp4")
    assert len(gt.frames) == 311                                   # = frames in the video
    assert sum(len(r.get("DRONE", [])) for r in gt.frames) == 311
    assert sum(len(r.get("BIRD", [])) for r in gt.frames) == 49
    assert gt.frames[123]["BIRD"] == [(184.0, 12.0, 11.0, 12.0)]
    # MATLAB [x y w h] (pixel (1,1) spans 0.5..1.5) -> 0-based pixel edges
    assert gt.boxes_xyxy(123)[0] == ("BIRD", 183.5, 11.5, 194.5, 23.5)


def test_halmstad_groundtruth_several_boxes_per_frame():
    gt = load_groundtruth(DATA / "IR_BIRD_015_LABELS.mat")
    assert len(gt.frames) == 323
    assert sum(len(r.get("BIRD", [])) for r in gt.frames) == 540
    assert gt.frames[68]["BIRD"] == [(88.0, 88.0, 13.0, 11.0), (256.0, 39.0, 14.0, 9.0)]


def test_halmstad_reader(tmp_path):
    vdir = tmp_path / "Data" / "Video_IR"
    vdir.mkdir(parents=True)
    (vdir / "IR_DRONE_104_LABELS.mat").write_bytes((DATA / "IR_DRONE_104_LABELS.mat").read_bytes())
    _video(vdir / "IR_DRONE_104.mp4", n=3)
    groups = list(Halmstad().groups(tmp_path, {"modalities": ["ir"]}))
    assert len(groups) == 1 and groups[0].modality == "ir" and len(groups[0].items) == 311
    assert {b[0] for b in groups[0].items[123].boxes} == {"drone", "bird"}
    assert list(Halmstad().groups(tmp_path, {"modalities": ["rgb"]})) == []


# ----------------------------------------------------------------------------- schema / helpers
@pytest.mark.parametrize("raw,expect", [("UAV", "drone"), ("Drone", "drone"), ("DJI Mavic 2", "drone"),
                                        ("Aeroplane", "airplane"), ("BIRD", "bird"), ("HELICOPTER", "helicopter"),
                                        ("tennis-ball", "ball"), ("index_card", "card"), ("cat", None)])
def test_normalize_label(raw, expect):
    assert normalize_label(raw) == expect


def test_normalize_label_overrides():
    assert normalize_label("0", default="drone") == "drone"
    assert normalize_label("quad_x", {"quad_x": "drone"}) == "drone"
    assert normalize_label("bird", {"bird": None}) is None          # mapped to null = dropped


def test_split_and_sequence_helpers(tmp_path):
    assert split_from_path(Path("DUT/train/img/1.jpg")) == "train"
    assert split_from_path(Path("export/valid/images/a.jpg")) == "val"
    assert split_from_path(Path("Anti-UAV300/test/seq_1/infrared.json")) == "test"
    assert split_from_path(Path("clips/seq/a.jpg")) is None
    root = tmp_path
    assert sequence_key(root / "a" / "clipA_000123.jpg", root) == "a/clipA"
    assert sequence_key(root / "a" / "000123.jpg", root) == "a/#1"
    assert sequence_key(root / "frame_7_jpg.rf.0123456789abcdef.jpg", root) == "frame"
    assert guess_box_format([[10, 20, 30, 15]]) == "xywh"
    assert guess_box_format([[10, 20, 30, 45], [12, 21, 33, 47]]) == "xyxy"


# ----------------------------------------------------------------------------- readers
def test_mav_vid_reader(tmp_path):
    for split, vid in (("train", "uav1"), ("val", "uav9")):
        for k in range(3):
            _img(tmp_path / split / "img" / f"{vid}_{k:04d}.jpg")
            (tmp_path / split / "img" / f"{vid}_{k:04d}.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    groups = list(MavVid().groups(tmp_path, {}))
    assert {(g.name, g.split) for g in groups} == {("uav1", "train"), ("uav9", "val")}
    it = groups[0].items[0]
    assert it.norm and it.boxes == [("drone", 0.5, 0.5, 0.1, 0.2)]


def test_drone_vs_bird_reader(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "clip_01.txt").write_text(
        "0 1 100 50 20 10 drone\n1 0\n2 2 100 50 20 10 drone 10 10 5 5 drone\n")
    (tmp_path / "annotations" / "clip_02.txt").write_text("0 1 1 1 4 4 drone\n")
    _video(tmp_path / "videos" / "clip_01.avi", n=3)
    groups = list(DroneVsBird().groups(tmp_path, {}))
    assert [g.name for g in groups] == ["clip_01"]                  # clip_02 has no video yet
    g = groups[0]
    assert [len(it.boxes) for it in g.items] == [1, 0, 2]
    assert g.items[0].boxes[0] == ("drone", 100.0, 50.0, 120.0, 60.0)


def test_anti_uav_reader_both_box_formats(tmp_path):
    seq = tmp_path / "Anti-UAV300" / "train" / "20190925_101846_1_1"
    _video(seq / "infrared.mp4", n=3)
    (seq / "infrared.json").write_text(json.dumps({"exist": [1, 0, 1], "gt_rect": [[10, 20, 30, 15], [], [12, 22, 30, 15]]}))
    _video(seq / "visible.mp4", n=2)
    (seq / "visible.json").write_text(json.dumps({"exist": [1, 1], "gt_rect": [[100, 50, 140, 70], [102, 52, 141, 71]]}))
    groups = {g.name: g for g in AntiUav().groups(tmp_path, {})}
    ir, rgb = groups["20190925_101846_1_1_ir"], groups["20190925_101846_1_1_rgb"]
    assert ir.split == "train" and ir.modality == "ir" and ir.notes["box_format"] == "xywh"
    assert ir.items[0].boxes == [("drone", 10.0, 20.0, 40.0, 35.0)] and ir.items[1].boxes == []
    assert rgb.notes["box_format"] == "xyxy" and rgb.items[0].boxes == [("drone", 100.0, 50.0, 140.0, 70.0)]


def test_roboflow_export_reader(tmp_path):
    (tmp_path / "data.yaml").write_text("names: ['Drone']\nnc: 1\n")
    for split in ("train", "valid"):
        _img(tmp_path / split / "images" / f"frame_{split}_1_jpg.rf.0123456789abcdef.jpg")
        (tmp_path / split / "labels").mkdir(parents=True)
        (tmp_path / split / "labels" / f"frame_{split}_1_jpg.rf.0123456789abcdef.txt").write_text("0 0.4 0.4 0.2 0.2\n")
    groups = list(RoboflowIndoor().groups(tmp_path, {"default_class": "drone"}))
    assert sorted(g.split for g in groups) == ["train", "val"]
    assert all(g.items[0].boxes[0][0] == "drone" for g in groups)


def test_generic_reader_voc_coco_labelme_csv_yolo(tmp_path):
    # VOC
    _img(tmp_path / "voc" / "JPEGImages" / "a1.jpg")
    (tmp_path / "voc" / "Annotations").mkdir(parents=True)
    (tmp_path / "voc" / "Annotations" / "a1.xml").write_text(
        "<annotation><filename>a1.jpg</filename><object><name>UAV</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
        "<xmax>50</xmax><ymax>40</ymax></bndbox></object></annotation>")
    # COCO
    _img(tmp_path / "coco" / "images" / "c1.jpg")
    (tmp_path / "coco" / "ann.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "c1.jpg", "width": 320, "height": 240}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [5, 6, 10, 12], "iscrowd": 0}],
        "categories": [{"id": 3, "name": "drone"}]}))
    # LabelMe
    _img(tmp_path / "lm" / "l1.jpg")
    (tmp_path / "lm" / "l1.json").write_text(json.dumps({"imagePath": "l1.jpg", "shapes": [
        {"label": "drone", "points": [[30, 40], [10, 20]], "shape_type": "rectangle"}]}))
    # CSV
    _img(tmp_path / "csv" / "s1.jpg")
    (tmp_path / "csv" / "boxes.csv").write_text("filename,xmin,ymin,xmax,ymax,class\ns1.jpg,1,2,11,12,drone\n")
    # YOLO + classes.txt
    _img(tmp_path / "yolo" / "y1.jpg")
    (tmp_path / "yolo" / "y1.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    (tmp_path / "yolo" / "classes.txt").write_text("quadcopter\n")
    items = {it.name: it for g in Generic().groups(tmp_path, {}) for it in g.items}
    assert items["a1"].boxes == [("drone", 10.0, 20.0, 50.0, 40.0)] and not items["a1"].norm
    assert items["c1"].boxes == [("drone", 5, 6, 15, 18)]
    assert items["l1"].boxes == [("drone", 10, 20, 30, 40)]
    assert items["s1"].boxes == [("drone", 1.0, 2.0, 11.0, 12.0)]
    assert items["y1"].norm and items["y1"].boxes == [("drone", 0.5, 0.5, 0.25, 0.25)]


# ----------------------------------------------------------------------------- builder
def _two_source_cfg(tmp_path) -> dict:
    raw = tmp_path / "raw"
    # source A: 6 MAV-VID-style clips of 4 frames (no official split -> hash split)
    for c in range(6):
        for k in range(4):
            p = raw / "a" / "clips" / f"clip{c}_{k:04d}.jpg"
            _img(p, seed=c * 10 + k)
            p.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n" if k != 3 else "")
    # source B: one large frame (crop path) with an official split, VOC labels
    big = raw / "b" / "train" / "big_0001.jpg"
    _img(big, w=1920, h=1080, seed=99)
    big.with_suffix(".xml").write_text(
        "<annotation><object><name>drone</name><bndbox><xmin>900</xmin><ymin>500</ymin><xmax>930</xmax>"
        "<ymax>520</ymax></bndbox></object><object><name>bird</name><bndbox><xmin>100</xmin><ymin>100</ymin>"
        "<xmax>110</xmax><ymax>108</ymax></bndbox></object></annotation>")
    cfg = B.load_settings(None)
    cfg.update({"raw_root": str(raw), "output": str(tmp_path / "out"), "val_frac": 0.3, "neg_stride_mult": 1,
                "sources": {"a": {"path": "a", "reader": "yolo", "default_class": "drone"},
                            "b": {"path": "b", "reader": "generic"}}})
    return cfg


def test_build_end_to_end(tmp_path):
    cfg = _two_source_cfg(tmp_path)
    rep = B.build(cfg, tmp_path, workers=1, preview=True, progress=lambda *a: None)
    out = tmp_path / "out"
    data = (out / "aimbot_drones.yaml").read_text()
    assert "names:" in data and "0: drone" in data and "1: bird" in data
    # every clip of source A lands entirely in one split
    tr, va = _read_labels(out, "train", "a"), _read_labels(out, "val", "a")
    clips_tr = {k.split("__")[0] for k in tr}
    clips_va = {k.split("__")[0] for k in va}
    assert clips_tr and clips_va and not (clips_tr & clips_va)
    assert len(tr) + len(va) == 24                               # 18 positives + 6 background frames
    lab = next(v for k, v in tr.items() if k.endswith("_0000"))
    assert lab[0] == "0 0.500000 0.500000 0.200000 0.200000"
    # source B: full frame downscaled to max_side 1280 + one native-resolution crop
    b = _read_labels(out, "train", "b")
    assert set(b) == {"train_big__big_0001", "train_big__big_0001_c0"} or len(b) == 2
    full = next(v for k, v in b.items() if not k.endswith("_c0"))
    k, cx, cy, w, h = full[0].split()
    assert k == "0" and abs(float(cx) - 915 / 1920) < 1e-4 and abs(float(w) - 30 / 1920) < 1e-4
    img = cv2.imread(str(next((out / "images" / "train" / "b").glob("*_c0.jpg"))))
    assert img.shape[:2] == (640, 640)
    crop = next(v for k, v in b.items() if k.endswith("_c0"))
    sizes = {(ln.split()[0], round(float(ln.split()[3]) * 640), round(float(ln.split()[4]) * 640)) for ln in crop if ln}
    assert sizes and sizes <= {("0", 30, 20), ("1", 10, 8)}                         # native pixels, not shrunk
    srcs = {s["source"]: s for s in rep["sources"]}
    assert srcs["b"]["splits"]["train"]["boxes"] == {"bird": 1, "drone": 1}
    assert (out / "previews" / "a.jpg").exists() and (out / "build_report.md").exists()


def test_build_incremental_and_class_change(tmp_path):
    cfg = _two_source_cfg(tmp_path)
    B.build(cfg, tmp_path, workers=1, preview=False, progress=lambda *a: None)
    out = tmp_path / "out"
    n_a = len(list((out / "labels").rglob("a/*.txt")))
    B.build(cfg, tmp_path, only=["b"], workers=1, preview=False, progress=lambda *a: None)
    assert len(list((out / "labels").rglob("a/*.txt"))) == n_a           # other sources untouched
    cfg2 = dict(cfg, classes=["drone"])
    with pytest.raises(SystemExit):
        B.build(cfg2, tmp_path, workers=1, preview=False, progress=lambda *a: None)
    rep = B.build(cfg2, tmp_path, clean=True, workers=1, preview=False, progress=lambda *a: None)
    b = {s["source"]: s for s in rep["sources"]}["b"]
    assert b["dropped_not_in_classes"] == {"bird": 1}


def test_select_items_strides():
    from aimbot.training.sources import Item

    items = [Item([("drone", 0, 0, 1, 1)]) for _ in range(20)] + [Item([]) for _ in range(20)]
    sel = B.select_items(items, stride=5, neg_stride=10, max_per_group=0, salt="x")
    assert sum(1 for i in sel if i < 20) == 4 and sum(1 for i in sel if i >= 20) == 2
    assert len(B.select_items(items, 1, 1, 7, "x")) == 7


# ----------------------------------------------------------------------------- fetch helpers
def test_extract_rejects_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ok/a.txt", "fine")
        zf.writestr("../../escaped.txt", "bad")
    out = extract(z, log=lambda *a: None)
    assert (out / "ok" / "a.txt").exists()
    assert not (tmp_path.parent / "escaped.txt").exists() and not (tmp_path / "escaped.txt").exists()


def _split(src: Path, n: int) -> list:
    """7-Zip style volumes: the archive bytes cut into name.001, name.002, ..."""
    data = src.read_bytes()
    step = len(data) // n + 1
    vols = []
    for i in range(n):
        v = src.with_name(f"{src.name}.{i + 1:03d}")
        v.write_bytes(data[i * step:(i + 1) * step])
        vols.append(v)
    src.unlink()
    return vols


def test_split_zip_volumes_extract_as_one_archive(tmp_path):
    import random

    from aimbot.training.fetch import JoinedVolumes, extract_all, is_archive

    blob = bytes(random.Random(0).getrandbits(8) for _ in range(300_000))   # incompressible: 3 real volumes
    z = tmp_path / "train_data.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("imgs/a.jpg", blob)
        zf.writestr("imgs/a.txt", "0 0.5 0.5 0.1 0.1\n")
    whole = z.read_bytes()
    vols = _split(z, 3)
    assert [is_archive(v) for v in vols] == [True, False, False]
    with JoinedVolumes(vols) as j:                     # reads across volume boundaries, seeks anywhere
        assert j.read() == whole
        j.seek(-10, 2)
        assert j.read(10) == whole[-10:]
        j.seek(len(vols[0].read_bytes()) - 3)
        assert j.read(6) == whole[len(vols[0].read_bytes()) - 3:][:6]
    assert extract_all(tmp_path, log=lambda *a: None) == 1
    assert (tmp_path / "train_data" / "imgs" / "a.jpg").read_bytes() == blob
    assert (tmp_path / ".train_data.zip.001.extracted").exists()
    assert extract_all(tmp_path, log=lambda *a: None) == 0          # done once


def test_split_volumes_wait_for_a_complete_set(tmp_path):
    from aimbot.training.fetch import extract, volume_set

    z = tmp_path / "set.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("a.bin", bytes(range(256)) * 400)
    v1, v2, v3 = _split(z, 3)
    msgs = []
    v2.rename(v2.with_name(v2.name + ".part"))                         # still downloading
    assert extract(v1, log=msgs.append) is None and "did not finish" in msgs[-1]
    v2.with_name(v2.name + ".part").unlink()                           # a volume missing in the middle
    assert volume_set(v1) == ([], "missing set.zip.002")
    assert extract(v1, log=msgs.append) is None and not (tmp_path / ".set.zip.001.extracted").exists()


def test_split_7z_volumes(tmp_path):
    py7zr = pytest.importorskip("py7zr")
    from aimbot.training.fetch import extract_all, status

    src = tmp_path / "files"
    (src / "imgs").mkdir(parents=True)
    for i in range(3):
        (src / "imgs" / f"{i}.jpg").write_bytes(bytes([i]) * 50_000)
    dest = tmp_path / "raw" / "drone_hunter"
    dest.mkdir(parents=True)
    a = dest / "train_data_0402.7z"
    with py7zr.SevenZipFile(a, "w", filters=[{"id": py7zr.FILTER_COPY}]) as z:
        z.writeall(src / "imgs", "imgs")
    _split(a, 3)
    (dest / "Code-master.zip").write_bytes(b"not data")               # skipped by the drone_hunter spec
    assert {r["key"]: r for r in status(tmp_path / "raw")}["drone_hunter"]["state"] == "archives not extracted"
    assert extract_all(dest, log=lambda *a: None, skip=["*-master.zip"]) == 1
    assert sorted(p.name for p in (dest / "train_data_0402" / "imgs").iterdir()) == ["0.jpg", "1.jpg", "2.jpg"]
    assert {r["key"]: r for r in status(tmp_path / "raw")}["drone_hunter"]["state"] == "ready"


def test_status_table(tmp_path):
    rows = {r["key"]: r for r in status(tmp_path)}
    assert rows["halmstad"]["state"] == "missing"
    (tmp_path / "drone_vs_bird").mkdir()
    (tmp_path / "drone_vs_bird" / "clip.txt").write_text("0 0\n")
    assert status(tmp_path)[-2]["state"].startswith("annotations only")


# ----------------------------------------------------------------------------- YOLO26 detector classes
def _weights(name):
    from aimbot.config import PACKAGE_ROOT

    p = PACKAGE_ROOT / "models" / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return p


def test_coco_fallback_maps_proxies_to_drone():
    pytest.importorskip("ultralytics")
    _weights("yolo26n.pt")
    from aimbot.perception.detectors import YoloDetector

    d = YoloDetector({"model": "models/does_not_exist.pt", "fallback_model": ["yolo26n.pt"], "device": "cpu",
                      "assist_model": ["yolo26n.pt"], "assist_classes": ["sports ball"]})
    assert d.fallback_used and d.logical("airplane") == "drone" and d.logical("sports ball") == "ball"
    assert d.assist is None                       # the assist would be the same model


def test_trained_model_keeps_bird_and_adds_coco_ball():
    pytest.importorskip("ultralytics")
    _weights("yolo26n.pt")
    _weights("aimbot_yolo26n.pt")
    from aimbot.perception.detectors import YoloDetector
    from aimbot.types import Frame

    d = YoloDetector({"model": "models/aimbot_yolo26n.pt", "fallback_model": ["yolo26n.pt"], "device": "cpu",
                      "assist_model": ["yolo26n.pt"], "assist_classes": ["sports ball", "airplane"]})
    assert not d.fallback_used
    assert {"drone", "bird", "airplane", "helicopter"} <= d.main.logical_set
    assert d.logical("bird") == "bird" and d.logical("airplane") == "airplane"     # no COCO proxy
    assert d.assist is not None and d.assist.logical_set == {"ball"}               # airplane already in main
    frame = Frame(frame_id=1, t_capture=0.0, image=np.zeros((480, 640, 3), np.uint8), sensor="vis")
    calls = []
    for m in d.models:
        orig = m.predict
        m.predict = (lambda mm, o: (lambda *a, **k: (calls.append(mm.path.name), o(*a, **k))[1]))(m, orig)
    d.focus = "drone"
    d.detect(frame, roi=(100, 100, 420, 420))
    assert calls == ["aimbot_yolo26n.pt"]                    # ROI while tracking a drone: drone model only
    calls.clear()
    d.detect(frame, roi=None)
    assert calls == ["aimbot_yolo26n.pt", "yolo26n.pt"]      # full frame: both


def test_generic_reader_pixel_txt_option(tmp_path):
    _img(tmp_path / "img" / "p1.jpg")
    (tmp_path / "img" / "p1.txt").write_text("UAV 10 20 30 15\n")
    _img(tmp_path / "img" / "p2.jpg", seed=1)
    (tmp_path / "img" / "p2.txt").write_text("0,5,6,25,16\n")
    assert list(Generic().groups(tmp_path, {})) == []                    # not YOLO, nothing guessed
    items = {it.name: it for g in Generic().groups(tmp_path, {"txt_format": "xywh", "default_class": "drone"})
             for it in g.items}
    assert items["p1"].boxes == [("drone", 10.0, 20.0, 40.0, 35.0)] and not items["p1"].norm
    assert items["p2"].boxes == [("drone", 5.0, 6.0, 30.0, 22.0)]


def test_generic_reader_pairs_labels_within_the_same_split(tmp_path):
    for split, x in (("train", 10), ("val", 100)):
        _img(tmp_path / split / "img" / "00001.jpg", seed=x)
        (tmp_path / split / "xml").mkdir(parents=True)
        (tmp_path / split / "xml" / "00001.xml").write_text(
            f"<annotation><object><name>UAV</name><bndbox><xmin>{x}</xmin><ymin>5</ymin><xmax>{x + 20}</xmax>"
            "<ymax>25</ymax></bndbox></object></annotation>")
    got = {(g.split, it.image.parent.parent.name, it.boxes[0][1]) for g in Generic().groups(tmp_path, {}) for it in g.items}
    assert got == {("train", "train", 10.0), ("val", "val", 100.0)}


def test_extract_tar(tmp_path):
    import io
    import tarfile

    t = tmp_path / "data.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        for name, text in (("seq/a.txt", b"a"), ("../bad.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(text)
            tf.addfile(info, io.BytesIO(text))
    out = extract(t, log=lambda *a: None)
    assert (out / "seq" / "a.txt").read_bytes() == b"a" and not (tmp_path / "bad.txt").exists()


def test_broken_model_file_falls_back(tmp_path):
    pytest.importorskip("ultralytics")
    _weights("yolo26n.pt")
    from aimbot.perception.detectors import YoloDetector

    bad = tmp_path / "aimbot_yolo26n.pt"
    bad.write_bytes(b"not a checkpoint")
    d = YoloDetector({"model": str(bad), "fallback_model": ["yolo26n.pt"], "device": "cpu"})
    assert "failed to load" in d.fallback_used and d.model_path.endswith("yolo26n.pt")


# ----------------------------------------------------------------------------- quality filter / audit / augmentation
def test_quality_checks_unit():
    from aimbot.training import quality as Q
    from aimbot.training.sources import Item

    q = Q.settings({})
    # degenerate / tiny boxes
    assert Q.label_reason([("drone", 0, 0, 90, 10)], 100, 100, q) == "degenerate_box"
    assert Q.label_reason([("drone", 10, 10, 13, 12)], 100, 100, q) == "tiny_target"
    assert Q.label_reason([("drone", 10, 10, 30, 25)], 100, 100, q) is None
    # one-frame glitch in otherwise smooth labels; a smooth fast mover is not flagged
    items = [Item([("drone", 10 + 5 * f, 10, 30 + 5 * f, 30)], frame=f) for f in range(8)]
    items[4] = Item([("drone", 200, 150, 220, 170)], frame=4)
    assert Q.temporal_outliers(items, q) == {4}
    fast = [Item([("drone", 40 * f, 10, 40 * f + 20, 30)], frame=f) for f in range(8)]
    assert Q.temporal_outliers(fast, q) == set()
    # exposure
    assert Q.exposure_reason(np.zeros((40, 40), np.uint8), q) == "dark"
    assert Q.exposure_reason(np.full((40, 40), 255, np.uint8), q) == "washed_out"
    # invisible target on plain sky vs a real dark drone on sky vs clutter (not judged)
    sky = np.full((120, 160), 180, np.uint8) + np.random.default_rng(0).integers(0, 3, (120, 160)).astype(np.uint8)
    assert Q.invisible_target(sky, [("drone", 10, 10, 30, 30), ("drone", 60, 50, 80, 64)], q) == 0
    with_drone = sky.copy()
    with_drone[10:30, 10:30] = 40
    with_drone[56:58, 69:71] = 60                      # a 2x2 px speck in a loose 20x14 box still counts
    assert Q.invisible_target(with_drone, [("drone", 10, 10, 30, 30), ("drone", 60, 50, 80, 64)], q) is None
    clutter = np.random.default_rng(1).integers(0, 255, (120, 160)).astype(np.uint8)
    assert Q.invisible_target(clutter, [("drone", 60, 50, 80, 64)], q) is None
    # duplicate detection
    a = np.random.default_rng(2).integers(0, 255, (64, 64)).astype(np.uint8)
    assert Q.hamming(Q.dhash(a), Q.dhash(a)) == 0
    assert Q.same_layout([("drone", 0, 0, 10, 10)], [("drone", 0, 0, 10, 10.5)])


def test_build_quality_filter_and_exclude_list(tmp_path):
    raw = tmp_path / "raw" / "q" / "clips"
    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for k in range(10):
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        if k == 3:
            img = np.zeros_like(img)                         # black frame
        if k == 5:
            img[100:140, 135:185] = 0                         # a real (dark) target in the labelled box ...
            img = cv2.GaussianBlur(img, (9, 9), 3)           # ... in a much blurrier frame
        if k in (7, 8):
            img = base.copy()                                # two identical frames
        p = raw / f"clip0_{k:04d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        p.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "exclude.txt").write_text("q/*__clip0_0001   # reviewed: bad label\n")
    cfg = B.load_settings(None)
    cfg.update({"raw_root": str(tmp_path / "raw"), "output": str(tmp_path / "out"), "val_frac": 0.0,
                "quality": {**B.DEFAULTS["quality"], "exclude_list": str(tmp_path / "exclude.txt"), "min_clip_frames": 4,
                            "sharpness_rel": 0.35},
                "sources": {"q": {"path": "q", "reader": "yolo", "default_class": "drone"}}})
    rep = B.build(cfg, tmp_path, workers=1, preview=True, progress=lambda *a: None)
    rj = rep["sources"][0]["rejected"]
    assert rj.get("dark") == 1 and rj.get("excluded") == 1 and rj.get("duplicate") == 1 and rj.get("blurry") == 1
    kept = {p.stem.split("__")[-1] for p in (tmp_path / "out" / "images" / "train" / "q").glob("*.jpg")}
    assert kept == {"clip0_0000", "clip0_0002", "clip0_0004", "clip0_0006", "clip0_0007", "clip0_0009"}
    assert (tmp_path / "out" / "previews" / "q_rejected.jpg").exists()
    assert "Frames removed by the quality filter" in (tmp_path / "out" / "build_report.md").read_text()


def test_audit_flags():
    import importlib.util

    spec = importlib.util.spec_from_file_location("audit_labels", Path(__file__).parent.parent / "tools" / "audit_labels.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    gt = [(0, 100, 100, 130, 120), (1, 300, 50, 320, 66), (0, 600, 300, 630, 320)]
    pred = [(0, 0.9, 101, 100, 131, 121),          # matches the first drone label
            (0, 0.8, 80, 85, 150, 135),             # loose duplicate around the same drone: not "missing"
            (0, 0.8, 500, 400, 530, 420),           # confident, nothing labelled there
            (2, 0.7, 300, 50, 320, 66)]             # airplane on top of a bird label
    flags = sorted(f[0] for f in A.audit_image(gt, pred))
    # the last drone label has no detection at all; the bird got an airplane detection (review only)
    assert flags == ["missed_target", "missing_label", "wrong_class"]
    # loose_box only when the LABEL is the loose one (a small confident detection inside it) ...
    assert [f[0] for f in A.audit_image([(0, 10, 10, 60, 50)], [(0, 0.8, 25, 22, 45, 38)])] == ["loose_box"]
    # ... not when the model's box is the bigger one around a tight label
    assert A.audit_image([(0, 25, 22, 45, 38)], [(0, 0.8, 10, 10, 60, 50)]) == []


def test_camera_aug_changes_image_only():
    from aimbot.training.augment import CameraAug

    img = np.zeros((64, 64, 3), np.uint8)
    img[30:34, :] = 255
    out = CameraAug(p_blur=1.0, p_noise=1.0)({"img": img.copy(), "cls": np.array([0])})
    assert out["img"].shape == img.shape and out["img"].dtype == np.uint8 and not np.array_equal(out["img"], img)
    assert out["cls"].tolist() == [0]


def test_camera_aug_installs_into_ultralytics_training_only(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    from aimbot.training.augment import install_camera_aug

    assert install_camera_aug(1.0, 1.0)
    d = tmp_path / "images" / "train"
    (tmp_path / "labels" / "train").mkdir(parents=True)
    for k in range(4):
        _img(d / f"a{k}.jpg", 128, 96, seed=k)
        (tmp_path / "labels" / "train" / f"a{k}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    kw = dict(img_path=str(d), imgsz=96, hyp=get_cfg(), data={"names": {0: "drone"}, "channels": 3})
    train = YOLODataset(augment=True, batch_size=2, **kw)
    names = [type(t).__name__ for t in train.transforms.transforms]
    assert names[-2:] == ["CameraAug", "Format"]
    assert tuple(train[0]["img"].shape) == (3, 96, 96)
    assert "CameraAug" not in [type(t).__name__ for t in YOLODataset(augment=False, **kw).transforms.transforms]



def _sky_with_object(x=100, y=90, w=20, h=12):
    rng = np.random.default_rng(0)
    sky = (np.full((200, 260), 170, np.float32) + rng.normal(0, 1.5, (200, 260))).clip(0, 255).astype(np.uint8)
    sky[y:y + h, x:x + w] = 60
    return cv2.GaussianBlur(sky, (3, 3), 0.8)


def test_box_fit_measures_objects():
    from aimbot.training import quality as Q

    q = Q.settings({})
    img = _sky_with_object()                                     # 20 x 12 object at (100, 90)
    assert Q.box_fit(img, [("drone", 99, 89, 121, 103)], q) is None        # tight
    assert Q.box_fit(img, [("drone", 96, 86, 124, 106)], q) is None        # normal 4 px margin
    r = Q.box_fit(img, [("drone", 85, 75, 135, 117)], q)                    # 50 x 42 around 20 x 12
    assert r is not None and r[0] == "loose_box"
    c, x1, y1, x2, y2 = Q.tighten_box(("drone", 85, 75, 135, 117), r[2], q)
    assert 95 <= x1 <= 100 and 120 <= x2 <= 125 and 85 <= y1 <= 90 and 102 <= y2 <= 107
    # a box half off the object is not tightened onto it, and only flagged with misplaced_check
    assert Q.box_fit(img, [("drone", 112, 95, 140, 115)], q) is None
    assert Q.box_fit(img, [("drone", 112, 95, 140, 115)], dict(q, misplaced_check=True))[0] == "misplaced_box"
    # tiny target with a normal few-pixel margin is fine; clutter is never judged
    tiny = _sky_with_object(120, 100, 5, 4)
    assert Q.box_fit(tiny, [("drone", 116, 96, 129, 108)], q) is None
    clutter = np.random.default_rng(1).integers(0, 255, (200, 260)).astype(np.uint8)
    assert Q.box_fit(clutter, [("drone", 85, 75, 135, 117)], q) is None
    # sky gradient is removed before measuring
    grad = np.tile(np.linspace(120, 200, 260, dtype=np.float32), (200, 1)).astype(np.uint8)
    grad[90:102, 100:120] = 60
    grad = cv2.GaussianBlur(grad, (3, 3), 0.8)
    assert Q.box_fit(grad, [("drone", 98, 88, 122, 104)], q) is None
    assert Q.box_fit(grad, [("drone", 85, 75, 135, 117)], q)[0] == "loose_box"


def _loose_label_source(tmp_path):
    raw = tmp_path / "raw" / "s" / "clips"
    for k in range(3):
        img = _sky_with_object(100 + 10 * k, 90)
        p = raw / f"clip0_{k:04d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])
        # loose label: 50 x 42 around the 20 x 12 object
        x1, y1, x2, y2 = 85 + 10 * k, 75, 135 + 10 * k, 117
        p.with_suffix(".txt").write_text(f"0 {(x1 + x2) / 520:.6f} {(y1 + y2) / 400:.6f} {(x2 - x1) / 260:.6f} {(y2 - y1) / 200:.6f}\n")
    cfg = B.load_settings(None)
    cfg.update({"raw_root": str(tmp_path / "raw"), "output": str(tmp_path / "out"), "val_frac": 0.0,
                "quality": {**B.DEFAULTS["quality"], "dedup_hamming": 0},
                "sources": {"s": {"path": "s", "reader": "yolo", "default_class": "drone"}}})
    return cfg


def test_build_tightens_loose_labels(tmp_path):
    cfg = _loose_label_source(tmp_path)
    rep = B.build(cfg, tmp_path, workers=1, preview=True, progress=lambda *a: None)
    src = rep["sources"][0]
    assert src["boxes_tightened"] == 3 and not src.get("rejected")
    for lab in (tmp_path / "out" / "labels" / "train" / "s").glob("*.txt"):
        k, cx, cy, w, h = lab.read_text().split()
        assert 20 <= float(w) * 260 <= 27 and 12 <= float(h) * 200 <= 19       # 20 x 12 object, was 50 x 42
    assert (tmp_path / "out" / "previews" / "s_tightened.jpg").exists()
    assert "Loose labels tightened" in (tmp_path / "out" / "build_report.md").read_text()


def test_build_drop_mode_excludes_loose_labels(tmp_path):
    cfg = _loose_label_source(tmp_path)
    cfg["quality"]["box_fit"] = "drop"
    rep = B.build(cfg, tmp_path, workers=1, preview=False, progress=lambda *a: None)
    assert rep["sources"][0]["rejected"] == {"loose_box": 3}
    assert not list((tmp_path / "out" / "images" / "train" / "s").glob("*.jpg"))


def test_official_split_false_ignores_split_like_folder_names(tmp_path):
    import cv2

    from aimbot.training.sources import get_source

    d = tmp_path / "train_data_0402" / "train_data"
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir()
    for clip in ("GOPR3128", "GOPR3134"):
        cv2.imwrite(str(d / "images" / f"{clip}_1.jpg"), np.zeros((40, 40, 3), np.uint8))
        (d / "labels" / f"{clip}_1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    src = get_source("drone_hunter", {})
    assert {g.split for g in src.groups(tmp_path, {"default_class": "drone"})} == {"train"}
    groups = list(src.groups(tmp_path, {"default_class": "drone", "official_split": False}))
    assert {g.split for g in groups} == {None} and len(groups) == 2


def test_empty_box_rejects_the_frame_instead_of_making_it_background(tmp_path):
    raw = tmp_path / "raw" / "q" / "clips"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for k, label in enumerate(["0 0.5 0.5 0.2 0.2\n", "0 0.48 0.02 0.0 0.0\n", ""]):
        cv2.imwrite(str(raw / f"clip0_{k:04d}.jpg"), rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
        (raw / f"clip0_{k:04d}.txt").write_text(label)
    cfg = B.load_settings(None)
    cfg.update({"raw_root": str(tmp_path / "raw"), "output": str(tmp_path / "out"), "val_frac": 0.0,
                "neg_stride_mult": 1, "sources": {"q": {"path": "q", "reader": "yolo", "default_class": "drone"}}})
    rep = B.build(cfg, tmp_path, workers=1, preview=False, progress=lambda *a: None)
    assert rep["sources"][0]["rejected"].get("degenerate_box") == 1
    kept = {p.stem.split("__")[-1] for p in (tmp_path / "out" / "images" / "train" / "q").glob("*.jpg")}
    assert kept == {"clip0_0000", "clip0_0002"}             # the empty label file stays a real background frame


def test_augmented_copies_stay_in_their_original_clip(tmp_path):
    from aimbot.training.sources import sequence_key

    d = tmp_path / "images"
    keys = {sequence_key(d / f"{n}.jpg", tmp_path) for n in
            ("GOPR3121_10_1002", "GOPR3121_10_1002_blur", "GOPR3121_10_1005_blur", "GOPR3121_10_1011")}
    assert keys == {"images/GOPR3121_10"}
    assert sequence_key(d / "3_1_7.jpg", tmp_path) == "images/3_1"


def test_dark_or_glaring_frames_are_kept_when_the_target_is_visible():
    from aimbot.training import quality as Q

    q = Q.settings({})
    night = np.full((480, 640, 3), 3, np.uint8)
    lit = night.copy()
    cv2.circle(lit, (320, 240), 4, (60, 220, 60), -1)                  # the drone's LEDs
    box = [("drone", 300, 225, 340, 255)]
    assert Q.exposure_reason(Q.small_gray(night), q, night, box) == "dark"          # nothing to see in the box
    assert Q.exposure_reason(Q.small_gray(lit), q, lit, box) is None                # lights in the box: kept
    assert Q.exposure_reason(Q.small_gray(lit), q) == "dark"                        # no label: still a dark frame
    sky = np.full((480, 640, 3), 255, np.uint8)
    drone = sky.copy()
    cv2.rectangle(drone, (305, 230), (335, 250), (40, 40, 40), -1)     # dark silhouette, blown-out sky
    assert Q.exposure_reason(Q.small_gray(sky), q, sky, box) == "washed_out"
    assert Q.exposure_reason(Q.small_gray(drone), q, drone, box) is None


def test_label_jumps_are_only_judged_in_video_like_groups():
    from aimbot.training import quality as Q
    from aimbot.training.sources import Item

    q = Q.settings({})
    video = [Item([("drone", 100 + 2 * f, 100, 120 + 2 * f, 115)], frame=f) for f in range(20)]
    video[10] = Item([("drone", 400, 300, 420, 315)], frame=10)          # a one-frame glitch
    assert Q.temporal_outliers(video, q) == {10}
    rng = np.random.default_rng(0)
    photos = []
    for f in range(20):                                                   # independent photos, numbered 0..19
        x, y = rng.integers(0, 600, 2)
        photos.append(Item([("drone", float(x), float(y), float(x) + 20, float(y) + 15)], frame=f))
    assert Q.temporal_outliers(photos, q) == set()
