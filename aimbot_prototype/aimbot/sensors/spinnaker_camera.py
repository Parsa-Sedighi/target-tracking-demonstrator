"""FLIR / Teledyne Blackfly S (BFS-U3-28S5C-C) backend using the Spinnaker SDK (PySpin).

Sensor: Sony IMX421, 1936 x 1464, 4.5 um pixels, 130 fps max in BayerRG8.
With the Kowa LM8JC 8 mm lens: fx ~= 1778 px, HFOV ~= 57 deg, VFOV ~= 45 deg.

Install on the Jetson (Ubuntu 22.04 / JetPack 6, Python 3.10): Spinnaker SDK
ARM64 .deb packages + the matching ``spinnaker_python-*-cp310-*-linux_aarch64.whl``.
On Windows: Spinnaker SDK + the cp310 or cp312 wheel. See docs/JETSON_SETUP.md.

Not tested against a physical camera yet - every PySpin call is wrapped so a
missing node gives a warning, not a crash. Run ``python tools/camera_test.py
--config config/jetson_hw.yaml`` the day the camera arrives.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from .. import clock
from ..perception.geometry import Intrinsics
from .camera import CameraBase, FrameBuffer

log = logging.getLogger(__name__)

# Spinnaker Bayer naming -> OpenCV conversion code (OpenCV names the 2nd row pattern).
_BAYER_TO_CV = {
    "BayerRG8": cv2.COLOR_BayerBG2BGR,
    "BayerGB8": cv2.COLOR_BayerGR2BGR,
    "BayerGR8": cv2.COLOR_BayerGB2BGR,
    "BayerBG8": cv2.COLOR_BayerRG2BGR,
}


class SpinnakerCamera(CameraBase):
    kind = "spinnaker"

    def __init__(self, name: str, cfg: dict, framebuf: FrameBuffer):
        super().__init__(name, cfg, framebuf)
        self.PySpin = None
        self.system = None
        self.cam_list = None
        self.cam = None
        self.pixel_format = str(cfg.get("pixel_format", "BayerRG8"))
        self._cv_code: Optional[int] = None
        self._ts_offset: Optional[float] = None   # host_time - camera_time (s)
        self._last_sync = 0.0
        self.exposure_us = float(cfg.get("exposure_us", 2000))
        self.serial = str(cfg.get("serial", "") or "")
        self.dropped = 0

    # ------------------------------------------------------------------ helpers
    def _try(self, what: str, fn):
        try:
            return fn()
        except Exception as exc:
            log.warning("Spinnaker: could not %s: %s", what, exc)
            return None

    def _sync_clock(self) -> None:
        """Map camera timestamps (ns, camera clock) onto clock.now()."""
        cam = self.cam
        best = None
        for _ in range(5):
            t0 = clock.now()
            try:
                cam.TimestampLatch.Execute()
                ts = cam.TimestampLatchValue.GetValue() * 1e-9
            except Exception:
                return
            t1 = clock.now()
            rt = t1 - t0
            if best is None or rt < best[0]:
                best = (rt, 0.5 * (t0 + t1) - ts)
        if best is not None:
            self._ts_offset = best[1]
            self._last_sync = clock.now()

    # ------------------------------------------------------------------ CameraBase
    def open(self) -> None:
        try:
            import PySpin  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PySpin (Spinnaker Python) is not installed - see docs/JETSON_SETUP.md") from exc
        self.PySpin = PySpin
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        n = self.cam_list.GetSize()
        if n == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError("no Spinnaker cameras found (check USB3 cable / udev rules / usbfs_memory_mb)")
        cam = None
        if self.serial:
            cam = self.cam_list.GetBySerial(self.serial)
        if cam is None:
            cam = self.cam_list.GetByIndex(0)
        self.cam = cam
        cam.Init()
        c = self.cfg
        P = PySpin
        self._try("stop previous acquisition", lambda: cam.EndAcquisition())
        self._try("set continuous mode", lambda: cam.AcquisitionMode.SetValue(P.AcquisitionMode_Continuous))
        fmt_enum = getattr(P, f"PixelFormat_{self.pixel_format}", None)
        if fmt_enum is not None:
            self._try(f"set pixel format {self.pixel_format}", lambda: cam.PixelFormat.SetValue(fmt_enum))
        if int(c.get("binning", 1)) > 1:
            b = int(c["binning"])
            ok = self._try("set binning", lambda: (cam.BinningHorizontal.SetValue(b), cam.BinningVertical.SetValue(b)))
            if ok is None:
                self._try("set decimation", lambda: (cam.DecimationHorizontal.SetValue(b), cam.DecimationVertical.SetValue(b)))
        # ROI: offsets to zero first so width/height can grow
        self._try("reset offsets", lambda: (cam.OffsetX.SetValue(0), cam.OffsetY.SetValue(0)))
        if c.get("width"):
            self._try("set width", lambda: cam.Width.SetValue(int(c["width"])))
        if c.get("height"):
            self._try("set height", lambda: cam.Height.SetValue(int(c["height"])))
        if c.get("offset_x"):
            self._try("set offset x", lambda: cam.OffsetX.SetValue(int(c["offset_x"])))
        if c.get("offset_y"):
            self._try("set offset y", lambda: cam.OffsetY.SetValue(int(c["offset_y"])))
        if str(c.get("exposure", "manual")).lower() == "auto":
            self._try("enable auto exposure", lambda: cam.ExposureAuto.SetValue(P.ExposureAuto_Continuous))
        else:
            self._try("disable auto exposure", lambda: cam.ExposureAuto.SetValue(P.ExposureAuto_Off))
            self._try("set exposure", lambda: cam.ExposureTime.SetValue(self.exposure_us))
        if c.get("gain_db") is not None:
            self._try("disable auto gain", lambda: cam.GainAuto.SetValue(P.GainAuto_Off))
            self._try("set gain", lambda: cam.Gain.SetValue(float(c["gain_db"])))
        if c.get("fps"):
            self._try("enable frame-rate control", lambda: cam.AcquisitionFrameRateEnable.SetValue(True))
            self._try("set frame rate", lambda: cam.AcquisitionFrameRate.SetValue(float(c["fps"])))
        if c.get("throughput_limit_mbps"):
            self._try("set link throughput", lambda: cam.DeviceLinkThroughputLimit.SetValue(int(float(c["throughput_limit_mbps"]) * 1e6)))
        # newest-only buffering = lowest latency (old frames are discarded by the driver)
        self._try("set NewestOnly buffering",
                  lambda: cam.TLStream.StreamBufferHandlingMode.SetValue(P.StreamBufferHandlingMode_NewestOnly))
        try:
            self.pixel_format = cam.PixelFormat.GetCurrentEntry().GetSymbolic()
        except Exception:
            pass
        self._cv_code = _BAYER_TO_CV.get(self.pixel_format)
        w = int(self._try("read width", lambda: cam.Width.GetValue()) or c.get("width", 1936))
        h = int(self._try("read height", lambda: cam.Height.GetValue()) or c.get("height", 1464))
        intr_cfg = dict(c.get("intrinsics", {}) or {})
        if not intr_cfg.get("fx") and intr_cfg.get("pixel_um"):
            intr_cfg["pixel_um"] = float(intr_cfg["pixel_um"]) * int(c.get("binning", 1))
        self.intr = Intrinsics.from_config(intr_cfg, w, h)
        self._sync_clock()
        cam.BeginAcquisition()
        log.info("Spinnaker camera %s: %dx%d %s, HFOV %.1f deg, clock sync %s", self.name, w, h, self.pixel_format,
                 self.intr.hfov_deg, "ok" if self._ts_offset is not None else "unavailable")

    def grab(self):
        P = self.PySpin
        cam = self.cam
        try:
            image = cam.GetNextImage(1000)
        except P.SpinnakerException as exc:
            self.last_error = str(exc)
            return None
        t_host = clock.now()
        try:
            if image.IsIncomplete():
                self.dropped += 1
                return None
            hw_ts = None
            try:
                hw_ts = int(image.GetTimeStamp())
            except Exception:
                pass
            arr = image.GetNDArray()
            raw = np.array(arr, copy=True)  # the driver buffer is recycled after Release()
        finally:
            image.Release()
        if raw.ndim == 2:
            if self._cv_code is not None:
                bgr = cv2.cvtColor(raw, self._cv_code)
            else:
                if raw.dtype != np.uint8:
                    raw8 = (raw >> (raw.dtype.itemsize * 8 - 8)).astype(np.uint8)
                else:
                    raw8 = raw
                bgr = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
        elif raw.shape[2] == 3:
            bgr = raw if self.pixel_format.startswith("BGR") else cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        # capture time: camera timestamp mapped to host clock (mid-exposure), else arrival - latency
        if hw_ts is not None and self._ts_offset is not None:
            t_capture = hw_ts * 1e-9 + self._ts_offset
            if str(self.cfg.get("timestamp_ref", "start_of_exposure")) == "start_of_exposure":
                t_capture += 0.5 * self.exposure_us * 1e-6
            if not (t_host - 0.5 < t_capture <= t_host + 0.002):
                t_capture = t_host - float(self.cfg.get("latency_ms", 10.0)) / 1000.0
        else:
            t_capture = t_host - float(self.cfg.get("latency_ms", 10.0)) / 1000.0
        if t_host - self._last_sync > 10.0 and self._ts_offset is not None:
            self._try("resync clock", self._sync_clock)
        keep_raw = raw if (raw.ndim == 2 and self.cfg.get("log_raw_bayer", True)) else None
        return bgr, t_capture, {"hw_ts": hw_ts, "raw": keep_raw}

    def close(self) -> None:
        cam = self.cam
        if cam is not None:
            self._try("end acquisition", lambda: cam.EndAcquisition())
            self._try("deinit camera", lambda: cam.DeInit())
            del cam
            self.cam = None
        if self.cam_list is not None:
            self._try("clear camera list", lambda: self.cam_list.Clear())
            self.cam_list = None
        if self.system is not None:
            time.sleep(0.05)
            self._try("release system", lambda: self.system.ReleaseInstance())
            self.system = None

    def info(self) -> dict:
        d = super().info()
        d.update({"pixel_format": self.pixel_format, "dropped_incomplete": self.dropped,
                  "clock_synced": self._ts_offset is not None, "exposure_us": self.exposure_us})
        return d
