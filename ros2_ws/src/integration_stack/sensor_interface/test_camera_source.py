"""PROJECT23-14 preliminary injected-frame tests; real camera hardware is untested."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_source import CameraConfig, CameraSource, load_camera_config, normalize_frame


class InjectedCapture:
    def __init__(self, frame, opened=True):
        self.frame = frame
        self.connected = opened
        self.released = False

    def isOpened(self):
        return self.connected and not self.released

    def read(self):
        return (True, self.frame) if self.isOpened() else (False, None)

    def release(self):
        self.released = True


def test_bgr_pixels_and_resolution_are_preserved():
    frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[:, ::-1]
    result = normalize_frame(frame, CameraConfig())
    np.testing.assert_array_equal(result, frame)
    assert result.flags.c_contiguous


@pytest.mark.parametrize("single_channel_axis", [False, True])
def test_mono8_reaches_existing_three_channel_contract(single_channel_axis):
    gray = np.array([[0, 80, 255]], dtype=np.uint8)
    raw = gray[:, :, None] if single_channel_axis else gray
    frame = normalize_frame(raw, CameraConfig(spectrum="swir", pixel_format="mono8"))
    assert frame.shape == (1, 3, 3)
    for channel in range(3):
        np.testing.assert_array_equal(frame[:, :, channel], gray)


def test_mono16_uses_fixed_white_level_and_clips_without_wrapping():
    raw = np.array([[0, 2048, 4095, 65535]], dtype=np.uint16)
    frame = normalize_frame(raw, CameraConfig(spectrum="swir", pixel_format="mono16", white_level=4095))
    assert frame.dtype == np.uint8
    np.testing.assert_array_equal(frame[0, :, 0], [0, 128, 255, 255])
    np.testing.assert_array_equal(frame[:, :, 0], frame[:, :, 2])
    # A dim frame must not be independently stretched to full brightness.
    dim = normalize_frame(np.array([[1024]], dtype=np.uint16),
                          CameraConfig(pixel_format="mono16", white_level=4095))
    assert dim[0, 0, 0] == 64


@pytest.mark.parametrize("frame,format", [
    (None, "bgr8"),
    (np.empty((0, 4, 3), dtype=np.uint8), "bgr8"),
    (np.zeros((3, 4), dtype=np.float32), "mono16"),
    (np.zeros((3, 4, 2), dtype=np.uint8), "bgr8"),
    (np.zeros((3, 4, 3), dtype=np.uint8), "mono8"),
])
def test_unsupported_frames_are_rejected(frame, format):
    with pytest.raises(ValueError):
        normalize_frame(frame, CameraConfig(pixel_format=format))


def test_restart_releases_visible_before_opening_swir_and_handles_disconnect():
    visible = InjectedCapture(np.zeros((48, 64, 3), dtype=np.uint8))
    swir = InjectedCapture(np.full((24, 32), 4095, dtype=np.uint16))
    opened = []

    def factory(config):
        if opened:
            assert opened[-1].released, "old device must be released before opening the next"
        capture = visible if config.spectrum == "visible" else swir
        opened.append(capture)
        return capture

    source = CameraSource(capture_factory=factory)
    assert source.restart()
    assert source.read()[1].shape == (48, 64, 3)
    visible.connected = False  # Inject physical-unplug equivalent at the capture boundary.
    assert source.read() == (False, None)
    assert "disconnected" in source.last_error
    assert visible.released
    assert source.restart(CameraConfig(spectrum="swir", pixel_format="mono16", white_level=4095))
    ok, frame = source.read()
    assert ok and frame.shape == (24, 32, 3) and frame.dtype == np.uint8
    assert np.all(frame == 255)
    assert source.last_error == ""
    source.release()
    assert swir.released


@pytest.mark.parametrize("capture", [
    InjectedCapture(np.zeros((2, 2, 3), dtype=np.uint8), opened=False),
    InjectedCapture(None),
    InjectedCapture(np.zeros((2, 2), dtype=np.float32)),
])
def test_failed_start_releases_device_and_returns_no_frame(capture):
    source = CameraSource(capture_factory=lambda config: capture)
    assert not source.restart()
    assert capture.released and source.last_error
    assert source.read() == (False, None)


def test_opencv_configuration_error_releases_capture(monkeypatch):
    import camera_source

    capture = InjectedCapture(np.zeros((2, 2), dtype=np.uint16))

    def fail_set(*args):
        raise camera_source.cv2.error("Injected backend configuration failure")

    capture.set = fail_set
    monkeypatch.setattr(camera_source.cv2, "VideoCapture", lambda device: capture)
    source = CameraSource(CameraConfig(spectrum="swir", pixel_format="mono16"))
    assert not source.restart()
    assert capture.released and "configuration failure" in source.last_error


@pytest.mark.parametrize("contents", ['[]', '{"backend": "unknown"}',
    '{"white_level": 0}', '{"device": true}', '{"unexpected": 1}', '{bad json'])
def test_invalid_configuration_reports_error(tmp_path, contents):
    path = tmp_path / "camera.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_camera_config(path)


@pytest.mark.parametrize("name", ["camera_visible_mock.json", "camera_swir_mock.json"])
def test_checked_in_mock_profiles_open_without_hardware(name):
    src = Path(__file__).resolve().parents[2]
    source = CameraSource(load_camera_config(src / "bringup" / "config" / name))
    try:
        assert source.restart()
        ok, frame = source.read()
        assert ok and frame.dtype == np.uint8 and frame.shape == (480, 640, 3)
        assert "PRELIMINARY MOCK" in source.config.label
    finally:
        source.release()
