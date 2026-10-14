"""
Unit tests for camera abstraction (BaseCamera, MockCamera, and Camera).
"""

import numpy as np
import pytest

from luminax_vision.camera import Camera, MockCamera


def test_mock_camera_stream():
    """Verify MockCamera frame generation, shapes, and properties."""
    with MockCamera(width=640, height=480, fps=30) as cam:
        assert cam.is_opened() is True
        assert cam.get_resolution() == (640, 480)

        ret, frame = cam.read()
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    assert cam.is_opened() is False


def test_mock_camera_max_frames():
    """Verify MockCamera terminates cleanly when max_frames is reached."""
    cam = MockCamera(width=320, height=240, max_frames=3)
    frames_read = 0
    while True:
        ret, _ = cam.read()
        if not ret:
            break
        frames_read += 1

    assert frames_read == 3
    cam.release()


def test_mock_camera_static_image():
    """Verify MockCamera replay of a static image."""
    static = np.full((100, 100, 3), 127, dtype=np.uint8)
    with MockCamera(width=200, height=200, static_image=static) as cam:
        ret, frame = cam.read()
        assert ret is True
        assert frame.shape == (200, 200, 3)
        assert np.all(frame == 127)


def test_physical_camera_invalid_index():
    """Verify that an invalid or disconnected camera index raises ConnectionError."""
    # Attempting to open non-existent camera index 999
    with pytest.raises(ConnectionError, match="could not be opened"):
        Camera(index=999)
