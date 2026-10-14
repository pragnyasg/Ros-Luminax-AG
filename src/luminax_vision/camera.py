"""
Camera abstraction interface for LUMI Vision.

Provides a clean hardware abstraction layer supporting laptop webcams,
external USB webcams on Raspberry Pi / Linux (/dev/video*), and synthetic MockCamera
for headless testing and CI environments.
"""

from __future__ import annotations

import abc
import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseCamera(abc.ABC):
    """Abstract base class defining the standard camera contract."""

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. Returns (success, frame_bgr)."""
        pass

    @abc.abstractmethod
    def is_opened(self) -> bool:
        """Check if camera device is open and accessible."""
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """Release underlying camera resources cleanly."""
        pass

    @abc.abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Return (width, height) of frames."""
        pass

    def __enter__(self) -> BaseCamera:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class Camera(BaseCamera):
    """
    Physical camera implementation wrapping OpenCV VideoCapture.

    Supports:
    - Built-in laptop webcam (index 0)
    - External USB cameras on Raspberry Pi / Linux (index 1, 2 or '/dev/video0')
    - Video files or RTSP streams (path string)
    """

    def __init__(
        self,
        index: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: Optional[int] = None,
    ) -> None:
        self.index = index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

        self._open_capture(backend)

    def _open_capture(self, backend: Optional[int] = None) -> None:
        """Initialize OpenCV VideoCapture with fallback options."""
        # Convert string digits to int if passed as string index
        target_index: Union[int, str] = self.index
        if isinstance(target_index, str) and target_index.isdigit():
            target_index = int(target_index)

        try:
            if backend is not None:
                self._cap = cv2.VideoCapture(target_index, backend)
            else:
                self._cap = cv2.VideoCapture(target_index)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to initialize camera at index/path '{self.index}': {exc}"
            ) from exc

        if not self._cap.isOpened():
            raise ConnectionError(
                f"Camera device '{self.index}' could not be opened. "
                "Verify camera connection, check /dev/video* permissions, or check camera index."
            )

        # Request resolution & FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.target_width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.target_height))
        self._cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

        # Log actual negotiated settings
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"Camera '{self.index}' opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Grab and decode the next video frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, frame_bgr) on success,
                                               (False, None) on failure.
        """
        if self._cap is None or not self._cap.isOpened():
            logger.warning("Camera.read() called while camera is closed.")
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning(f"Failed to read valid frame from camera '{self.index}'.")
            return False, None

        return True, frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_resolution(self) -> Tuple[int, int]:
        if not self.is_opened():
            return 0, 0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def get_fps(self) -> float:
        if not self.is_opened():
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def release(self) -> None:
        if self._cap is not None:
            logger.debug(f"Releasing camera '{self.index}'.")
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __del__(self) -> None:
        self.release()


class MockCamera(BaseCamera):
    """
    Simulated camera for automated unit tests, CI pipelines, and headless servers.
    Generates synthetic frames or replays a static image.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        static_image: Optional[np.ndarray] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.max_frames = max_frames
        self.frame_count = 0
        self._opened = True

        if static_image is not None:
            self.static_image = cv2.resize(static_image, (width, height))
        else:
            self.static_image = None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None

        if self.max_frames is not None and self.frame_count >= self.max_frames:
            return False, None

        self.frame_count += 1

        if self.static_image is not None:
            return True, self.static_image.copy()

        # Generate a synthetic test pattern frame with subtle animation
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = np.linspace(20, 60, self.width, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(20, 40, self.width, dtype=np.uint8)
        frame[:, :, 2] = np.linspace(30, 80, self.width, dtype=np.uint8)

        # Draw a moving reference circle to verify video fluidity
        cx = int((self.width / 2) + 100 * np.sin(self.frame_count * 0.1))
        cy = int(self.height / 2)
        cv2.circle(frame, (cx, cy), 30, (0, 200, 255), -1)
        cv2.putText(
            frame,
            f"MOCK CAMERA FRAME #{self.frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return True, frame

    def is_opened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self._opened = False

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height
