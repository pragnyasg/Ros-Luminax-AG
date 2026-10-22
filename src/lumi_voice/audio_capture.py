"""Audio capture from microphone with configurable device selection."""

from __future__ import annotations

import logging
import threading
import queue
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from a selected microphone device."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 100,
        device_index: Optional[int] = None,
        dtype: str = "float32",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.device_index = device_index
        self.dtype = dtype
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)

        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_recording = False
        self._lock = threading.Lock()
        self._level_callback: Optional[Callable[[float], None]] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def set_level_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for real-time audio level updates."""
        self._level_callback = callback

    def set_device(self, device_index: Optional[int]) -> None:
        """Change the capture device. Restarts stream if currently recording."""
        was_recording = self._is_recording
        if was_recording:
            self.stop()
        self.device_index = device_index
        if was_recording:
            self.start()

    def start(self) -> None:
        """Start capturing audio from the microphone."""
        with self._lock:
            if self._is_recording:
                logger.warning("Audio capture already running")
                return

            # Clear any stale audio
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=self.chunk_size,
                    device=self.device_index,
                    dtype=self.dtype,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_recording = True
                logger.info(
                    "Audio capture started (device=%s, rate=%d, channels=%d)",
                    self.device_index if self.device_index is not None else "default",
                    self.sample_rate,
                    self.channels,
                )
            except Exception as e:
                logger.error("Failed to start audio capture: %s", e)
                self._is_recording = False
                raise

    def stop(self) -> None:
        """Stop capturing audio."""
        with self._lock:
            self._is_recording = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning("Error stopping audio stream: %s", e)
                finally:
                    self._stream = None
                logger.info("Audio capture stopped")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback invoked by sounddevice for each audio chunk."""
        if status:
            logger.warning("Audio capture status: %s", status)
        if self._is_recording:
            chunk = indata.copy()
            self._audio_queue.put(chunk)
            if self._level_callback is not None:
                level = float(np.max(np.abs(chunk)))
                try:
                    self._level_callback(level)
                except Exception:
                    pass  # Don't let callback errors crash audio

    def read_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Read one audio chunk from the queue. Returns None on timeout."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all_chunks(self) -> list[np.ndarray]:
        """Read all available audio chunks from the queue."""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def __del__(self) -> None:
        self.stop()
