"""Audio playback module with device selection and queue management."""

from __future__ import annotations

import io
import logging
import threading
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Plays audio through a selected output device.

    Prevents overlapping playback by serializing play requests.
    """

    def __init__(self, device_index: Optional[int] = None) -> None:
        self.device_index = device_index
        self._playback_lock = threading.Lock()
        self._is_playing = False
        self._stop_event = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def set_device(self, device_index: Optional[int]) -> None:
        """Change the output device."""
        self.device_index = device_index
        logger.info("Output device set to: %s", device_index or "default")

    def play_wav_bytes(self, wav_data: bytes) -> bool:
        """Play WAV audio data. Blocks until playback completes.

        Returns True if playback was successful.
        """
        if not wav_data:
            logger.warning("No audio data to play")
            return False

        with self._playback_lock:
            self._is_playing = True
            self._stop_event.clear()

            try:
                # Parse WAV data
                wav_buffer = io.BytesIO(wav_data)
                with wave.open(wav_buffer, "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())

                # Convert to numpy array
                if sample_width == 2:
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:
                    audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                elif sample_width == 1:
                    audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                else:
                    logger.error("Unsupported sample width: %d", sample_width)
                    return False

                if channels > 1:
                    audio = audio.reshape(-1, channels)

                if self._stop_event.is_set():
                    return False

                # Play audio
                sd.play(audio, samplerate=sample_rate, device=self.device_index)
                sd.wait()

                if self._stop_event.is_set():
                    return False

                logger.debug(
                    "Playback complete (rate=%d, channels=%d, samples=%d)",
                    sample_rate,
                    channels,
                    len(audio),
                )
                return True

            except Exception as e:
                logger.error("Audio playback failed: %s", e)
                return False
            finally:
                self._is_playing = False

    def play_numpy(self, audio: np.ndarray, sample_rate: int = 22050) -> bool:
        """Play a numpy audio array. Blocks until complete."""
        with self._playback_lock:
            self._is_playing = True
            self._stop_event.clear()
            try:
                sd.play(audio, samplerate=sample_rate, device=self.device_index)
                sd.wait()
                return True
            except Exception as e:
                logger.error("Audio playback failed: %s", e)
                return False
            finally:
                self._is_playing = False

    def stop(self) -> None:
        """Stop current playback."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        self._is_playing = False
        logger.info("Playback stopped")
