"""Voice Activity Detection — energy-based with silence-timeout end-of-utterance detection."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Detects speech segments using energy thresholds and silence timeouts."""

    def __init__(
        self,
        energy_threshold: float = 0.01,
        silence_duration_s: float = 1.5,
        min_speech_duration_s: float = 0.3,
        pre_speech_padding_s: float = 0.3,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.silence_duration_s = silence_duration_s
        self.min_speech_duration_s = min_speech_duration_s
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms

        # Pre-speech buffer for capturing onset
        chunks_per_second = 1000 / chunk_duration_ms
        buffer_size = max(1, int(pre_speech_padding_s * chunks_per_second))
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

        # State
        self._speech_chunks: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        """Reset detector state for a new utterance."""
        self._speech_chunks.clear()
        self._pre_buffer.clear()
        self._is_speaking = False
        self._silence_start = None
        self._speech_start = None

    def process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process an audio chunk.

        Returns:
            Complete utterance audio (np.ndarray) when speech ends,
            or None if still listening.
        """
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        is_speech = energy > self.energy_threshold
        now = time.monotonic()

        if not self._is_speaking:
            # Not currently in a speech segment
            if is_speech:
                # Speech onset
                self._is_speaking = True
                self._speech_start = now
                self._silence_start = None
                # Include pre-speech buffer
                self._speech_chunks = list(self._pre_buffer)
                self._speech_chunks.append(chunk)
                logger.debug("Speech onset detected (energy=%.4f)", energy)
            else:
                # Still silence — buffer for potential onset capture
                self._pre_buffer.append(chunk)
        else:
            # Currently in a speech segment
            self._speech_chunks.append(chunk)

            if is_speech:
                # Continue speech
                self._silence_start = None
            else:
                # Silence within speech
                if self._silence_start is None:
                    self._silence_start = now
                elif (now - self._silence_start) >= self.silence_duration_s:
                    # End of utterance
                    speech_duration = now - (self._speech_start or now)
                    if speech_duration >= self.min_speech_duration_s:
                        utterance = np.concatenate(self._speech_chunks, axis=0)
                        logger.info(
                            "Utterance complete: %.2fs, %d samples",
                            speech_duration,
                            len(utterance),
                        )
                        self.reset()
                        return utterance
                    else:
                        logger.debug(
                            "Speech too short (%.2fs), discarding",
                            speech_duration,
                        )
                        self.reset()

        return None
