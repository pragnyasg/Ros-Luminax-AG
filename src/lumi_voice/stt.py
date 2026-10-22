"""Speech-to-Text module using faster-whisper."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from lumi_voice.config import STTConfig

logger = logging.getLogger(__name__)


class SpeechToText:
    """Transcribes audio using faster-whisper (CTranslate2 Whisper implementation)."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> None:
        """Load the Whisper model. Downloads if not cached."""
        try:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model: size=%s, device=%s, compute=%s",
                self.config.model_size,
                self.config.device,
                self.config.compute_type,
            )
            start = time.monotonic()
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            elapsed = time.monotonic() - start
            self._is_loaded = True
            logger.info("Whisper model loaded in %.2fs", elapsed)
        except ImportError:
            logger.error(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            )
            raise
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise

    def transcribe_audio(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio data to text.

        Args:
            audio: NumPy array of audio samples (float32, mono, 16kHz expected).

        Returns:
            Transcribed text or None if transcription is empty/invalid.
        """
        if not self._is_loaded or self._model is None:
            logger.error("STT model not loaded. Call load_model() first.")
            return None

        # Ensure correct shape: (samples,) mono
        if audio.ndim > 1:
            audio = audio.flatten()

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            logger.warning("Empty audio data provided")
            return None

        try:
            start = time.monotonic()
            segments, info = self._model.transcribe(
                audio,
                language=self.config.language if self.config.language else None,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
            )

            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            full_text = " ".join(text_parts).strip()
            elapsed = time.monotonic() - start

            if len(full_text) < self.config.min_transcript_length:
                logger.debug(
                    "Transcript too short (%d chars): '%s'",
                    len(full_text),
                    full_text,
                )
                return None

            logger.info(
                "Transcription (%.2fs, lang=%s, prob=%.2f): %s",
                elapsed,
                info.language,
                info.language_probability,
                full_text,
            )
            return full_text

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return None

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._is_loaded = False
        logger.info("Whisper model unloaded")
