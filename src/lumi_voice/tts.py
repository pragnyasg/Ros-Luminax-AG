"""Text-to-Speech module using Piper TTS."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from lumi_voice.config import TTSConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Synthesizes speech from text using Piper TTS.

    Piper is invoked as a command-line process. The voice model must be
    downloaded separately. See:
      https://github.com/rhasspy/piper/releases
      https://huggingface.co/rhasspy/piper-voices

    Voice models consist of:
      - An ONNX model file (e.g., en_US-lessac-medium.onnx)
      - A JSON config file (e.g., en_US-lessac-medium.onnx.json)
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._piper_path: Optional[str] = None
        self._is_ready = False
        self._check_piper()

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _check_piper(self) -> None:
        """Check if Piper is available on the system."""
        piper_cmd = shutil.which("piper") or shutil.which("piper.exe")
        if piper_cmd:
            self._piper_path = piper_cmd
            self._is_ready = True
            logger.info("Piper TTS found at: %s", piper_cmd)
        else:
            # Also check common locations
            common_paths = [
                PROJECT_ROOT / "piper" / "piper",
                PROJECT_ROOT / "piper" / "piper.exe",
                Path.home() / "piper" / "piper",
                Path.home() / "piper" / "piper.exe",
                Path(".") / "piper" / "piper",
                Path(".") / "piper" / "piper.exe",
            ]
            for p in common_paths:
                if p.exists():
                    self._piper_path = str(p)
                    self._is_ready = True
                    logger.info("Piper TTS found at: %s", p)
                    return

            self._is_ready = False
            logger.warning(
                "Piper TTS not found. Install it from: "
                "https://github.com/rhasspy/piper/releases\n"
                "Then add it to your PATH or place it in a ./piper/ directory."
            )

    def get_voice_model_path(self) -> Optional[str]:
        """Get the configured or resolved voice model path."""
        if self.config.voice_model_path:
            # Check direct or relative to PROJECT_ROOT
            raw_path = Path(self.config.voice_model_path)
            if raw_path.exists():
                return str(raw_path)
            proj_path = PROJECT_ROOT / self.config.voice_model_path
            if proj_path.exists():
                return str(proj_path)
            return self.config.voice_model_path

        # Auto-detect voice model in voices/
        candidates = [
            PROJECT_ROOT / "voices" / f"{self.config.voice_model}.onnx",
            Path("voices") / f"{self.config.voice_model}.onnx",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

        # Fallback to model name
        return self.config.voice_model

    def check_availability(self) -> tuple[bool, str]:
        """Check if TTS is ready."""
        if not self._is_ready:
            return False, (
                "Piper TTS not found. Install from: "
                "https://github.com/rhasspy/piper/releases "
                "and add to PATH."
            )

        model_path = self.get_voice_model_path()
        if self.config.voice_model_path:
            raw_path = Path(self.config.voice_model_path)
            proj_path = PROJECT_ROOT / self.config.voice_model_path
            if not raw_path.exists() and not proj_path.exists():
                return False, f"Voice model not found at: {self.config.voice_model_path}"

        return True, f"Piper TTS ready (voice: {model_path})"

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to WAV audio bytes.

        Args:
            text: Text to convert to speech.

        Returns:
            WAV audio data as bytes, or None on failure.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return None

        if not self._is_ready or not self._piper_path:
            logger.error("Piper TTS is not available")
            return None

        try:
            start = time.monotonic()

            cmd = [self._piper_path]

            # Model specification
            model_path = self.get_voice_model_path()
            if model_path:
                cmd.extend(["--model", model_path])

            # Output format
            cmd.extend(["--output-raw"])

            # Length scale (speed)
            if self.config.length_scale != 1.0:
                cmd.extend(["--length-scale", str(self.config.length_scale)])

            # Sentence silence
            if self.config.sentence_silence_s > 0:
                cmd.extend(["--sentence-silence", str(self.config.sentence_silence_s)])

            # Speaker ID for multi-speaker models
            if self.config.speaker_id is not None:
                cmd.extend(["--speaker", str(self.config.speaker_id)])

            logger.debug("Piper command: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                logger.error("Piper TTS failed: %s", stderr)
                return None

            raw_audio = result.stdout
            if not raw_audio:
                logger.error("Piper produced no audio output")
                return None

            # Convert raw PCM to WAV
            # Piper outputs 16-bit PCM at 22050 Hz mono by default
            sample_rate = 22050
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(raw_audio)

            wav_data = wav_buffer.getvalue()
            elapsed = time.monotonic() - start
            logger.info(
                "TTS synthesis completed in %.2fs (%d bytes audio)",
                elapsed,
                len(wav_data),
            )
            return wav_data

        except subprocess.TimeoutExpired:
            logger.error("Piper TTS timed out")
            return None
        except FileNotFoundError:
            logger.error("Piper executable not found at: %s", self._piper_path)
            self._is_ready = False
            return None
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return None

    def synthesize_to_file(self, text: str, output_path: str | Path) -> bool:
        """Synthesize text and save to a WAV file."""
        wav_data = self.synthesize(text)
        if wav_data is None:
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_data)
        logger.info("Audio saved to: %s", output_path)
        return True
