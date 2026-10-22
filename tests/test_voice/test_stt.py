"""Tests for Speech-to-Text module."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from lumi_voice.config import STTConfig
from lumi_voice.stt import SpeechToText


class TestSpeechToText:
    def test_not_loaded_initially(self):
        stt = SpeechToText(STTConfig())
        assert stt.is_loaded is False

    def test_transcribe_without_loading(self):
        stt = SpeechToText(STTConfig())
        audio = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe_audio(audio)
        assert result is None

    def test_transcribe_empty_audio(self):
        stt = SpeechToText(STTConfig())
        stt._is_loaded = True
        stt._model = MagicMock()
        result = stt.transcribe_audio(np.array([], dtype=np.float32))
        assert result is None

    def test_transcribe_filters_short(self):
        stt = SpeechToText(STTConfig(min_transcript_length=5))
        stt._is_loaded = True
        mock_model = MagicMock()

        # Mock segment with short text
        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        stt._model = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        result = stt.transcribe_audio(audio)
        assert result is None  # "hi" is < 5 chars

    def test_transcribe_success(self):
        stt = SpeechToText(STTConfig(min_transcript_length=2))
        stt._is_loaded = True
        mock_model = MagicMock()

        mock_segment = MagicMock()
        mock_segment.text = "Hello LUMI how are you"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        stt._model = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        result = stt.transcribe_audio(audio)
        assert result == "Hello LUMI how are you"

    def test_unload_model(self):
        stt = SpeechToText(STTConfig())
        stt._is_loaded = True
        stt._model = MagicMock()
        stt.unload_model()
        assert stt.is_loaded is False
        assert stt._model is None

    def test_transcribe_2d_audio(self):
        """Test that 2D audio (multi-channel) gets flattened."""
        stt = SpeechToText(STTConfig(min_transcript_length=2))
        stt._is_loaded = True
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Test audio"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        stt._model = mock_model

        audio = np.random.randn(16000, 1).astype(np.float32)  # 2D
        result = stt.transcribe_audio(audio)
        assert result == "Test audio"
        # Verify the model received 1D audio
        call_args = mock_model.transcribe.call_args
        assert call_args[0][0].ndim == 1
