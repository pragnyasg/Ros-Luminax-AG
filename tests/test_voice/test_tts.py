"""Tests for Text-to-Speech module."""

from unittest.mock import patch, MagicMock

import pytest

from lumi_voice.config import TTSConfig
from lumi_voice.tts import TextToSpeech


class TestTextToSpeech:
    @patch("lumi_voice.tts.Path.exists", return_value=False)
    @patch("lumi_voice.tts.shutil.which", return_value=None)
    def test_piper_not_found(self, mock_which, mock_exists):
        tts = TextToSpeech(TTSConfig())
        assert tts.is_ready is False

    @patch("lumi_voice.tts.shutil.which", return_value="/usr/bin/piper")
    def test_piper_found(self, mock_which):
        tts = TextToSpeech(TTSConfig())
        assert tts.is_ready is True

    def test_synthesize_empty_text(self):
        tts = TextToSpeech(TTSConfig())
        result = tts.synthesize("")
        assert result is None

    def test_synthesize_whitespace(self):
        tts = TextToSpeech(TTSConfig())
        result = tts.synthesize("   ")
        assert result is None

    @patch("lumi_voice.tts.Path.exists", return_value=False)
    @patch("lumi_voice.tts.shutil.which", return_value=None)
    def test_synthesize_when_unavailable(self, mock_which, mock_exists):
        tts = TextToSpeech(TTSConfig())
        result = tts.synthesize("Hello")
        assert result is None

    @patch("lumi_voice.tts.shutil.which", return_value="/usr/bin/piper")
    def test_check_availability_ready(self, mock_which):
        tts = TextToSpeech(TTSConfig())
        avail, msg = tts.check_availability()
        assert avail is True
        assert "ready" in msg.lower() or "Piper" in msg

    @patch("lumi_voice.tts.Path.exists", return_value=False)
    @patch("lumi_voice.tts.shutil.which", return_value=None)
    def test_check_availability_not_ready(self, mock_which, mock_exists):
        tts = TextToSpeech(TTSConfig())
        avail, msg = tts.check_availability()
        assert avail is False

    @patch("lumi_voice.tts.shutil.which", return_value="/usr/bin/piper")
    def test_check_availability_missing_model_path(self, mock_which):
        tts = TextToSpeech(TTSConfig(voice_model_path="/nonexistent/model.onnx"))
        avail, msg = tts.check_availability()
        assert avail is False
        assert "not found" in msg.lower()
