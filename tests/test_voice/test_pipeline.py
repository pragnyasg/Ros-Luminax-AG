"""Tests for the voice pipeline orchestration."""

from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

from lumi_voice.config import LumiConfig
from lumi_voice.pipeline import VoicePipeline, PipelineState, ComponentStatus


class TestComponentStatus:
    def test_defaults(self):
        status = ComponentStatus()
        assert status.microphone == "unavailable"
        assert status.stt == "unavailable"
        assert status.llm == "unavailable"
        assert status.tts == "unavailable"
        assert status.speaker == "unavailable"
        assert status.ros2 == "not configured"

    def test_to_dict(self):
        status = ComponentStatus()
        d = status.to_dict()
        assert isinstance(d, dict)
        assert "microphone" in d
        assert len(d) == 6


class TestPipelineState:
    def test_states_exist(self):
        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.LISTENING.value == "listening"
        assert PipelineState.TRANSCRIBING.value == "transcribing"
        assert PipelineState.THINKING.value == "thinking"
        assert PipelineState.SPEAKING.value == "speaking"


class TestVoicePipeline:
    @patch("lumi_voice.pipeline.AudioCapture")
    @patch("lumi_voice.pipeline.AudioPlayer")
    @patch("lumi_voice.pipeline.SpeechToText")
    @patch("lumi_voice.pipeline.LLMEngine")
    @patch("lumi_voice.pipeline.TextToSpeech")
    def test_pipeline_creation(self, mock_tts, mock_llm, mock_stt, mock_player, mock_capture):
        config = LumiConfig()
        pipeline = VoicePipeline(config=config)
        assert pipeline.state == PipelineState.IDLE

    @patch("lumi_voice.pipeline.AudioCapture")
    @patch("lumi_voice.pipeline.AudioPlayer")
    @patch("lumi_voice.pipeline.SpeechToText")
    @patch("lumi_voice.pipeline.LLMEngine")
    @patch("lumi_voice.pipeline.TextToSpeech")
    def test_clear_conversation(self, mock_tts, mock_llm, mock_stt, mock_player, mock_capture):
        config = LumiConfig()
        pipeline = VoicePipeline(config=config)
        pipeline.conversation.add_user_message("test")
        assert pipeline.conversation.message_count == 1
        pipeline.clear_conversation()
        assert pipeline.conversation.message_count == 0

    @patch("lumi_voice.pipeline.AudioCapture")
    @patch("lumi_voice.pipeline.AudioPlayer")
    @patch("lumi_voice.pipeline.SpeechToText")
    @patch("lumi_voice.pipeline.LLMEngine")
    @patch("lumi_voice.pipeline.TextToSpeech")
    def test_set_devices(self, mock_tts, mock_llm, mock_stt, mock_player, mock_capture):
        config = LumiConfig()
        pipeline = VoicePipeline(config=config)
        pipeline.set_input_device(2)
        assert config.audio_input.device_index == 2
        pipeline.set_output_device(3)
        assert config.audio_output.device_index == 3
