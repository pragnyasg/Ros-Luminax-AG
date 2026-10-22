"""Tests for audio device management."""

from unittest.mock import patch, MagicMock

import pytest

from lumi_voice.audio_devices import AudioDevice, AudioDeviceManager


class TestAudioDevice:
    def test_is_input(self):
        dev = AudioDevice(index=0, name="Mic", max_input_channels=2)
        assert dev.is_input is True
        assert dev.is_output is False

    def test_is_output(self):
        dev = AudioDevice(index=1, name="Speaker", max_output_channels=2)
        assert dev.is_input is False
        assert dev.is_output is True

    def test_display_name_default(self):
        dev = AudioDevice(
            index=0, name="My Mic",
            max_input_channels=1,
            is_default_input=True,
        )
        name = dev.display_name()
        assert "My Mic" in name
        assert "Default Input" in name

    def test_display_name_no_defaults(self):
        dev = AudioDevice(index=2, name="USB Audio")
        assert dev.display_name() == "[2] USB Audio"


class TestAudioDeviceManager:
    @patch("lumi_voice.audio_devices.sd")
    def test_refresh_enumerates_devices(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {
                "name": "Built-in Mic",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
            {
                "name": "Built-in Speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
        ]
        mock_sd.query_hostapis.return_value = [{"name": "Core Audio"}]
        mock_sd.default.device = [0, 1]

        mgr = AudioDeviceManager()
        assert len(mgr.input_devices) == 1
        assert len(mgr.output_devices) == 1
        assert mgr.input_devices[0].name == "Built-in Mic"
        assert mgr.output_devices[0].name == "Built-in Speaker"

    @patch("lumi_voice.audio_devices.sd")
    def test_select_input(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000, "hostapi": 0}
        ]
        mock_sd.query_hostapis.return_value = [{"name": "API"}]
        mock_sd.default.device = [0, 0]

        mgr = AudioDeviceManager()
        dev = mgr.select_input(0)
        assert dev is not None
        assert dev.name == "Mic"
        assert mgr.selected_input == dev

    @patch("lumi_voice.audio_devices.sd")
    def test_select_nonexistent(self, mock_sd):
        mock_sd.query_devices.return_value = []
        mock_sd.query_hostapis.return_value = []
        mock_sd.default.device = [0, 0]

        mgr = AudioDeviceManager()
        dev = mgr.select_input(999)
        assert dev is None

    @patch("lumi_voice.audio_devices.sd")
    def test_select_none_resets(self, mock_sd):
        mock_sd.query_devices.return_value = []
        mock_sd.query_hostapis.return_value = []
        mock_sd.default.device = [0, 0]

        mgr = AudioDeviceManager()
        mgr.select_input(None)
        assert mgr.selected_input is None
        assert mgr.get_input_device_index() is None
