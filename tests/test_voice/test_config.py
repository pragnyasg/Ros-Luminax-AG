"""Tests for configuration loading and validation."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from lumi_voice.config import (
    LumiConfig,
    load_config,
    save_user_config,
    _deep_merge,
    AudioInputConfig,
    STTConfig,
    LLMConfig,
    TTSConfig,
)


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_empty_override(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}


class TestLumiConfig:
    def test_default_values(self):
        config = LumiConfig()
        assert config.audio_input.sample_rate == 16000
        assert config.stt.model_size == "tiny"
        assert config.llm.backend == "ollama"
        assert config.tts.voice_model == "en_US-lessac-medium"
        assert config.interaction.mode == "push_to_talk"
        assert config.ros2.enabled is False

    def test_custom_values(self):
        config = LumiConfig(
            stt=STTConfig(model_size="base", device="cuda"),
            llm=LLMConfig(backend="ollama"),
        )
        assert config.stt.model_size == "base"
        assert config.stt.device == "cuda"

    def test_audio_input_defaults(self):
        cfg = AudioInputConfig()
        assert cfg.device_index is None
        assert cfg.channels == 1
        assert cfg.dtype == "float32"


class TestLoadConfig:
    def test_load_nonexistent_defaults(self):
        # Should fall back to Pydantic defaults
        config = load_config(
            default_path="/nonexistent/default.yaml",
            user_path="/nonexistent/user.yaml",
        )
        assert isinstance(config, LumiConfig)
        assert config.stt.model_size == "tiny"

    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "stt": {"model_size": "small", "language": "fr"},
            "llm": {"ollama": {"model": "llama3"}},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(default_path=config_file, user_path="/nonexistent")
        assert config.stt.model_size == "small"
        assert config.stt.language == "fr"
        assert config.llm.ollama.model == "llama3"

    def test_user_config_overrides_default(self, tmp_path):
        default_data = {"stt": {"model_size": "tiny"}}
        user_data = {"stt": {"model_size": "base"}}

        default_file = tmp_path / "default.yaml"
        user_file = tmp_path / "user.yaml"

        with open(default_file, "w") as f:
            yaml.dump(default_data, f)
        with open(user_file, "w") as f:
            yaml.dump(user_data, f)

        config = load_config(default_path=default_file, user_path=user_file)
        assert config.stt.model_size == "base"

    def test_explicit_overrides(self):
        config = load_config(
            default_path="/nonexistent",
            user_path="/nonexistent",
            overrides={"stt": {"model_size": "large-v3"}},
        )
        assert config.stt.model_size == "large-v3"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "test-model")
        config = load_config(
            default_path="/nonexistent",
            user_path="/nonexistent",
        )
        assert config.llm.ollama.model == "test-model"


class TestSaveConfig:
    def test_save_and_reload(self, tmp_path):
        config = LumiConfig(stt=STTConfig(model_size="medium"))
        save_path = tmp_path / "saved.yaml"
        save_user_config(config, path=save_path)

        assert save_path.exists()
        reloaded = load_config(
            default_path="/nonexistent",
            user_path=save_path,
        )
        assert reloaded.stt.model_size == "medium"
