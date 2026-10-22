"""Centralized configuration management for LUMI Voice Assistant."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
USER_CONFIG_PATH = CONFIG_DIR / "user.yaml"


class AudioInputConfig(BaseModel):
    device_index: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100
    dtype: str = "float32"


class AudioOutputConfig(BaseModel):
    device_index: Optional[int] = None
    sample_rate: int = 22050
    channels: int = 1


class VADConfig(BaseModel):
    energy_threshold: float = 0.01
    silence_duration_s: float = 1.5
    min_speech_duration_s: float = 0.3
    pre_speech_padding_s: float = 0.3


class STTConfig(BaseModel):
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    min_transcript_length: int = 2


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "gemma3:1b"
    timeout_s: int = 60
    stream: bool = True


class ConversationConfig(BaseModel):
    max_history: int = 20
    system_prompt: str = (
        "You are LUMI, a friendly, warm, and patient healthcare companion robot. "
        "You assist elderly users and people who need care. "
        "Keep your spoken responses concise — typically 1-3 sentences. "
        "Be helpful, kind, and conversational. Ask clarifying questions when needed. "
        "Be honest about your limitations. Never diagnose medical conditions. "
        "Never invent medication schedules or medical advice. "
        "Never present yourself as a substitute for professional medical care. "
        "If someone describes a medical emergency, urge them to call emergency services immediately."
    )


class LLMConfig(BaseModel):
    backend: str = "ollama"
    ollama: OllamaConfig = OllamaConfig()
    conversation: ConversationConfig = ConversationConfig()


class TTSConfig(BaseModel):
    voice_model: str = "en_US-lessac-medium"
    voice_model_path: Optional[str] = None
    speaker_id: Optional[int] = None
    length_scale: float = 1.0
    sentence_silence_s: float = 0.3


class InteractionConfig(BaseModel):
    mode: str = "push_to_talk"
    wake_word: Optional[str] = None


class PrivacyConfig(BaseModel):
    save_conversation_history: bool = False
    save_audio_recordings: bool = False
    log_transcripts: bool = False
    conversation_history_path: str = "logs/conversations"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None


class ROS2TopicsConfig(BaseModel):
    recognized_speech: str = "recognized_speech"
    response_text: str = "response_text"
    status: str = "voice_status"
    command: str = "voice_command"


class ROS2Config(BaseModel):
    enabled: bool = False
    namespace: str = "/lumi"
    topics: ROS2TopicsConfig = ROS2TopicsConfig()


class LumiConfig(BaseModel):
    """Root configuration model for LUMI Voice Assistant."""
    audio_input: AudioInputConfig = AudioInputConfig()
    audio_output: AudioOutputConfig = AudioOutputConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    interaction: InteractionConfig = InteractionConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    logging: LoggingConfig = LoggingConfig()
    ros2: ROS2Config = ROS2Config()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    default_path: Path | str | None = None,
    user_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> LumiConfig:
    """Load configuration from YAML files with optional overrides.

    Priority (highest to lowest):
      1. Explicit overrides dict
      2. User config file (config/user.yaml)
      3. Default config file (config/default.yaml)
      4. Pydantic model defaults
    """
    default_path = Path(default_path) if default_path else DEFAULT_CONFIG_PATH
    user_path = Path(user_path) if user_path else USER_CONFIG_PATH

    config_data: dict[str, Any] = {}

    # Load default config
    if default_path.exists():
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_data = loaded
                    logger.debug("Loaded default config from %s", default_path)
        except Exception as e:
            logger.warning("Failed to load default config %s: %s", default_path, e)
    else:
        logger.info("No default config file found at %s, using built-in defaults", default_path)

    # Load user config
    if user_path.exists():
        try:
            with open(user_path, "r", encoding="utf-8") as f:
                user_data = yaml.safe_load(f)
                if isinstance(user_data, dict):
                    config_data = _deep_merge(config_data, user_data)
                    logger.debug("Loaded user config from %s", user_path)
        except Exception as e:
            logger.warning("Failed to load user config %s: %s", user_path, e)

    # Apply explicit overrides
    if overrides:
        config_data = _deep_merge(config_data, overrides)

    # Apply environment variable overrides
    env_host = os.environ.get("OLLAMA_HOST")
    env_model = os.environ.get("OLLAMA_MODEL")
    env_log = os.environ.get("LOG_LEVEL")

    if env_host:
        config_data.setdefault("llm", {}).setdefault("ollama", {})["host"] = env_host
    if env_model:
        config_data.setdefault("llm", {}).setdefault("ollama", {})["model"] = env_model
    if env_log:
        config_data.setdefault("logging", {})["level"] = env_log

    return LumiConfig(**config_data)


def save_user_config(config: LumiConfig, path: Path | str | None = None) -> None:
    """Save current configuration to user config file."""
    path = Path(path) if path else USER_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved user config to %s", path)
