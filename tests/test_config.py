"""
Unit tests for configuration loading and validation in LUMI Vision.
"""

from pathlib import Path
import pytest

from luminax_vision.config import Config, load_config, validate_config, CameraConfig, DetectionConfig, RecognitionConfig


def test_load_default_config(tmp_path: Path):
    """Test loading configuration from example file and default resolution."""
    config = load_config(project_root=tmp_path)
    assert config is not None
    assert config.camera.width == 640
    assert config.camera.height == 480
    assert config.detection.score_threshold == 0.80
    assert config.recognition.distance_metric == "cosine"
    assert config.recognition.cosine_threshold == 0.363


def test_invalid_camera_dimensions():
    """Verify that negative or zero camera resolutions raise ValueError."""
    cfg = Config()
    cfg.camera.width = -1
    with pytest.raises(ValueError, match="Invalid camera resolution"):
        validate_config(cfg)

    cfg2 = Config()
    cfg2.camera.height = 0
    with pytest.raises(ValueError, match="Invalid camera resolution"):
        validate_config(cfg2)


def test_invalid_camera_fps():
    """Verify that non-positive fps raises ValueError."""
    cfg = Config()
    cfg.camera.fps = 0
    with pytest.raises(ValueError, match="Invalid camera.fps"):
        validate_config(cfg)


def test_invalid_score_threshold():
    """Verify that score threshold outside [0, 1] raises ValueError."""
    cfg = Config()
    cfg.detection.score_threshold = 1.5
    with pytest.raises(ValueError, match="Invalid detection.score_threshold"):
        validate_config(cfg)

    cfg.detection.score_threshold = -0.1
    with pytest.raises(ValueError, match="Invalid detection.score_threshold"):
        validate_config(cfg)


def test_invalid_distance_metric():
    """Verify that unsupported distance metric raises ValueError."""
    cfg = Config()
    cfg.recognition.distance_metric = "unsupported_metric"
    with pytest.raises(ValueError, match="Invalid recognition.distance_metric"):
        validate_config(cfg)


def test_custom_yaml_override(tmp_path: Path):
    """Verify loading from an explicit custom YAML configuration file."""
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
camera:
  index: 2
  width: 1280
  height: 720
  fps: 60
detection:
  score_threshold: 0.90
recognition:
  distance_metric: "l2"
  l2_threshold: 1.05
"""
    )
    loaded = load_config(config_path=custom_yaml, project_root=tmp_path)
    assert loaded.camera.index == 2
    assert loaded.camera.width == 1280
    assert loaded.camera.height == 720
    assert loaded.camera.fps == 60
    assert loaded.detection.score_threshold == 0.90
    assert loaded.recognition.distance_metric == "l2"
    assert loaded.recognition.l2_threshold == 1.05
