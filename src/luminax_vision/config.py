"""
Configuration loader and validator for LUMI Vision.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class CameraConfig:
    index: Union[int, str] = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class DetectionConfig:
    model_path: Path = Path("models/face_detection_yunet_2023mar.onnx")
    score_threshold: float = 0.80
    nms_threshold: float = 0.30
    top_k: int = 5000


@dataclass
class RecognitionConfig:
    model_path: Path = Path("models/face_recognition_sface_2021dec.onnx")
    distance_metric: str = "cosine"
    cosine_threshold: float = 0.363
    l2_threshold: float = 1.128


@dataclass
class DatabaseConfig:
    path: Path = Path("data/identities.json")


@dataclass
class DisplayConfig:
    show_window: bool = True
    window_name: str = "LUMI Vision - Healthcare Companion"
    show_fps: bool = True
    show_score: bool = True
    show_landmarks: bool = False
    box_color_known: Tuple[int, int, int] = (0, 220, 100)
    box_color_unknown: Tuple[int, int, int] = (0, 80, 240)


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    project_root: Path = field(default_factory=Path.cwd)


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Traverse upwards to find the project root marked by pyproject.toml, .git, or requirements.txt.
    """
    current = (start_path or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file() or (parent / "requirements.txt").is_file() or (parent / ".git").exists():
            return parent
    return current


def _resolve_path(root: Path, path_val: Union[str, Path]) -> Path:
    p = Path(path_val)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def validate_config(cfg: Config) -> None:
    """Validate configuration fields and ranges."""
    # Camera validation
    if isinstance(cfg.camera.index, int) and cfg.camera.index < 0:
        raise ValueError(f"Invalid camera.index: {cfg.camera.index}. Must be >= 0.")
    if cfg.camera.width <= 0 or cfg.camera.height <= 0:
        raise ValueError(f"Invalid camera resolution ({cfg.camera.width}x{cfg.camera.height}). Values must be > 0.")
    if cfg.camera.fps <= 0:
        raise ValueError(f"Invalid camera.fps: {cfg.camera.fps}. Must be > 0.")

    # Detection validation
    if not (0.0 <= cfg.detection.score_threshold <= 1.0):
        raise ValueError(f"Invalid detection.score_threshold: {cfg.detection.score_threshold}. Must be between 0.0 and 1.0.")
    if not (0.0 <= cfg.detection.nms_threshold <= 1.0):
        raise ValueError(f"Invalid detection.nms_threshold: {cfg.detection.nms_threshold}. Must be between 0.0 and 1.0.")
    if cfg.detection.top_k < 1:
        raise ValueError(f"Invalid detection.top_k: {cfg.detection.top_k}. Must be >= 1.")

    # Recognition validation
    valid_metrics = ("cosine", "l2")
    if cfg.recognition.distance_metric.lower() not in valid_metrics:
        raise ValueError(
            f"Invalid recognition.distance_metric: '{cfg.recognition.distance_metric}'. Must be one of {valid_metrics}."
        )
    if cfg.recognition.cosine_threshold < -1.0 or cfg.recognition.cosine_threshold > 1.0:
        raise ValueError(f"Invalid recognition.cosine_threshold: {cfg.recognition.cosine_threshold}. Must be between -1.0 and 1.0.")
    if cfg.recognition.l2_threshold < 0.0:
        raise ValueError(f"Invalid recognition.l2_threshold: {cfg.recognition.l2_threshold}. Must be >= 0.0.")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Config:
    """
    Load configuration from YAML file, with automatic fallback and resolution.

    Args:
        config_path: Explicit path to YAML config file. If None, checks default locations.
        project_root: Project root directory. If None, auto-detected.

    Returns:
        Validated Config dataclass instance.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()
    active_path: Optional[Path] = None

    if config_path is not None:
        p = Path(config_path)
        active_path = p if p.is_absolute() else (root / p)
        if not active_path.is_file():
            raise FileNotFoundError(f"Specified configuration file not found: {active_path}")
    else:
        # Check environment variable
        env_cfg = os.environ.get("LUMI_VISION_CONFIG")
        if env_cfg:
            p = Path(env_cfg)
            active_path = p if p.is_absolute() else (root / p)

        # Check default paths
        if not active_path or not active_path.is_file():
            candidate_active = root / "config" / "config.yaml"
            candidate_example = root / "config" / "config.example.yaml"

            if candidate_active.is_file():
                active_path = candidate_active
            elif candidate_example.is_file():
                # Auto-copy example config to config.yaml if it does not exist
                candidate_active.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate_example, candidate_active)
                active_path = candidate_active
            else:
                active_path = None

    raw_data: Dict[str, Any] = {}
    if active_path and active_path.is_file():
        try:
            with open(active_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_data = loaded
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML configuration at {active_path}: {exc}") from exc

    # Parse camera
    cam_raw = raw_data.get("camera", {})
    camera = CameraConfig(
        index=cam_raw.get("index", 0),
        width=int(cam_raw.get("width", 640)),
        height=int(cam_raw.get("height", 480)),
        fps=int(cam_raw.get("fps", 30)),
    )

    # Parse detection
    det_raw = raw_data.get("detection", {})
    det_model = _resolve_path(root, det_raw.get("model_path", "models/face_detection_yunet_2023mar.onnx"))
    detection = DetectionConfig(
        model_path=det_model,
        score_threshold=float(det_raw.get("score_threshold", 0.80)),
        nms_threshold=float(det_raw.get("nms_threshold", 0.30)),
        top_k=int(det_raw.get("top_k", 5000)),
    )

    # Parse recognition
    rec_raw = raw_data.get("recognition", {})
    rec_model = _resolve_path(root, rec_raw.get("model_path", "models/face_recognition_sface_2021dec.onnx"))
    recognition = RecognitionConfig(
        model_path=rec_model,
        distance_metric=str(rec_raw.get("distance_metric", "cosine")).lower(),
        cosine_threshold=float(rec_raw.get("cosine_threshold", 0.363)),
        l2_threshold=float(rec_raw.get("l2_threshold", 1.128)),
    )

    # Parse database
    db_raw = raw_data.get("database", {})
    db_path = _resolve_path(root, db_raw.get("path", "data/identities.json"))
    database = DatabaseConfig(path=db_path)

    # Parse display
    disp_raw = raw_data.get("display", {})
    box_k = disp_raw.get("box_color_known", [0, 220, 100])
    box_u = disp_raw.get("box_color_unknown", [0, 80, 240])
    display = DisplayConfig(
        show_window=bool(disp_raw.get("show_window", True)),
        window_name=str(disp_raw.get("window_name", "LUMI Vision - Healthcare Companion")),
        show_fps=bool(disp_raw.get("show_fps", True)),
        show_score=bool(disp_raw.get("show_score", True)),
        show_landmarks=bool(disp_raw.get("show_landmarks", False)),
        box_color_known=(int(box_k[0]), int(box_k[1]), int(box_k[2])),
        box_color_unknown=(int(box_u[0]), int(box_u[1]), int(box_u[2])),
    )

    config = Config(
        camera=camera,
        detection=detection,
        recognition=recognition,
        database=database,
        display=display,
        project_root=root,
    )

    validate_config(config)
    return config
