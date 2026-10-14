"""
LUMI Vision - Standalone Computer Vision Module for LUMI Healthcare Companion Robot.

Provides real-time face detection (YuNet), face feature embedding & recognition (SFace),
a persistent identity database, and a hardware camera abstraction.
"""

from luminax_vision.config import Config, load_config
from luminax_vision.camera import Camera, MockCamera, BaseCamera
from luminax_vision.face_detector import FaceDetector, FaceDetection
from luminax_vision.face_recognizer import FaceRecognizer, RecognitionResult
from luminax_vision.face_database import FaceDatabase

__version__ = "0.1.0"
__all__ = [
    "Config",
    "load_config",
    "Camera",
    "MockCamera",
    "BaseCamera",
    "FaceDetector",
    "FaceDetection",
    "FaceRecognizer",
    "RecognitionResult",
    "FaceDatabase",
    "__version__",
]
