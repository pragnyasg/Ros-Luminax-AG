"""
LUMI Vision - Standalone Computer Vision Module for LUMI Healthcare Companion Robot.

Provides real-time face detection (YuNet), face feature embedding & recognition (SFace),
human body pose estimation (MediaPipe), person tracking, activity & posture classification,
temporal fall-detection state machine, patient presence monitoring, emergency event management,
and a caregiver REST / WebSocket notification API.
"""

from luminax_vision.activity_classifier import (
    POSTURE_FALL_CONFIRMED,
    POSTURE_FALL_SUSPECTED,
    POSTURE_LYING_DOWN,
    POSTURE_SITTING,
    POSTURE_STANDING,
    POSTURE_UNKNOWN,
    POSTURE_WALKING,
    ActivityClassifier,
    PostureResult,
)
from luminax_vision.camera import BaseCamera, Camera, MockCamera
from luminax_vision.config import Config, load_config
from luminax_vision.emergency_manager import (
    EVENT_CAMERA_ERROR,
    EVENT_CAMERA_RECOVERED,
    EVENT_FALL_CONFIRMED,
    EVENT_FALL_SUSPECTED,
    EVENT_PERSON_MISSING,
    EVENT_PERSON_REAPPEARED,
    EVENT_SYSTEM_ERROR,
    EmergencyEvent,
    EmergencyEventManager,
)
from luminax_vision.face_database import FaceDatabase
from luminax_vision.face_detector import FaceDetection, FaceDetector
from luminax_vision.face_recognizer import FaceRecognizer, RecognitionResult
from luminax_vision.fall_detector import FallDetector, FallState, FallStateResult
from luminax_vision.notification_api import (
    NotificationServerThread,
    TelemetryState,
    WebSocketBroadcaster,
    create_notification_app,
)
from luminax_vision.person_tracker import PersonTrack, PersonTracker, compute_iou
from luminax_vision.pose_estimator import (
    LandmarkPoint,
    MockPoseEstimator,
    PoseEstimator,
    PoseFeatures,
    PoseResult,
)
from luminax_vision.presence_monitor import PresenceMonitor, PresenceState, PresenceResult

__version__ = "0.2.0"
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
    "PoseEstimator",
    "MockPoseEstimator",
    "PoseResult",
    "PoseFeatures",
    "LandmarkPoint",
    "PersonTracker",
    "PersonTrack",
    "compute_iou",
    "ActivityClassifier",
    "PostureResult",
    "FallDetector",
    "FallState",
    "FallStateResult",
    "PresenceMonitor",
    "PresenceState",
    "PresenceResult",
    "EmergencyEventManager",
    "EmergencyEvent",
    "create_notification_app",
    "NotificationServerThread",
    "TelemetryState",
    "WebSocketBroadcaster",
    "__version__",
]
