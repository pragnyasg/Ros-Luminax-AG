"""
Pose estimation and geometric feature extraction for LUMI Vision.

Wraps MediaPipe PoseLandmarker (Tasks API) for high-precision, low-latency
body landmark extraction on CPU (XNNPACK) and provides geometric feature
computation for posture and fall classification.
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Key landmark indices matching standard MediaPipe Pose 33-point topology
LANDMARK_INDEX_MAP = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Anatomical skeleton connections for rendering
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]


@dataclass
class LandmarkPoint:
    """Single normalized 3D body landmark with visibility/confidence score."""
    x: float  # Normalized [0.0, 1.0], image width
    y: float  # Normalized [0.0, 1.0], image height (0 at top, 1 at bottom)
    z: float = 0.0  # Depth relative to hips
    visibility: float = 1.0  # Landmark visibility confidence [0.0, 1.0]
    presence: float = 1.0  # Landmark presence in frame [0.0, 1.0]

    @property
    def is_visible(self) -> bool:
        return self.visibility >= 0.5


@dataclass
class PoseFeatures:
    """
    Extracted geometric features from body landmarks.

    Coordinate system: Normalized image coordinates [0, 1].
    X increases to the right. Y increases downward.
    """
    torso_center: Tuple[float, float]
    shoulder_mid: Tuple[float, float]
    hip_mid: Tuple[float, float]
    knee_mid: Optional[Tuple[float, float]]
    ankle_mid: Optional[Tuple[float, float]]
    # Angle of torso relative to vertical axis (0° = upright, 90° = horizontal)
    torso_angle_deg: float
    # Angle of torso relative to horizontal axis (90° = upright, 0° = horizontal)
    torso_horizontal_angle_deg: float
    # Aspect ratio: height / width of body bounding box
    aspect_ratio: float
    # Bounding box in pixel coordinates (x, y, w, h)
    bbox: Tuple[int, int, int, int]
    # Normalized bounding box (nx, ny, nw, nh) in [0.0, 1.0]
    bbox_normalized: Tuple[float, float, float, float]
    # Torso length (distance between shoulder midpoint and hip midpoint)
    torso_length: float
    # Confidence metrics
    avg_visibility: float
    is_valid: bool
    diagnostic_notes: str = ""


@dataclass
class PoseResult:
    """Represents a detected person's full body pose in a frame."""
    landmarks: Dict[str, LandmarkPoint] = field(default_factory=dict)
    raw_landmarks: Optional[List[LandmarkPoint]] = None
    features: Optional[PoseFeatures] = None
    score: float = 0.0

    def get_landmark(self, name: str) -> Optional[LandmarkPoint]:
        return self.landmarks.get(name)


def compute_angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute angle in degrees of vector p1 -> p2 relative to vertical downward vector (0, 1).
    Returns value in [0, 180]. 0° = straight vertical down, 90° = horizontal, 180° = upside down.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    # Dot product with vertical unit vector (0, 1)
    cos_theta = dy / norm
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def compute_horizontal_angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute absolute angle in degrees of vector p1 -> p2 relative to the horizontal plane.
    Returns value in [0, 90]. 90° = vertical upright, 0° = horizontal.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 90.0
    sin_theta = abs(dy) / norm
    sin_theta = max(0.0, min(1.0, sin_theta))
    return math.degrees(math.asin(sin_theta))


def extract_pose_features(
    landmarks: Dict[str, LandmarkPoint],
    image_width: int,
    image_height: int,
    min_confidence: float = 0.4,
) -> PoseFeatures:
    """
    Calculate geometric posture metrics from normalized landmarks.

    Validates key landmarks (shoulders, hips) before computing posture features.
    If key landmarks are missing or below min_confidence, marks features as invalid.
    """
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")

    # Check essential landmark presence and confidence
    essential = [ls, rs, lh, rh]
    if any(p is None for p in essential):
        return PoseFeatures(
            torso_center=(0.0, 0.0),
            shoulder_mid=(0.0, 0.0),
            hip_mid=(0.0, 0.0),
            knee_mid=None,
            ankle_mid=None,
            torso_angle_deg=0.0,
            torso_horizontal_angle_deg=90.0,
            aspect_ratio=1.0,
            bbox=(0, 0, 0, 0),
            bbox_normalized=(0.0, 0.0, 0.0, 0.0),
            torso_length=0.0,
            avg_visibility=0.0,
            is_valid=False,
            diagnostic_notes="Missing essential torso landmarks (shoulders or hips)",
        )

    visibilities = [p.visibility for p in essential if p is not None]
    avg_vis = float(np.mean(visibilities))
    if avg_vis < min_confidence:
        return PoseFeatures(
            torso_center=(0.0, 0.0),
            shoulder_mid=(0.0, 0.0),
            hip_mid=(0.0, 0.0),
            knee_mid=None,
            ankle_mid=None,
            torso_angle_deg=0.0,
            torso_horizontal_angle_deg=90.0,
            aspect_ratio=1.0,
            bbox=(0, 0, 0, 0),
            bbox_normalized=(0.0, 0.0, 0.0, 0.0),
            torso_length=0.0,
            avg_visibility=avg_vis,
            is_valid=False,
            diagnostic_notes=f"Low torso landmark visibility: {avg_vis:.2f} < {min_confidence}",
        )

    # Compute midpoints
    shoulder_mid = ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)
    hip_mid = ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
    torso_center = ((shoulder_mid[0] + hip_mid[0]) / 2.0, (shoulder_mid[1] + hip_mid[1]) / 2.0)

    # Optional lower body midpoints
    lk = landmarks.get("left_knee")
    rk = landmarks.get("right_knee")
    knee_mid: Optional[Tuple[float, float]] = None
    if lk and rk and lk.visibility >= min_confidence and rk.visibility >= min_confidence:
        knee_mid = ((lk.x + rk.x) / 2.0, (lk.y + rk.y) / 2.0)

    la = landmarks.get("left_ankle")
    ra = landmarks.get("right_ankle")
    ankle_mid: Optional[Tuple[float, float]] = None
    if la and ra and la.visibility >= min_confidence and ra.visibility >= min_confidence:
        ankle_mid = ((la.x + ra.x) / 2.0, (la.y + ra.y) / 2.0)

    # Torso angles: vector from shoulder_mid to hip_mid
    # dy = hip.y - shoulder.y (normally positive when standing because Y increases downwards)
    torso_angle = compute_angle_deg(shoulder_mid, hip_mid)
    torso_h_angle = compute_horizontal_angle_deg(shoulder_mid, hip_mid)
    torso_len = math.hypot(hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])

    # Compute bounding box across all visible landmarks
    valid_pts = [(p.x, p.y) for p in landmarks.values() if p.visibility >= min_confidence]
    if not valid_pts:
        valid_pts = [(p.x, p.y) for p in essential]

    xs = [pt[0] for pt in valid_pts]
    ys = [pt[1] for pt in valid_pts]
    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))

    # Add small padding
    pad_x = 0.05 * (max_x - min_x + 1e-4)
    pad_y = 0.05 * (max_y - min_y + 1e-4)
    min_x = max(0.0, min_x - pad_x)
    max_x = min(1.0, max_x + pad_x)
    min_y = max(0.0, min_y - pad_y)
    max_y = min(1.0, max_y + pad_y)

    norm_w = max_x - min_x
    norm_h = max_y - min_y
    aspect_ratio = (norm_h / norm_w) if norm_w > 1e-4 else 1.0

    px_x = int(min_x * image_width)
    px_y = int(min_y * image_height)
    px_w = int(norm_w * image_width)
    px_h = int(norm_h * image_height)

    return PoseFeatures(
        torso_center=torso_center,
        shoulder_mid=shoulder_mid,
        hip_mid=hip_mid,
        knee_mid=knee_mid,
        ankle_mid=ankle_mid,
        torso_angle_deg=torso_angle,
        torso_horizontal_angle_deg=torso_h_angle,
        aspect_ratio=aspect_ratio,
        bbox=(px_x, px_y, px_w, px_h),
        bbox_normalized=(min_x, min_y, norm_w, norm_h),
        torso_length=torso_len,
        avg_visibility=avg_vis,
        is_valid=True,
        diagnostic_notes="Valid pose landmarks",
    )


class BasePoseEstimator(abc.ABC):
    """Abstract interface for human pose estimators."""

    @abc.abstractmethod
    def estimate(self, image: np.ndarray) -> List[PoseResult]:
        """Process BGR image and return list of detected poses."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Release underlying inference resources."""
        pass

    def __enter__(self) -> BasePoseEstimator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def draw_pose(
        self,
        image: np.ndarray,
        pose: PoseResult,
        color_skeleton: Tuple[int, int, int] = (0, 255, 255),
        color_points: Tuple[int, int, int] = (255, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw skeleton connections and landmark points on an image copy."""
        output = image.copy()
        h, w = output.shape[:2]

        if not pose.landmarks:
            return output

        # Draw skeleton connections
        for start_name, end_name in SKELETON_CONNECTIONS:
            p1 = pose.get_landmark(start_name)
            p2 = pose.get_landmark(end_name)
            if p1 and p2 and p1.is_visible and p2.is_visible:
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(output, pt1, pt2, color_skeleton, thickness, cv2.LINE_AA)

        # Draw landmark dots
        for name, pt in pose.landmarks.items():
            if pt.is_visible:
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(output, (cx, cy), 3, color_points, -1, cv2.LINE_AA)

        return output


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    Real-time multi-person / single-person human pose estimation using MediaPipe Tasks PoseLandmarker.
    Operates on standard BGR frames.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/pose_landmarker_lite.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 4,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe PoseLandmarker model not found at {self.model_path}. "
                "Run 'python scripts/download_models.py' to download it."
            )

        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.num_poses = int(num_poses)

        # Import mediapipe tasks
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=self.num_poses,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize MediaPipe PoseLandmarker: {exc}") from exc

        logger.info(
            f"MediaPipePoseEstimator initialized with model '{self.model_path.name}', "
            f"num_poses={self.num_poses}, det_conf={self.min_detection_confidence}"
        )

    def estimate(self, image: np.ndarray) -> List[PoseResult]:
        """
        Estimate body landmarks on a BGR image.

        Args:
            image: uint8 BGR image array with shape (H, W, 3).

        Returns:
            List of PoseResult instances sorted by confidence descending.
        """
        if image is None or image.size == 0 or len(image.shape) != 3:
            return []

        h, w = image.shape[:2]

        # MediaPipe expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_image)

        detection_result = self._landmarker.detect(mp_img)

        if not detection_result.pose_landmarks:
            return []

        results: List[PoseResult] = []

        # Iterate over detected individuals
        for pose_idx, raw_landmarks in enumerate(detection_result.pose_landmarks):
            landmark_dict: Dict[str, LandmarkPoint] = {}
            raw_list: List[LandmarkPoint] = []

            for name, idx in LANDMARK_INDEX_MAP.items():
                if idx < len(raw_landmarks):
                    lm = raw_landmarks[idx]
                    pt = LandmarkPoint(
                        x=float(np.clip(lm.x, 0.0, 1.0)),
                        y=float(np.clip(lm.y, 0.0, 1.0)),
                        z=float(lm.z) if hasattr(lm, "z") else 0.0,
                        visibility=float(lm.visibility) if hasattr(lm, "visibility") else 1.0,
                        presence=float(lm.presence) if hasattr(lm, "presence") else 1.0,
                    )
                    landmark_dict[name] = pt
                    raw_list.append(pt)

            features = extract_pose_features(
                landmark_dict,
                image_width=w,
                image_height=h,
                min_confidence=self.min_detection_confidence,
            )

            score = features.avg_visibility if features.is_valid else 0.0
            results.append(
                PoseResult(
                    landmarks=landmark_dict,
                    raw_landmarks=raw_list,
                    features=features,
                    score=score,
                )
            )

        # Sort poses by confidence descending
        results.sort(key=lambda p: p.score, reverse=True)
        return results

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


class MockPoseEstimator(BasePoseEstimator):
    """
    Synthetic mock pose estimator for deterministic automated tests and CI.
    Generates synthetic landmark configurations for standing, sitting, fallen, walking, etc.
    """

    def __init__(self, mode: str = "standing") -> None:
        self.mode = mode
        self.custom_poses: List[PoseResult] = []

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_custom_poses(self, poses: List[PoseResult]) -> None:
        self.custom_poses = poses

    def estimate(self, image: np.ndarray) -> List[PoseResult]:
        if self.custom_poses:
            return self.custom_poses

        h, w = (480, 640) if image is None or image.size == 0 else image.shape[:2]

        if self.mode == "none":
            return []

        landmarks: Dict[str, LandmarkPoint] = {}

        if self.mode == "standing":
            # Upright standing: head at y=0.15, shoulders at y=0.25, hips at y=0.55, knees at y=0.75, ankles at y=0.92
            landmarks["nose"] = LandmarkPoint(x=0.50, y=0.15, visibility=0.95)
            landmarks["left_shoulder"] = LandmarkPoint(x=0.45, y=0.25, visibility=0.95)
            landmarks["right_shoulder"] = LandmarkPoint(x=0.55, y=0.25, visibility=0.95)
            landmarks["left_hip"] = LandmarkPoint(x=0.46, y=0.55, visibility=0.95)
            landmarks["right_hip"] = LandmarkPoint(x=0.54, y=0.55, visibility=0.95)
            landmarks["left_knee"] = LandmarkPoint(x=0.47, y=0.75, visibility=0.95)
            landmarks["right_knee"] = LandmarkPoint(x=0.53, y=0.75, visibility=0.95)
            landmarks["left_ankle"] = LandmarkPoint(x=0.47, y=0.92, visibility=0.95)
            landmarks["right_ankle"] = LandmarkPoint(x=0.53, y=0.92, visibility=0.95)
            landmarks["left_elbow"] = LandmarkPoint(x=0.42, y=0.38, visibility=0.90)
            landmarks["right_elbow"] = LandmarkPoint(x=0.58, y=0.38, visibility=0.90)
            landmarks["left_wrist"] = LandmarkPoint(x=0.40, y=0.50, visibility=0.90)
            landmarks["right_wrist"] = LandmarkPoint(x=0.60, y=0.50, visibility=0.90)

        elif self.mode == "sitting":
            # Sitting upright on a chair: shoulders y=0.35, hips y=0.60, knees forward/bent y=0.62, ankles y=0.85
            landmarks["nose"] = LandmarkPoint(x=0.50, y=0.25, visibility=0.95)
            landmarks["left_shoulder"] = LandmarkPoint(x=0.45, y=0.35, visibility=0.95)
            landmarks["right_shoulder"] = LandmarkPoint(x=0.55, y=0.35, visibility=0.95)
            landmarks["left_hip"] = LandmarkPoint(x=0.46, y=0.60, visibility=0.95)
            landmarks["right_hip"] = LandmarkPoint(x=0.54, y=0.60, visibility=0.95)
            landmarks["left_knee"] = LandmarkPoint(x=0.43, y=0.64, visibility=0.95)
            landmarks["right_knee"] = LandmarkPoint(x=0.57, y=0.64, visibility=0.95)
            landmarks["left_ankle"] = LandmarkPoint(x=0.43, y=0.85, visibility=0.90)
            landmarks["right_ankle"] = LandmarkPoint(x=0.57, y=0.85, visibility=0.90)
            landmarks["left_elbow"] = LandmarkPoint(x=0.42, y=0.45, visibility=0.90)
            landmarks["right_elbow"] = LandmarkPoint(x=0.58, y=0.45, visibility=0.90)
            landmarks["left_wrist"] = LandmarkPoint(x=0.42, y=0.58, visibility=0.90)
            landmarks["right_wrist"] = LandmarkPoint(x=0.58, y=0.58, visibility=0.90)

        elif self.mode in ("fallen", "lying_down"):
            # Lying horizontally on the ground: y coordinates all around y=0.80 - 0.88, x spread out horizontally
            landmarks["nose"] = LandmarkPoint(x=0.20, y=0.82, visibility=0.95)
            landmarks["left_shoulder"] = LandmarkPoint(x=0.32, y=0.80, visibility=0.95)
            landmarks["right_shoulder"] = LandmarkPoint(x=0.32, y=0.86, visibility=0.95)
            landmarks["left_hip"] = LandmarkPoint(x=0.55, y=0.81, visibility=0.95)
            landmarks["right_hip"] = LandmarkPoint(x=0.55, y=0.87, visibility=0.95)
            landmarks["left_knee"] = LandmarkPoint(x=0.70, y=0.82, visibility=0.95)
            landmarks["right_knee"] = LandmarkPoint(x=0.70, y=0.88, visibility=0.95)
            landmarks["left_ankle"] = LandmarkPoint(x=0.85, y=0.83, visibility=0.95)
            landmarks["right_ankle"] = LandmarkPoint(x=0.85, y=0.89, visibility=0.95)
            landmarks["left_elbow"] = LandmarkPoint(x=0.25, y=0.78, visibility=0.90)
            landmarks["right_elbow"] = LandmarkPoint(x=0.25, y=0.88, visibility=0.90)
            landmarks["left_wrist"] = LandmarkPoint(x=0.18, y=0.79, visibility=0.90)
            landmarks["right_wrist"] = LandmarkPoint(x=0.18, y=0.89, visibility=0.90)

        elif self.mode == "invalid_confidence":
            # Landmarks with very low visibility
            landmarks["nose"] = LandmarkPoint(x=0.50, y=0.15, visibility=0.1)
            landmarks["left_shoulder"] = LandmarkPoint(x=0.45, y=0.25, visibility=0.1)
            landmarks["right_shoulder"] = LandmarkPoint(x=0.55, y=0.25, visibility=0.1)
            landmarks["left_hip"] = LandmarkPoint(x=0.46, y=0.55, visibility=0.1)
            landmarks["right_hip"] = LandmarkPoint(x=0.54, y=0.55, visibility=0.1)

        features = extract_pose_features(landmarks, image_width=w, image_height=h)
        return [PoseResult(landmarks=landmarks, features=features, score=0.9)]

    def close(self) -> None:
        pass


# Standard alias
PoseEstimator = MediaPipePoseEstimator
