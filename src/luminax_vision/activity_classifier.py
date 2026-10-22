"""
Human activity and posture classification module for LUMI Vision.

Classifies body postures (standing, sitting, walking, lying_down, fall_suspected,
fall_confirmed, unknown) using geometric pose landmarks, body orientation angles,
aspect ratio, and temporal movement dynamics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from luminax_vision.person_tracker import PersonTrack
from luminax_vision.pose_estimator import PoseFeatures, PoseResult

logger = logging.getLogger(__name__)

# Valid posture labels
POSTURE_STANDING = "standing"
POSTURE_SITTING = "sitting"
POSTURE_WALKING = "walking"
POSTURE_LYING_DOWN = "lying_down"
POSTURE_FALL_SUSPECTED = "fall_suspected"
POSTURE_FALL_CONFIRMED = "fall_confirmed"
POSTURE_UNKNOWN = "unknown"

ALL_POSTURES = {
    POSTURE_STANDING,
    POSTURE_SITTING,
    POSTURE_WALKING,
    POSTURE_LYING_DOWN,
    POSTURE_FALL_SUSPECTED,
    POSTURE_FALL_CONFIRMED,
    POSTURE_UNKNOWN,
}


@dataclass
class PostureResult:
    """Output of activity and posture classification for a single tracked person."""
    posture: str
    confidence: float
    torso_angle_deg: float
    torso_horizontal_deg: float
    aspect_ratio: float
    velocity_y: float
    velocity_x: float
    is_upright: bool
    is_horizontal: bool
    diagnostic_details: Dict[str, Any] = field(default_factory=dict)


class ActivityClassifier:
    """
    Rule-based geometric and kinematic posture classifier.

    Features:
    - Analyzes torso angle, aspect ratio (H/W), hip-to-knee distances, and velocity.
    - Differentiates intentional sitting or lying down from sudden uncontrolled falls.
    - Rejects low-confidence landmarks to prevent false positive classifications.
    """

    def __init__(
        self,
        upright_angle_threshold: float = 35.0,  # Max deviation from vertical for standing/sitting
        horizontal_angle_threshold: float = 40.0,  # Max angle above horizontal for lying down
        upright_aspect_ratio: float = 1.35,  # H/W threshold above which body is upright
        horizontal_aspect_ratio: float = 0.85,  # H/W threshold below which body is horizontal
        walking_speed_threshold: float = 0.15,  # Normalized units per second
        rapid_descent_velocity: float = 0.60,  # Normalized Y downward speed triggering fall suspicion
        min_landmark_confidence: float = 0.40,
    ) -> None:
        self.upright_angle_threshold = float(upright_angle_threshold)
        self.horizontal_angle_threshold = float(horizontal_angle_threshold)
        self.upright_aspect_ratio = float(upright_aspect_ratio)
        self.horizontal_aspect_ratio = float(horizontal_aspect_ratio)
        self.walking_speed_threshold = float(walking_speed_threshold)
        self.rapid_descent_velocity = float(rapid_descent_velocity)
        self.min_landmark_confidence = float(min_landmark_confidence)

    def classify(self, track: PersonTrack) -> PostureResult:
        """
        Classify posture and activity of a tracked individual.

        Args:
            track: PersonTrack with current pose, velocity, and history.

        Returns:
            PostureResult with classification label, confidence, and diagnostic metrics.
        """
        pose = track.pose
        if pose is None or pose.features is None or not pose.features.is_valid:
            return PostureResult(
                posture=POSTURE_UNKNOWN,
                confidence=0.0,
                torso_angle_deg=0.0,
                torso_horizontal_deg=90.0,
                aspect_ratio=1.0,
                velocity_y=track.velocity_y,
                velocity_x=track.velocity_x,
                is_upright=False,
                is_horizontal=False,
                diagnostic_details={"reason": "Invalid or missing pose features"},
            )

        features: PoseFeatures = pose.features
        torso_angle = features.torso_angle_deg
        torso_h_angle = features.torso_horizontal_angle_deg
        ar = features.aspect_ratio
        vy = track.velocity_y
        vx = track.velocity_x
        speed_h = abs(vx)

        # 1. Orientation flags
        # Upright: torso angle is close to vertical (0 deg deviation) and horizontal angle > 50 deg
        is_upright = (torso_angle <= self.upright_angle_threshold) and (torso_h_angle >= 50.0)
        # Horizontal: torso angle relative to horizontal is small (< threshold, e.g. < 40 deg)
        is_horizontal = (torso_h_angle <= self.horizontal_angle_threshold) or (ar <= self.horizontal_aspect_ratio)

        details = {
            "torso_angle_deg": round(torso_angle, 2),
            "torso_h_angle_deg": round(torso_h_angle, 2),
            "aspect_ratio": round(ar, 2),
            "velocity_y": round(vy, 3),
            "velocity_x": round(vx, 3),
            "avg_visibility": round(features.avg_visibility, 2),
        }

        # 2. Check for rapid descent indicating sudden drop / fall suspected
        # High downward vertical velocity (vy > rapid_descent_velocity)
        if vy > self.rapid_descent_velocity and not is_upright:
            return PostureResult(
                posture=POSTURE_FALL_SUSPECTED,
                confidence=min(1.0, 0.70 + (vy / 2.0)),
                torso_angle_deg=torso_angle,
                torso_horizontal_deg=torso_h_angle,
                aspect_ratio=ar,
                velocity_y=vy,
                velocity_x=vx,
                is_upright=is_upright,
                is_horizontal=is_horizontal,
                diagnostic_details={**details, "trigger": "Rapid downward velocity into non-upright configuration"},
            )

        # 3. Horizontal body orientation: Lying down
        if is_horizontal:
            conf = min(1.0, 0.75 + (1.0 - (torso_h_angle / 90.0)) * 0.25)
            return PostureResult(
                posture=POSTURE_LYING_DOWN,
                confidence=conf,
                torso_angle_deg=torso_angle,
                torso_horizontal_deg=torso_h_angle,
                aspect_ratio=ar,
                velocity_y=vy,
                velocity_x=vx,
                is_upright=False,
                is_horizontal=True,
                diagnostic_details={**details, "trigger": "Horizontal torso and low aspect ratio"},
            )

        # 4. Upright body orientation: Walking, Standing, or Sitting
        if is_upright or ar >= 1.1:
            # Check knee and ankle geometry for Sitting vs Standing
            hip_mid = features.hip_mid
            knee_mid = features.knee_mid
            ankle_mid = features.ankle_mid

            is_sitting = False
            if knee_mid is not None and hip_mid is not None:
                # Vertical distance between hips and knees relative to torso length
                hip_knee_dy = knee_mid[1] - hip_mid[1]
                thigh_ratio = hip_knee_dy / max(1e-4, features.torso_length)
                # When standing, thigh_ratio is typically >= 0.55.
                # When sitting, thighs are foreshortened/horizontal, thigh_ratio < 0.45 or ar < 1.30.
                if thigh_ratio < 0.45 or (hip_knee_dy < 0.15 and ar < 1.5):
                    is_sitting = True

            if is_sitting:
                return PostureResult(
                    posture=POSTURE_SITTING,
                    confidence=0.85,
                    torso_angle_deg=torso_angle,
                    torso_horizontal_deg=torso_h_angle,
                    aspect_ratio=ar,
                    velocity_y=vy,
                    velocity_x=vx,
                    is_upright=True,
                    is_horizontal=False,
                    diagnostic_details={**details, "trigger": "Upright torso with compressed hip-knee distance"},
                )

            # Check walking vs standing
            if speed_h > self.walking_speed_threshold:
                return PostureResult(
                    posture=POSTURE_WALKING,
                    confidence=0.80,
                    torso_angle_deg=torso_angle,
                    torso_horizontal_deg=torso_h_angle,
                    aspect_ratio=ar,
                    velocity_y=vy,
                    velocity_x=vx,
                    is_upright=True,
                    is_horizontal=False,
                    diagnostic_details={**details, "trigger": "Upright torso with sustained horizontal movement"},
                )

            # Default upright posture is Standing
            return PostureResult(
                posture=POSTURE_STANDING,
                confidence=0.90,
                torso_angle_deg=torso_angle,
                torso_horizontal_deg=torso_h_angle,
                aspect_ratio=ar,
                velocity_y=vy,
                velocity_x=vx,
                is_upright=True,
                is_horizontal=False,
                diagnostic_details={**details, "trigger": "Upright vertical torso and high aspect ratio"},
            )

        # 5. Ambiguous geometry: Unknown
        return PostureResult(
            posture=POSTURE_UNKNOWN,
            confidence=0.50,
            torso_angle_deg=torso_angle,
            torso_horizontal_deg=torso_h_angle,
            aspect_ratio=ar,
            velocity_y=vy,
            velocity_x=vx,
            is_upright=False,
            is_horizontal=False,
            diagnostic_details={**details, "trigger": "Ambiguous torso geometry and aspect ratio"},
        )
