"""
Unit tests for Pose Estimation and Geometric Feature Calculation.
"""

import pytest
import numpy as np

from luminax_vision.pose_estimator import (
    BasePoseEstimator,
    LandmarkPoint,
    MockPoseEstimator,
    PoseFeatures,
    PoseResult,
    compute_angle_deg,
    compute_horizontal_angle_deg,
    extract_pose_features,
)


def test_angle_calculations():
    # Straight vertical down vector (0, 0) -> (0, 1) has angle 0 deg
    assert compute_angle_deg((0.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0, abs=1e-3)
    # Straight horizontal right vector (0, 0) -> (1, 0) has vertical angle 90 deg
    assert compute_angle_deg((0.0, 0.0), (1.0, 0.0)) == pytest.approx(90.0, abs=1e-3)
    # Straight vertical up vector (0, 0) -> (0, -1) has angle 180 deg
    assert compute_angle_deg((0.0, 0.0), (0.0, -1.0)) == pytest.approx(180.0, abs=1e-3)

    # Horizontal angle: 90 deg for vertical upright, 0 deg for horizontal
    assert compute_horizontal_angle_deg((0.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0, abs=1e-3)
    assert compute_horizontal_angle_deg((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0, abs=1e-3)


def test_mock_pose_estimator_standing():
    estimator = MockPoseEstimator(mode="standing")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = estimator.estimate(img)

    assert len(poses) == 1
    pose = poses[0]
    assert pose.features is not None
    assert pose.features.is_valid is True
    # In standing mode, torso should be upright: vertical angle <= 30 deg, horizontal angle >= 60 deg
    assert pose.features.torso_angle_deg < 30.0
    assert pose.features.torso_horizontal_angle_deg > 60.0
    assert pose.features.aspect_ratio > 1.2
    estimator.close()


def test_mock_pose_estimator_lying_down():
    estimator = MockPoseEstimator(mode="lying_down")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = estimator.estimate(img)

    assert len(poses) == 1
    pose = poses[0]
    assert pose.features is not None
    assert pose.features.is_valid is True
    # In lying down mode, torso should be horizontal: horizontal angle < 30 deg, aspect ratio < 1.0
    assert pose.features.torso_horizontal_angle_deg < 30.0
    assert pose.features.aspect_ratio < 1.0
    estimator.close()


def test_extract_pose_features_missing_landmarks():
    # Only nose and shoulders, hips missing
    partial_landmarks = {
        "nose": LandmarkPoint(0.5, 0.2),
        "left_shoulder": LandmarkPoint(0.45, 0.3),
        "right_shoulder": LandmarkPoint(0.55, 0.3),
    }
    features = extract_pose_features(partial_landmarks, image_width=640, image_height=480)
    assert features.is_valid is False
    assert "Missing essential torso" in features.diagnostic_notes


def test_extract_pose_features_low_confidence():
    # Essential landmarks present but visibility is 0.1 (below min_confidence=0.4)
    low_conf_landmarks = {
        "left_shoulder": LandmarkPoint(0.45, 0.3, visibility=0.1),
        "right_shoulder": LandmarkPoint(0.55, 0.3, visibility=0.1),
        "left_hip": LandmarkPoint(0.46, 0.6, visibility=0.1),
        "right_hip": LandmarkPoint(0.54, 0.6, visibility=0.1),
    }
    features = extract_pose_features(low_conf_landmarks, image_width=640, image_height=480, min_confidence=0.4)
    assert features.is_valid is False
    assert "Low torso landmark visibility" in features.diagnostic_notes
