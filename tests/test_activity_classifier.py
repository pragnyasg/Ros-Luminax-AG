"""
Unit tests for Activity and Posture Classification.
"""

import pytest
import numpy as np

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
from luminax_vision.person_tracker import PersonTrack
from luminax_vision.pose_estimator import LandmarkPoint, MockPoseEstimator, PoseFeatures, PoseResult


def make_track(pose: PoseResult, vy: float = 0.0, vx: float = 0.0) -> PersonTrack:
    track = PersonTrack(
        track_id=1,
        bbox=(100, 100, 200, 350),
        bbox_normalized=(0.2, 0.2, 0.3, 0.6),
        centroid=(0.35, 0.5),
        pose=pose,
    )
    track.velocity_y = vy
    track.velocity_x = vx
    return track


def test_classify_standing():
    classifier = ActivityClassifier()
    estimator = MockPoseEstimator(mode="standing")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = estimator.estimate(img)
    track = make_track(poses[0], vy=0.01)

    result = classifier.classify(track)
    assert result.posture == POSTURE_STANDING
    assert result.is_upright is True
    assert result.is_horizontal is False
    assert result.confidence > 0.6


def test_classify_lying_down():
    classifier = ActivityClassifier()
    estimator = MockPoseEstimator(mode="lying_down")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = estimator.estimate(img)
    track = make_track(poses[0], vy=0.0)

    result = classifier.classify(track)
    assert result.posture == POSTURE_LYING_DOWN
    assert result.is_horizontal is True
    assert result.is_upright is False


def test_classify_rapid_descent_triggers_fall_suspected():
    classifier = ActivityClassifier(rapid_descent_velocity=0.45)
    estimator = MockPoseEstimator(mode="fallen")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = estimator.estimate(img)
    # Simulate high downward descent velocity
    track = make_track(poses[0], vy=0.75)

    result = classifier.classify(track)
    assert result.posture == POSTURE_FALL_SUSPECTED
    assert result.confidence >= 0.70


def test_classify_invalid_pose_returns_unknown():
    classifier = ActivityClassifier()
    empty_pose = PoseResult()  # No features
    track = make_track(empty_pose)

    result = classifier.classify(track)
    assert result.posture == POSTURE_UNKNOWN
    assert result.confidence == 0.0
