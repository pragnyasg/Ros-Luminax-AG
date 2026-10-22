"""
Unit tests for person tracker and face-to-track association.
"""

import numpy as np
import pytest

from luminax_vision.face_detector import FaceDetection
from luminax_vision.face_recognizer import RecognitionResult
from luminax_vision.person_tracker import PersonTracker, compute_iou
from luminax_vision.pose_estimator import MockPoseEstimator


def test_compute_iou():
    """Verify IoU calculation on overlapping and disjoint bounding boxes."""
    # Exact overlap
    box_a = (10, 10, 50, 50)
    assert pytest.approx(compute_iou(box_a, box_a), abs=1e-3) == 1.0

    # Disjoint boxes
    box_b = (100, 100, 50, 50)
    assert compute_iou(box_a, box_b) == 0.0

    # Half overlap: (0,0,10,10) and (5,0,10,10)
    box_c = (0, 0, 10, 10)
    box_d = (5, 0, 10, 10)
    # Inter: 5 * 10 = 50. Union: 100 + 100 - 50 = 150. IoU = 50 / 150 = 1/3
    assert pytest.approx(compute_iou(box_c, box_d), abs=1e-3) == 1.0 / 3.0


def test_persistent_track_id_across_frames():
    """A person present in consecutive frames maintains the exact same track_id."""
    tracker = PersonTracker()
    mock = MockPoseEstimator("standing")
    poses = mock.estimate(np.zeros((480, 640, 3), dtype=np.uint8))

    # Frame 1
    tracks_f1 = tracker.update(poses, (480, 640), current_time=0.0)
    assert len(tracks_f1) == 1
    initial_id = tracks_f1[0].track_id

    # Frame 2 with same pose
    tracks_f2 = tracker.update(poses, (480, 640), current_time=0.033)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == initial_id
    assert tracks_f2[0].consecutive_frames_seen == 2


def test_face_association_with_tracked_person():
    """Face detected within the person's upper body correctly associates identity."""
    tracker = PersonTracker()
    mock = MockPoseEstimator("standing")
    poses = mock.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
    tracks = tracker.update(poses, (480, 640), current_time=0.0)
    assert tracks[0].identity_name == "Unknown"

    person_bbox = tracks[0].bbox  # (x, y, w, h)
    # Place face near head (top center of person bbox)
    fx = person_bbox[0] + int(person_bbox[2] * 0.25)
    fy = person_bbox[1] + 5
    fw = int(person_bbox[2] * 0.5)
    fh = int(person_bbox[3] * 0.2)

    fake_det = FaceDetection(
        bbox=(fx, fy, fw, fh),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        score=0.95,
        raw_face=np.zeros(15, dtype=np.float32),
    )
    fake_rec = RecognitionResult(
        name="Pragnya",
        score=0.88,
        is_known=True,
        metric="cosine",
        threshold=0.363,
    )

    tracker.associate_faces([(fake_det, fake_rec)], current_time=0.0)
    assert tracks[0].identity_name == "Pragnya"
    assert tracks[0].is_known_identity is True
    assert tracks[0].identity_score == 0.88


def test_unidentified_person_remains_unknown():
    """Unrecognized face preserves track as Unknown without inventing an identity."""
    tracker = PersonTracker()
    mock = MockPoseEstimator("standing")
    poses = mock.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
    tracks = tracker.update(poses, (480, 640), current_time=0.0)

    person_bbox = tracks[0].bbox
    fx = person_bbox[0] + 10
    fy = person_bbox[1] + 10

    fake_det = FaceDetection(
        bbox=(fx, fy, 40, 40),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        score=0.90,
        raw_face=np.zeros(15, dtype=np.float32),
    )
    unknown_rec = RecognitionResult(
        name="Unknown",
        score=0.15,
        is_known=False,
        metric="cosine",
        threshold=0.363,
    )

    tracker.associate_faces([(fake_det, unknown_rec)], current_time=0.0)
    assert tracks[0].identity_name == "Unknown"
    assert tracks[0].is_known_identity is False


def test_tracker_velocity_calculation():
    """Verify velocity_y correctly reflects downward vertical displacement."""
    tracker = PersonTracker()
    mock = MockPoseEstimator("standing")
    poses = mock.estimate(np.zeros((480, 640, 3), dtype=np.uint8))

    # Frame 1 at t=10.0
    tracks = tracker.update(poses, (480, 640), current_time=10.0)
    assert len(tracks) == 1

    # Frame 2 at t=10.2: person moved downward
    # Modify bounding box downward in normalized and pixel coords
    p = poses[0]
    orig_norm = p.features.bbox_normalized
    orig_px = p.features.bbox
    p.features.bbox_normalized = (orig_norm[0], orig_norm[1] + 0.1, orig_norm[2], orig_norm[3])
    p.features.bbox = (orig_px[0], orig_px[1] + 48, orig_px[2], orig_px[3])

    tracks2 = tracker.update(poses, (480, 640), current_time=10.2)
    assert len(tracks2) == 1
    assert tracks2[0].velocity_y > 0.0


def test_tracker_disappearance_and_pruning():
    """Tracks not detected for > max_disappeared frames are deregistered."""
    tracker = PersonTracker(max_disappeared=3)
    mock = MockPoseEstimator("standing")
    poses = mock.estimate(np.zeros((480, 640, 3), dtype=np.uint8))

    tracks = tracker.update(poses, (480, 640), current_time=1.0)
    assert len(tracks) == 1
    assert len(tracker.tracks) == 1

    # 3 consecutive misses
    tracker.update([], (480, 640), current_time=1.1)
    tracker.update([], (480, 640), current_time=1.2)
    tracker.update([], (480, 640), current_time=1.3)
    assert len(tracker.tracks) == 1

    # 4th miss exceeds max_disappeared=3
    tracker.update([], (480, 640), current_time=1.4)
    assert len(tracker.tracks) == 0

