"""
Unit tests for the temporal fall-detection state machine.
"""

import pytest

from luminax_vision.activity_classifier import (
    POSTURE_FALL_CONFIRMED,
    POSTURE_FALL_SUSPECTED,
    POSTURE_LYING_DOWN,
    POSTURE_STANDING,
    POSTURE_UNKNOWN,
    PostureResult,
)
from luminax_vision.fall_detector import FallDetector, FallState
from luminax_vision.person_tracker import PersonTrack


def make_track(track_id: int = 1, vy: float = 0.0) -> PersonTrack:
    return PersonTrack(
        track_id=track_id,
        bbox=(100, 100, 150, 200),
        bbox_normalized=(0.2, 0.2, 0.2, 0.4),
        centroid=(0.3, 0.4),
        velocity_y=vy,
    )


def make_standing_posture() -> PostureResult:
    return PostureResult(
        posture=POSTURE_STANDING,
        confidence=0.90,
        torso_angle_deg=5.0,
        torso_horizontal_deg=85.0,
        aspect_ratio=2.0,
        velocity_y=0.0,
        velocity_x=0.0,
        is_upright=True,
        is_horizontal=False,
    )


def make_horizontal_posture(vy: float = 0.0) -> PostureResult:
    return PostureResult(
        posture=POSTURE_LYING_DOWN,
        confidence=0.95,
        torso_angle_deg=85.0,
        torso_horizontal_deg=5.0,
        aspect_ratio=0.6,
        velocity_y=vy,
        velocity_x=0.0,
        is_upright=False,
        is_horizontal=True,
    )


def test_rapid_drop_with_persistent_horizontal_confirms_fall():
    """Rapid descent followed by horizontal persistence confirms fall event."""
    detector = FallDetector(
        persistence_duration_sec=2.0,
        rapid_descent_velocity=0.50,
    )
    track = make_track(1)

    # 1. Normal upright standing
    res = detector.update(track, make_standing_posture(), current_time=0.0)
    assert res.state == FallState.NORMAL
    assert res.is_emergency is False

    # 2. Sudden downward drop (vy=0.85)
    drop_posture = make_horizontal_posture(vy=0.85)
    track.velocity_y = 0.85
    res = detector.update(track, drop_posture, current_time=0.2)
    assert res.state in (FallState.POSSIBLE_FALL, FallState.FALL_SUSPECTED)
    assert res.is_emergency is False

    # 3. Transitions to horizontal
    track.velocity_y = 0.0
    res = detector.update(track, make_horizontal_posture(), current_time=0.5)
    assert res.state == FallState.FALL_SUSPECTED
    assert res.is_emergency is False

    # 4. Remains horizontal for 1.0 second (less than 2.0s persistence)
    res = detector.update(track, make_horizontal_posture(), current_time=1.5)
    assert res.state == FallState.FALL_SUSPECTED
    assert res.is_emergency is False

    # 5. Remains horizontal past 2.0s persistence (at t=2.6s)
    res = detector.update(track, make_horizontal_posture(), current_time=2.6)
    assert res.state == FallState.FALL_CONFIRMED
    assert res.is_emergency is True


def test_single_low_frame_does_not_confirm_fall():
    """A single frame with horizontal posture without duration must not confirm fall."""
    detector = FallDetector(persistence_duration_sec=2.0)
    track = make_track(1, vy=0.9)

    # Initial frame
    detector.update(track, make_standing_posture(), current_time=0.0)
    # One fast drop frame
    res = detector.update(track, make_horizontal_posture(vy=0.9), current_time=0.1)

    assert res.state != FallState.FALL_CONFIRMED
    assert res.is_emergency is False


def test_brief_downward_movement_that_recovers_upright_cancels_suspicion():
    """Stumbling down but quickly standing up cancels fall suspicion."""
    detector = FallDetector(persistence_duration_sec=2.0)
    track = make_track(1)

    # Normal
    detector.update(track, make_standing_posture(), current_time=0.0)
    # Sudden drop
    track.velocity_y = 0.8
    detector.update(track, make_horizontal_posture(vy=0.8), current_time=0.2)

    # Immediately stands back upright at t=0.5s
    track.velocity_y = 0.0
    res = detector.update(track, make_standing_posture(), current_time=0.5)

    assert res.state == FallState.NORMAL
    assert res.is_emergency is False


def test_repeated_frames_do_not_create_duplicate_alerts():
    """Once an emergency alert is emitted, subsequent frames during same fall do not re-emit."""
    detector = FallDetector(persistence_duration_sec=1.0)
    track = make_track(1)

    # Trigger fall
    track.velocity_y = 0.8
    detector.update(track, make_horizontal_posture(0.8), current_time=0.0)
    detector.update(track, make_horizontal_posture(), current_time=0.2)
    res_confirmed = detector.update(track, make_horizontal_posture(), current_time=1.3)

    assert res_confirmed.state == FallState.FALL_CONFIRMED
    assert res_confirmed.is_emergency is True

    # Next consecutive frame while still down
    res_next = detector.update(track, make_horizontal_posture(), current_time=1.4)
    assert res_next.state == FallState.ALERT_SENT
    assert res_next.is_emergency is False  # Deduplicated!

    # Another frame 2 seconds later still down
    res_later = detector.update(track, make_horizontal_posture(), current_time=3.5)
    assert res_later.state == FallState.ALERT_SENT
    assert res_later.is_emergency is False


def test_recovery_allows_future_independent_incident():
    """After recovering from a fall, a subsequent fall is detected independently."""
    detector = FallDetector(
        persistence_duration_sec=1.0,
        recovery_upright_duration_sec=1.0,
        cooldown_duration_sec=5.0,
    )
    track = make_track(1)

    # 1. First fall
    track.velocity_y = 0.8
    detector.update(track, make_horizontal_posture(0.8), current_time=0.0)
    detector.update(track, make_horizontal_posture(), current_time=0.2)
    res1 = detector.update(track, make_horizontal_posture(), current_time=1.3)
    assert res1.is_emergency is True

    # 2. Person stands up
    detector.update(track, make_standing_posture(), current_time=2.0)
    res_rec = detector.update(track, make_standing_posture(), current_time=3.2)
    assert res_rec.state in (FallState.RECOVERED, FallState.NORMAL)

    # 3. Cooldown expires and person is normal at t=7.0s
    res_norm = detector.update(track, make_standing_posture(), current_time=7.0)
    assert res_norm.state == FallState.NORMAL

    # 4. Second independent fall at t=8.0s
    track.velocity_y = 0.9
    detector.update(track, make_horizontal_posture(0.9), current_time=8.0)
    detector.update(track, make_horizontal_posture(), current_time=8.2)
    res2 = detector.update(track, make_horizontal_posture(), current_time=9.3)
    assert res2.state == FallState.FALL_CONFIRMED
    assert res2.is_emergency is True


def test_missing_pose_data_does_not_mean_fall_occurred():
    """Missing or low-confidence pose should not trigger fall confirmed."""
    detector = FallDetector(persistence_duration_sec=1.0)
    track = make_track(1)

    unknown_posture = PostureResult(
        posture=POSTURE_UNKNOWN,
        confidence=0.0,
        torso_angle_deg=0.0,
        torso_horizontal_deg=90.0,
        aspect_ratio=1.0,
        velocity_y=0.0,
        velocity_x=0.0,
        is_upright=False,
        is_horizontal=False,
    )

    for step in range(5):
        res = detector.update(track, unknown_posture, current_time=float(step))
        assert res.state == FallState.NORMAL
        assert res.is_emergency is False
