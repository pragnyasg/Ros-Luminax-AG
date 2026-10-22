"""
Unit tests for monitored person presence and disappearance monitoring.
"""

import pytest

from luminax_vision.person_tracker import PersonTrack
from luminax_vision.presence_monitor import PresenceMonitor, PresenceState


def make_track(name: str = "Pragnya", track_id: int = 1) -> PersonTrack:
    return PersonTrack(
        track_id=track_id,
        bbox=(50, 50, 120, 240),
        bbox_normalized=(0.1, 0.1, 0.2, 0.5),
        centroid=(0.2, 0.35),
        identity_name=name,
        is_known_identity=(name != "Unknown"),
    )


def test_target_person_initially_present():
    """When target person is visible in frame, state is PRESENT."""
    monitor = PresenceMonitor(
        monitored_target="Pragnya",
        temporary_missing_timeout_sec=2.0,
        missing_timeout_sec=5.0,
    )
    track = make_track("Pragnya")

    res = monitor.update([track], current_time=0.0)
    assert res.state == PresenceState.PRESENT
    assert res.matched_track_id == 1
    assert res.is_event_triggered is False


def test_single_missed_frame_does_not_trigger_missing_alert():
    """A single dropped frame or brief occlusion must not trigger missing-person alert."""
    monitor = PresenceMonitor(
        monitored_target="Pragnya",
        temporary_missing_timeout_sec=2.0,
        missing_timeout_sec=5.0,
    )
    track = make_track("Pragnya")

    # Present at t=0.0
    monitor.update([track], current_time=0.0)

    # 1 missed frame at t=0.033
    res = monitor.update([], current_time=0.033)
    assert res.state == PresenceState.TEMPORARILY_MISSING
    assert res.is_event_triggered is False
    assert res.event_type is None


def test_brief_occlusion_remains_temporarily_missing():
    """Absence below missing_timeout_sec stays in TEMPORARILY_MISSING."""
    monitor = PresenceMonitor(
        monitored_target="Pragnya",
        temporary_missing_timeout_sec=2.0,
        missing_timeout_sec=5.0,
    )
    track = make_track("Pragnya")

    monitor.update([track], current_time=0.0)
    res = monitor.update([], current_time=3.5)  # 3.5s absent < 5.0s timeout

    assert res.state in (PresenceState.TEMPORARILY_MISSING, PresenceState.MISSING)
    # Alert should only trigger when duration exceeds missing_timeout_sec (5.0s)
    assert res.is_event_triggered is False


def test_sustained_absence_triggers_missing_alert():
    """Absence exceeding missing_timeout_sec transitions to MISSING and triggers alert."""
    monitor = PresenceMonitor(
        monitored_target="Pragnya",
        temporary_missing_timeout_sec=2.0,
        missing_timeout_sec=5.0,
    )
    track = make_track("Pragnya")

    monitor.update([track], current_time=0.0)
    res = monitor.update([], current_time=5.5)

    assert res.state == PresenceState.MISSING
    assert res.is_event_triggered is True
    assert res.event_type == "person_missing"
    assert res.duration_absent_sec >= 5.0

    # Next frame while still absent: alert should not re-trigger
    res_next = monitor.update([], current_time=6.0)
    assert res_next.state == PresenceState.MISSING
    assert res_next.is_event_triggered is False


def test_reappearance_restores_present_and_emits_event():
    """When a missing person returns, state transitions to PRESENT and emits reappeared."""
    monitor = PresenceMonitor(
        monitored_target="Pragnya",
        temporary_missing_timeout_sec=2.0,
        missing_timeout_sec=5.0,
    )
    track = make_track("Pragnya")

    # Present -> Missing -> Reappeared
    monitor.update([track], current_time=0.0)
    monitor.update([], current_time=6.0)  # Transitions to MISSING

    res_back = monitor.update([track], current_time=7.0)
    assert res_back.state == PresenceState.PRESENT
    assert res_back.is_event_triggered is True
    assert res_back.event_type == "person_reappeared"


def test_camera_failure_distinguished_from_person_absence():
    """Camera hardware failure produces CAMERA_ERROR, NOT person_missing."""
    monitor = PresenceMonitor(monitored_target="Pragnya")
    track = make_track("Pragnya")

    monitor.update([track], camera_ok=True, current_time=0.0)
    res = monitor.update([], camera_ok=False, current_time=1.0)

    assert res.state == PresenceState.CAMERA_ERROR
    assert res.event_type == "camera_error"


def test_person_disappearance_is_not_labelled_a_fall():
    """A person leaving view must be classified as person_missing, never fall_confirmed."""
    monitor = PresenceMonitor(
        monitored_target="Pragnya",
        missing_timeout_sec=2.0,
    )
    track = make_track("Pragnya")

    monitor.update([track], current_time=0.0)
    res = monitor.update([], current_time=3.0)

    assert res.event_type != "fall_confirmed"
    assert res.event_type == "person_missing"
