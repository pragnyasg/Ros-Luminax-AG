"""
Unit tests for event manager and SQLite persistence.
"""

from pathlib import Path
import numpy as np
import pytest

from luminax_vision.event_manager import (
    EVENT_CAMERA_ERROR,
    EVENT_FALL_CONFIRMED,
    EVENT_PERSON_MISSING,
    EventManager,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
)


def test_create_and_retrieve_event(tmp_path: Path):
    """Verify event creation, SQLite persistence, and retrieval by event_id."""
    db_file = tmp_path / "events.db"
    snaps_dir = tmp_path / "snapshots"
    em = EventManager(db_path=db_file, snapshots_dir=snaps_dir)

    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    ev = em.create_event(
        event_type=EVENT_FALL_CONFIRMED,
        severity=SEVERITY_CRITICAL,
        person_name="Pragnya",
        track_id=2,
        confidence=0.92,
        posture="lying_down",
        bbox=[10, 20, 100, 200],
        frame=frame,
        details={"torso_angle": 88.5, "peak_velocity": 0.82},
    )

    assert ev.event_id is not None
    assert len(ev.event_id) == 36  # UUID4 string
    assert ev.timestamp.endswith("+00:00") or "T" in ev.timestamp
    assert ev.acknowledged is False
    assert ev.snapshot_path is not None
    assert (snaps_dir / f"{ev.event_id}.jpg").is_file()

    # Retrieve from DB
    fetched = em.get_event(ev.event_id)
    assert fetched is not None
    assert fetched.event_id == ev.event_id
    assert fetched.person_name == "Pragnya"
    assert fetched.confidence == 0.92
    assert fetched.bbox == [10, 20, 100, 200]
    assert fetched.details["torso_angle"] == 88.5
    # Robot location must be None when no sensor is attached
    assert fetched.robot_location is None


def test_list_events_with_filters(tmp_path: Path):
    """Verify event listing with type and unacknowledged filtering."""
    db_file = tmp_path / "events.db"
    em = EventManager(db_path=db_file, save_snapshots=False)

    em.create_event(event_type=EVENT_FALL_CONFIRMED, severity=SEVERITY_CRITICAL)
    ev_miss = em.create_event(event_type=EVENT_PERSON_MISSING, severity=SEVERITY_WARNING)
    em.create_event(event_type=EVENT_CAMERA_ERROR, severity=SEVERITY_WARNING)

    # Filter by type
    fall_events = em.list_events(event_type=EVENT_FALL_CONFIRMED)
    assert len(fall_events) == 1
    assert fall_events[0].event_type == EVENT_FALL_CONFIRMED

    # Acknowledge one event
    em.acknowledge_event(ev_miss.event_id, acknowledged_by="Nurse Joy")
    unack_events = em.list_events(unacknowledged_only=True)
    assert len(unack_events) == 2
    assert ev_miss.event_id not in [e.event_id for e in unack_events]


def test_event_acknowledgement(tmp_path: Path):
    """Verify acknowledging an event updates status and records timestamp."""
    db_file = tmp_path / "events.db"
    em = EventManager(db_path=db_file, save_snapshots=False)

    ev = em.create_event(event_type=EVENT_FALL_CONFIRMED)
    assert ev.acknowledged is False

    ok = em.acknowledge_event(ev.event_id, acknowledged_by="Caregiver_Unit_4")
    assert ok is True

    updated = em.get_event(ev.event_id)
    assert updated.acknowledged is True
    assert updated.acknowledged_by == "Caregiver_Unit_4"
    assert updated.acknowledged_at is not None

    # Acknowledging non-existent event returns False
    assert em.acknowledge_event("non-existent-id") is False


def test_snapshot_failure_handled_gracefully(tmp_path: Path):
    """Corrupt or None frame must safely result in snapshot_path=None without error."""
    db_file = tmp_path / "events.db"
    snaps_dir = tmp_path / "snapshots"
    em = EventManager(db_path=db_file, snapshots_dir=snaps_dir)

    ev = em.create_event(
        event_type=EVENT_FALL_CONFIRMED,
        frame=None,  # No frame available
    )
    assert ev.snapshot_path is None

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    ev2 = em.create_event(
        event_type=EVENT_FALL_CONFIRMED,
        frame=empty_frame,
    )
    assert ev2.snapshot_path is None
