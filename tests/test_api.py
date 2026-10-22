"""
Unit and integration tests for FastAPI Caregiver API and WebSockets.
"""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from luminax_vision.api import create_app
from luminax_vision.event_manager import EVENT_FALL_CONFIRMED, EventManager, SEVERITY_CRITICAL


def test_api_health_endpoint(tmp_path: Path):
    """Verify GET /health returns operational subsystem status."""
    em = EventManager(db_path=tmp_path / "events.db")
    app = create_app(event_manager=em)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "database" in data["subsystems"]
    assert "camera" in data["subsystems"]


def test_api_status_endpoint(tmp_path: Path):
    """Verify GET /status retrieves live pipeline telemetry."""
    em = EventManager(db_path=tmp_path / "events.db")
    mock_status = {
        "monitoring_active": True,
        "presence_state": "PRESENT",
        "target_person": "Pragnya",
        "current_posture": "standing",
        "camera_fps": 28.5,
    }
    app = create_app(event_manager=em, status_provider=lambda: mock_status)
    client = TestClient(app)

    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["presence_state"] == "PRESENT"
    assert data["target_person"] == "Pragnya"
    assert data["current_posture"] == "standing"
    assert data["camera_fps"] == 28.5


def test_api_events_crud_and_ack(tmp_path: Path):
    """Verify listing, fetching, and acknowledging emergency events via REST."""
    em = EventManager(db_path=tmp_path / "events.db")
    ev = em.create_event(
        event_type=EVENT_FALL_CONFIRMED,
        severity=SEVERITY_CRITICAL,
        person_name="Pragnya",
    )

    app = create_app(event_manager=em)
    client = TestClient(app)

    # 1. List events
    res = client.get("/events")
    assert res.status_code == 200
    assert len(res.json()) == 1

    # 2. Get specific event
    res_single = client.get(f"/events/{ev.event_id}")
    assert res_single.status_code == 200
    assert res_single.json()["event_id"] == ev.event_id

    # 3. Get non-existent event returns 404
    res_404 = client.get("/events/non-existent-id")
    assert res_404.status_code == 404

    # 4. Acknowledge event
    ack_res = client.post(
        f"/events/{ev.event_id}/acknowledge",
        json={"acknowledged_by": "Nurse Ratched", "note": "Assisted patient safely"},
    )
    assert ack_res.status_code == 200
    assert ack_res.json()["success"] is True

    # 5. Verify updated in DB
    updated = em.get_event(ev.event_id)
    assert updated.acknowledged is True
    assert updated.acknowledged_by == "Nurse Ratched"


def test_api_authentication_enforcement(tmp_path: Path):
    """Verify that protected routes reject unauthorized requests when api_key is enabled."""
    em = EventManager(db_path=tmp_path / "events.db")
    ev = em.create_event(event_type=EVENT_FALL_CONFIRMED)

    # Create app with secret key
    app = create_app(event_manager=em, api_key="secret-key-123")
    client = TestClient(app)

    # 1. Unauthenticated request to protected route is rejected with 401
    res_no_key = client.post(f"/events/{ev.event_id}/acknowledge", json={"acknowledged_by": "Dr. Who"})
    assert res_no_key.status_code == 401

    # 2. Invalid key is rejected with 401
    res_bad_key = client.post(
        f"/events/{ev.event_id}/acknowledge",
        headers={"X-API-Key": "wrong-key"},
        json={"acknowledged_by": "Dr. Who"},
    )
    assert res_bad_key.status_code == 401

    # 3. Valid key is accepted with 200
    res_ok = client.post(
        f"/events/{ev.event_id}/acknowledge",
        headers={"X-API-Key": "secret-key-123"},
        json={"acknowledged_by": "Dr. Who"},
    )
    assert res_ok.status_code == 200


def test_api_websocket_stream(tmp_path: Path):
    """Verify WebSocket connection handshake and message streaming."""
    em = EventManager(db_path=tmp_path / "events.db")
    app = create_app(event_manager=em)
    client = TestClient(app)

    with client.websocket_connect("/ws/events") as ws:
        welcome = ws.receive_json()
        assert welcome["type"] == "connection_established"

        # Ping pong test
        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"
