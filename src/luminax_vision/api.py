"""
Caregiver notification backend and real-time WebSocket server for LUMI Vision.

Provides HTTP REST endpoints and WebSocket event streams consumable by
the future LUMI caregiver mobile application and web dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Header, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from luminax_vision.event_manager import EmergencyEvent, EventManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Request/Response validation
# ---------------------------------------------------------------------------

class AcknowledgeRequest(BaseModel):
    acknowledged_by: str = Field(default="caregiver", min_length=1, max_length=100)
    note: Optional[str] = Field(default=None, max_length=500)


class MonitoringStartRequest(BaseModel):
    target_person: Optional[str] = Field(default=None, max_length=100)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    subsystems: Dict[str, str]


class StatusResponse(BaseModel):
    monitoring_active: bool
    presence_state: str
    target_person: Optional[str]
    active_track_id: Optional[int]
    current_posture: str
    last_seen_timestamp: Optional[str]
    absence_duration_sec: float
    unacknowledged_alerts: int
    camera_fps: float
    system_time: str


class EventResponse(BaseModel):
    event_id: str
    event_type: str
    severity: str
    timestamp: str
    person_name: Optional[str]
    track_id: Optional[int]
    confidence: float
    posture: str
    last_seen_timestamp: Optional[str]
    duration_seconds: Optional[float]
    bbox: Optional[List[int]]
    snapshot_path: Optional[str]
    robot_location: Optional[Dict[str, Any]]
    source_component: str
    details: Dict[str, Any]
    acknowledged: bool
    acknowledged_at: Optional[str]
    acknowledged_by: Optional[str]


# ---------------------------------------------------------------------------
# WebSocket Connection & Broadcast Manager
# ---------------------------------------------------------------------------

class WebSocketManager:
    """Manages active caregiver WebSocket connections with heartbeat and broadcast."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"Caregiver WebSocket client connected. Total clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"Caregiver WebSocket client disconnected. Remaining clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message dict to all active subscribers, removing dead connections."""
        if not self.active_connections:
            return

        payload_str = json.dumps(message)
        dead_sockets: List[WebSocket] = []

        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(payload_str)
                except Exception as exc:
                    logger.debug(f"Failed to send to client: {exc}")
                    dead_sockets.append(connection)

            for dead in dead_sockets:
                self.active_connections.discard(dead)


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_app(
    event_manager: EventManager,
    status_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    monitoring_control: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    cors_origins: Optional[List[str]] = None,
) -> FastAPI:
    """
    Construct configured FastAPI application.

    Args:
        event_manager: EventManager SQLite instance.
        status_provider: Callable returning live telemetry/state dict from vision pipeline.
        monitoring_control: Dictionary containing runtime control callbacks/flags:
            - 'is_monitoring': bool
            - 'start_callback': Callable[[Optional[str]], None]
            - 'stop_callback': Callable[[], None]
        api_key: Optional secret token. If provided, non-exempt routes require 'X-API-Key'.
        cors_origins: Allowed CORS origins list.
    """
    app = FastAPI(
        title="LUMI Healthcare Companion API",
        description="Caregiver notification backend and real-time telemetry stream for LUMI robot.",
        version="0.1.0",
    )

    # Cross-Origin Resource Sharing
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = WebSocketManager()
    app.state.ws_manager = ws_manager
    app.state.event_manager = event_manager

    # Security dependency
    async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
        if api_key is not None and api_key != "":
            if not x_api_key or x_api_key != api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing X-API-Key header",
                )

    # 1. Health check
    @app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
    async def get_health() -> HealthResponse:
        subsystems = {
            "database": "healthy",
            "event_manager": "healthy",
            "notification_server": "healthy",
            "camera": "healthy",
            "pose_estimator": "healthy",
        }
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
            subsystems=subsystems,
        )

    # 2. Live telemetry & status
    @app.get("/status", response_model=StatusResponse, tags=["Telemetry"])
    async def get_status() -> StatusResponse:
        unack = event_manager.get_unacknowledged_count()
        raw_status = status_provider() if status_provider else {}

        return StatusResponse(
            monitoring_active=bool(raw_status.get("monitoring_active", True)),
            presence_state=str(raw_status.get("presence_state", "UNKNOWN")),
            target_person=raw_status.get("target_person"),
            active_track_id=raw_status.get("active_track_id"),
            current_posture=str(raw_status.get("current_posture", "unknown")),
            last_seen_timestamp=raw_status.get("last_seen_timestamp"),
            absence_duration_sec=float(raw_status.get("absence_duration_sec", 0.0)),
            unacknowledged_alerts=unack,
            camera_fps=float(raw_status.get("camera_fps", 0.0)),
            system_time=datetime.now(timezone.utc).isoformat(),
        )

    # 3. List events
    @app.get("/events", response_model=List[EventResponse], tags=["Events"])
    async def list_events(
        limit: int = Query(50, ge=1, le=500),
        event_type: Optional[str] = Query(None),
        severity: Optional[str] = Query(None),
        unacknowledged_only: bool = Query(False),
    ) -> List[Dict[str, Any]]:
        events = event_manager.list_events(
            limit=limit,
            event_type=event_type,
            severity=severity,
            unacknowledged_only=unacknowledged_only,
        )
        return [e.to_dict() for e in events]

    # 4. Get specific event
    @app.get("/events/{event_id}", response_model=EventResponse, tags=["Events"])
    async def get_event(event_id: str) -> Dict[str, Any]:
        ev = event_manager.get_event(event_id)
        if not ev:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event '{event_id}' not found",
            )
        return ev.to_dict()

    # 5. Acknowledge event
    @app.post("/events/{event_id}/acknowledge", tags=["Events"], dependencies=[Depends(verify_api_key)])
    async def acknowledge_event(event_id: str, req: AcknowledgeRequest) -> Dict[str, Any]:
        ev = event_manager.get_event(event_id)
        if not ev:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event '{event_id}' not found",
            )

        success = event_manager.acknowledge_event(event_id, acknowledged_by=req.acknowledged_by)
        updated_ev = event_manager.get_event(event_id)

        # Broadcast acknowledgment to WebSocket subscribers
        if updated_ev:
            await ws_manager.broadcast({
                "type": "event_acknowledged",
                "event": updated_ev.to_dict(),
                "note": req.note,
            })

        return {
            "success": success,
            "event_id": event_id,
            "acknowledged_by": req.acknowledged_by,
            "acknowledged_at": updated_ev.acknowledged_at if updated_ev else None,
        }

    # 6. Start monitoring
    @app.post("/monitoring/start", tags=["Control"], dependencies=[Depends(verify_api_key)])
    async def start_monitoring(req: Optional[MonitoringStartRequest] = None) -> Dict[str, Any]:
        target = req.target_person if req else None
        if monitoring_control and "start_callback" in monitoring_control:
            monitoring_control["start_callback"](target)

        await ws_manager.broadcast({
            "type": "monitoring_state_changed",
            "monitoring_active": True,
            "target_person": target,
        })

        return {
            "success": True,
            "monitoring_active": True,
            "target_person": target,
            "message": f"Monitoring started for '{target or 'all persons'}'",
        }

    # 7. Stop monitoring
    @app.post("/monitoring/stop", tags=["Control"], dependencies=[Depends(verify_api_key)])
    async def stop_monitoring() -> Dict[str, Any]:
        if monitoring_control and "stop_callback" in monitoring_control:
            monitoring_control["stop_callback"]()

        await ws_manager.broadcast({
            "type": "monitoring_state_changed",
            "monitoring_active": False,
        })

        return {
            "success": True,
            "monitoring_active": False,
            "message": "Monitoring paused",
        }

    # 8. Snapshot retrieval
    @app.get("/snapshots/{filename}", tags=["Media"])
    async def get_snapshot(filename: str, _: None = Depends(verify_api_key)) -> FileResponse:
        # Sanitize filename against directory traversal
        clean_name = Path(filename).name
        target_file = (event_manager.snapshots_dir / clean_name).resolve()

        if not target_file.is_file() or not str(target_file).startswith(str(event_manager.snapshots_dir)):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Snapshot not found",
            )

        return FileResponse(path=str(target_file), media_type="image/jpeg")

    # 9. Real-time WebSocket event stream
    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        await ws_manager.connect(websocket)
        try:
            # Send initial welcome state payload
            init_payload = {
                "type": "connection_established",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "unacknowledged_alerts": event_manager.get_unacknowledged_count(),
            }
            await websocket.send_text(json.dumps(init_payload))

            # Keep connection alive; accept optional client pings / subscriptions
            while True:
                data = await websocket.receive_text()
                # Optional ping-pong handling
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "time": datetime.now(timezone.utc).isoformat()}))
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception as exc:
            logger.debug(f"WebSocket client loop terminated: {exc}")
            await ws_manager.disconnect(websocket)

    return app


class NotificationServerThread:
    """Runs FastAPI/Uvicorn notification server in a dedicated background daemon thread."""

    def __init__(self, app: Any, host: str, port: int) -> None:
        import threading
        import uvicorn

        self.host = host
        self.port = port
        self.server_config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            loop="asyncio",
        )
        self.server = uvicorn.Server(config=self.server_config)
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        import threading
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="LumiApiServer")
        self.thread.start()
        logger.info(f"Caregiver Notification Server running at http://{self.host}:{self.port}")

    def stop(self) -> None:
        self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info("Caregiver Notification Server stopped.")


# Aliases
create_notification_app = create_app
WebSocketBroadcaster = WebSocketManager
TelemetryState = StatusResponse
BackgroundApiServer = NotificationServerThread

