"""
Emergency event management and SQLite persistence module for LUMI Vision.

Records, indexes, and queries emergency and diagnostic events:
fall_suspected, fall_confirmed, person_missing, person_reappeared,
camera_error, camera_recovered, system_error.

Saves associated snapshot frames to disk with configurable retention limits.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard Event Types
EVENT_FALL_SUSPECTED = "fall_suspected"
EVENT_FALL_CONFIRMED = "fall_confirmed"
EVENT_PERSON_MISSING = "person_missing"
EVENT_PERSON_REAPPEARED = "person_reappeared"
EVENT_CAMERA_ERROR = "camera_error"
EVENT_CAMERA_RECOVERED = "camera_recovered"
EVENT_SYSTEM_ERROR = "system_error"

# Severities
SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass
class EmergencyEvent:
    """Standardized schema for all emergency and operational events."""
    event_id: str
    event_type: str
    severity: str
    timestamp: str  # ISO 8601 UTC string
    person_name: Optional[str] = None
    track_id: Optional[int] = None
    confidence: float = 0.0
    posture: str = "unknown"
    last_seen_timestamp: Optional[str] = None
    duration_seconds: Optional[float] = None
    bbox: Optional[List[int]] = None  # [x, y, w, h]
    snapshot_path: Optional[str] = None
    robot_location: Optional[Dict[str, Any]] = None  # None unless physical localization attached
    source_component: str = "vision_pipeline"
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to JSON-serializable dictionary."""
        d = asdict(self)
        return d


class EventManager:
    """
    Persistent SQLite storage and snapshot manager for emergency events.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/events.db",
        snapshots_dir: Union[str, Path] = "data/snapshots",
        save_snapshots: bool = True,
        max_snapshots: int = 500,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.snapshots_dir = Path(snapshots_dir).resolve()
        self.save_snapshots = save_snapshots
        self.max_snapshots = int(max_snapshots)

        self._ensure_dirs()
        self._init_db()

    def _ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_snapshots:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create events table and indices if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    person_name TEXT,
                    track_id INTEGER,
                    confidence REAL,
                    posture TEXT,
                    last_seen_timestamp TEXT,
                    duration_seconds REAL,
                    bbox_json TEXT,
                    snapshot_path TEXT,
                    robot_location_json TEXT,
                    source_component TEXT,
                    details_json TEXT,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ack ON events(acknowledged)")
            conn.commit()
        logger.debug(f"EventManager initialized with database at {self.db_path}")

    def save_snapshot(self, frame: Optional[np.ndarray], event_id: str) -> Optional[str]:
        """
        Save an annotated frame as a JPEG image for an incident.
        Returns relative file path string, or None if frame is missing or write fails.
        """
        if not self.save_snapshots or frame is None or frame.size == 0:
            return None

        filename = f"{event_id}.jpg"
        dest_path = self.snapshots_dir / filename

        try:
            # Enforce max snapshots retention
            self._prune_snapshots_if_needed()

            # Encode and write JPEG with high quality
            cv2.imwrite(str(dest_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            logger.debug(f"Saved incident snapshot: {dest_path.name}")
            return f"snapshots/{filename}"
        except Exception as exc:
            logger.error(f"Failed to save snapshot for event {event_id}: {exc}")
            return None

    def _prune_snapshots_if_needed(self) -> None:
        """Keep snapshot directory bounded below max_snapshots."""
        try:
            snapshots = sorted(self.snapshots_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
            if len(snapshots) >= self.max_snapshots:
                excess = len(snapshots) - self.max_snapshots + 1
                for old_file in snapshots[:excess]:
                    try:
                        old_file.unlink()
                    except OSError:
                        pass
        except Exception as exc:
            logger.warning(f"Snapshot directory pruning error: {exc}")

    def create_event(
        self,
        event_type: str,
        severity: str = SEVERITY_WARNING,
        person_name: Optional[str] = None,
        track_id: Optional[int] = None,
        confidence: float = 1.0,
        posture: str = "unknown",
        last_seen_timestamp: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        bbox: Optional[Sequence[int]] = None,
        frame: Optional[np.ndarray] = None,
        robot_location: Optional[Dict[str, Any]] = None,
        source_component: str = "vision_pipeline",
        details: Optional[Dict[str, Any]] = None,
        custom_event_id: Optional[str] = None,
        custom_timestamp: Optional[str] = None,
    ) -> EmergencyEvent:
        """
        Create, persist, and return a new EmergencyEvent.
        """
        event_id = custom_event_id or str(uuid.uuid4())
        now_utc = custom_timestamp or datetime.now(timezone.utc).isoformat()
        box_list = list(bbox) if bbox is not None else None

        # Capture snapshot if frame provided
        snapshot_rel_path = self.save_snapshot(frame, event_id)

        event = EmergencyEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=now_utc,
            person_name=person_name,
            track_id=track_id,
            confidence=float(confidence),
            posture=posture,
            last_seen_timestamp=last_seen_timestamp,
            duration_seconds=duration_seconds,
            bbox=box_list,
            snapshot_path=snapshot_rel_path,
            robot_location=robot_location,
            source_component=source_component,
            details=details or {},
            acknowledged=False,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, event_type, severity, timestamp, person_name,
                    track_id, confidence, posture, last_seen_timestamp,
                    duration_seconds, bbox_json, snapshot_path, robot_location_json,
                    source_component, details_json, acknowledged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.severity,
                    event.timestamp,
                    event.person_name,
                    event.track_id,
                    event.confidence,
                    event.posture,
                    event.last_seen_timestamp,
                    event.duration_seconds,
                    json.dumps(event.bbox) if event.bbox is not None else None,
                    event.snapshot_path,
                    json.dumps(event.robot_location) if event.robot_location is not None else None,
                    event.source_component,
                    json.dumps(event.details),
                    0,
                ),
            )
            conn.commit()

        logger.info(
            f"[EVENT STORED] ID={event.event_id} Type={event.event_type} "
            f"Severity={event.severity} Person={event.person_name}"
        )
        return event

    def get_event(self, event_id: str) -> Optional[EmergencyEvent]:
        """Retrieve single event by event_id."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_event(row)

    def list_events(
        self,
        limit: int = 50,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        unacknowledged_only: bool = False,
    ) -> List[EmergencyEvent]:
        """
        Query recent emergency events.
        """
        query = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        if unacknowledged_only:
            query += " AND acknowledged = 0"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))

        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_event(r) for r in rows]

    def acknowledge_event(
        self,
        event_id: str,
        acknowledged_by: str = "caregiver",
    ) -> bool:
        """
        Mark an emergency event as acknowledged.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE events
                SET acknowledged = 1, acknowledged_at = ?, acknowledged_by = ?
                WHERE event_id = ?
                """,
                (now_utc, acknowledged_by, event_id),
            )
            conn.commit()
            updated = cursor.rowcount > 0

        if updated:
            logger.info(f"[EVENT ACKNOWLEDGED] ID={event_id} by {acknowledged_by}")
        return updated

    def get_unacknowledged_count(self) -> int:
        """Return total count of unacknowledged emergency events."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events WHERE acknowledged = 0")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def _row_to_event(self, row: sqlite3.Row) -> EmergencyEvent:
        bbox_val = json.loads(row["bbox_json"]) if row["bbox_json"] else None
        robot_loc = json.loads(row["robot_location_json"]) if row["robot_location_json"] else None
        details_val = json.loads(row["details_json"]) if row["details_json"] else {}

        return EmergencyEvent(
            event_id=row["event_id"],
            event_type=row["event_type"],
            severity=row["severity"],
            timestamp=row["timestamp"],
            person_name=row["person_name"],
            track_id=row["track_id"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else 0.0,
            posture=row["posture"] or "unknown",
            last_seen_timestamp=row["last_seen_timestamp"],
            duration_seconds=row["duration_seconds"],
            bbox=bbox_val,
            snapshot_path=row["snapshot_path"],
            robot_location=robot_loc,
            source_component=row["source_component"] or "vision_pipeline",
            details=details_val,
            acknowledged=bool(row["acknowledged"]),
            acknowledged_at=row["acknowledged_at"],
            acknowledged_by=row["acknowledged_by"],
        )


# Backward compatibility alias
EmergencyEventManager = EventManager
