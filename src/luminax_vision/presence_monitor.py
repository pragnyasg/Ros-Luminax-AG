"""
Monitored-Person Presence & Disappearance Tracking Module for LUMI Vision.

Features:
- Tracks presence of a designated patient/individual or any active track in general mode.
- Explicit presence states: PRESENT, TEMPORARILY_MISSING, MISSING, CAMERA_ERROR, UNKNOWN.
- Tolerates brief tracking occlusions without triggering premature caregiver alarms.
- Accurately distinguishes camera hardware failure from person departure.
- NEVER equates disappearance with a confirmed fall.
- Generates event triggers on confirmed disappearance and subsequent reappearance.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from luminax_vision.person_tracker import PersonTrack

logger = logging.getLogger(__name__)


class PresenceState(str, enum.Enum):
    PRESENT = "PRESENT"
    TEMPORARILY_MISSING = "TEMPORARILY_MISSING"
    MISSING = "MISSING"
    CAMERA_ERROR = "CAMERA_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class PresenceStatus:
    """Current evaluated presence status for monitored individual."""
    state: PresenceState
    target_person: Optional[str]
    event_triggered: Optional[Dict[str, Any]]
    absence_duration_sec: float
    last_seen_timestamp: Optional[float]
    last_known_bbox: Optional[Tuple[int, int, int, int]]
    monitored_track_id: Optional[int]
    matched_person_name: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def matched_track_id(self) -> Optional[int]:
        return self.monitored_track_id

    @property
    def is_event_triggered(self) -> bool:
        return self.event_triggered is not None

    @property
    def event_type(self) -> Optional[str]:
        return self.event_triggered.get("event_type") if self.event_triggered else None

    @property
    def duration_absent_sec(self) -> float:
        return self.absence_duration_sec

    @property
    def monitored_target(self) -> str:
        return self.target_person or "General"


# Backward and forward compatibility alias
PresenceResult = PresenceStatus


class PresenceMonitor:
    """
    Monitored-person presence and disappearance supervisor.
    """

    def __init__(
        self,
        target_person: Optional[str] = None,
        occlusion_tolerance_sec: float = 2.0,
        missing_timeout_sec: float = 5.0,
        alert_on_missing: bool = True,
        monitored_target: Optional[str] = None,
        temporary_missing_timeout_sec: Optional[float] = None,
    ) -> None:
        target = monitored_target if monitored_target is not None else target_person
        self.target_person = target.strip() if target else None
        self.monitored_target = self.target_person or "General"
        self.occlusion_tolerance_sec = float(temporary_missing_timeout_sec if temporary_missing_timeout_sec is not None else occlusion_tolerance_sec)
        self.missing_timeout_sec = float(missing_timeout_sec)
        self.alert_on_missing = bool(alert_on_missing)

        self._state: PresenceState = PresenceState.UNKNOWN
        self._last_seen_time: Optional[float] = None
        self._last_known_bbox: Optional[Tuple[int, int, int, int]] = None
        self._monitored_track_id: Optional[int] = None
        self._matched_name: str = "Unknown"
        self._missing_alert_emitted: bool = False
        self._camera_error_emitted: bool = False

    @property
    def current_state(self) -> PresenceState:
        return self._state

    def reset(self) -> None:
        self._state = PresenceState.UNKNOWN
        self._last_seen_time = None
        self._last_known_bbox = None
        self._monitored_track_id = None
        self._matched_name = "Unknown"
        self._missing_alert_emitted = False
        self._camera_error_emitted = False

    def set_target_person(self, name: Optional[str]) -> None:
        self.target_person = name.strip() if name else None
        self.monitored_target = self.target_person or "General"

    def set_monitored_target(self, name: Optional[str]) -> None:
        self.set_target_person(name)

    def report_camera_error(self, message: str = "Camera read failed", current_time: Optional[float] = None) -> PresenceStatus:
        return self.update([], camera_healthy=False, current_time=current_time)

    def update(
        self,
        active_tracks: List[PersonTrack],
        camera_healthy: bool = True,
        camera_ok: Optional[bool] = None,
        current_time: Optional[float] = None,
    ) -> PresenceStatus:
        """
        Evaluate presence of monitored person among active person tracks.
        """
        if current_time is None:
            current_time = time.time()

        is_healthy = camera_ok if camera_ok is not None else camera_healthy
        prev_state = self._state
        event_triggered: Optional[Dict[str, Any]] = None

        if not is_healthy:
            self._state = PresenceState.CAMERA_ERROR
            if not self._camera_error_emitted:
                event_triggered = {"event_type": "camera_error"}
                self._camera_error_emitted = True
            dur = (current_time - self._last_seen_time) if self._last_seen_time else 0.0
            return PresenceStatus(
                state=self._state,
                target_person=self.target_person,
                event_triggered=event_triggered,
                absence_duration_sec=round(dur, 2),
                last_seen_timestamp=self._last_seen_time,
                last_known_bbox=self._last_known_bbox,
                monitored_track_id=None,
                matched_person_name="None",
            )

        if prev_state == PresenceState.CAMERA_ERROR:
            self._camera_error_emitted = False
            event_triggered = {"event_type": "camera_recovered"}

        # Find matching track
        matched_track: Optional[PersonTrack] = None
        if not self.target_person:
            # General monitoring mode: any visible tracked person
            if active_tracks:
                matched_track = active_tracks[0]
        else:
            # Target individual mode
            t_lower = self.target_person.lower()
            for t in active_tracks:
                if t.identity_name and t.identity_name.lower() == t_lower:
                    matched_track = t
                    break

        if matched_track is not None:
            self._last_seen_time = current_time
            self._last_known_bbox = matched_track.bbox
            self._monitored_track_id = matched_track.track_id
            self._matched_name = matched_track.identity_name

            if prev_state == PresenceState.MISSING:
                event_triggered = {"event_type": "person_reappeared"}
                logger.info(f"Monitored person '{self.target_person or 'Person'}' REAPPEARED")

            self._state = PresenceState.PRESENT
            self._missing_alert_emitted = False
            absence_dur = 0.0
        else:
            if self._last_seen_time is None:
                self._state = PresenceState.UNKNOWN
                absence_dur = 0.0
            else:
                absence_dur = current_time - self._last_seen_time
                if absence_dur <= self.occlusion_tolerance_sec:
                    self._state = PresenceState.TEMPORARILY_MISSING
                else:
                    self._state = PresenceState.MISSING
                    if not self._missing_alert_emitted and absence_dur >= self.missing_timeout_sec:
                        if self.alert_on_missing:
                            event_triggered = {"event_type": "person_missing"}
                        self._missing_alert_emitted = True
                        logger.warning(f"Monitored person missing for {absence_dur:.1f}s")

        return PresenceStatus(
            state=self._state,
            target_person=self.target_person,
            event_triggered=event_triggered,
            absence_duration_sec=round(absence_dur, 2),
            last_seen_timestamp=self._last_seen_time,
            last_known_bbox=self._last_known_bbox,
            monitored_track_id=self._monitored_track_id if self._state == PresenceState.PRESENT else None,
            matched_person_name=self._matched_name if self._state == PresenceState.PRESENT else "None",
        )


# Alias
PresenceResult = PresenceStatus
