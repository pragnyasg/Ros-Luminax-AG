"""
Temporal Fall-Detection State Machine for LUMI Vision.

Implements a robust multi-stage temporal state machine per tracked person:
    NORMAL -> POSSIBLE_FALL -> FALL_SUSPECTED -> FALL_CONFIRMED -> ALERT_SENT -> RECOVERED -> NORMAL

Key algorithmic safeguards:
- Combines vertical descent velocity, torso inclination, and aspect ratio.
- Requires sustained horizontal posture for a configurable persistence duration.
- Prevents false alarms from deliberate sitting, bending, or floor exercises.
- Enforces event deduplication and alert cooldown to prevent notification floods.
- Detects recovery when person stands back up and resets state for future incidents.
"""

from __future__ import annotations

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from luminax_vision.activity_classifier import (
    POSTURE_FALL_CONFIRMED,
    POSTURE_FALL_SUSPECTED,
    POSTURE_LYING_DOWN,
    POSTURE_SITTING,
    POSTURE_STANDING,
    POSTURE_UNKNOWN,
    POSTURE_WALKING,
    PostureResult,
)
from luminax_vision.person_tracker import PersonTrack

logger = logging.getLogger(__name__)


class FallState(str, enum.Enum):
    NORMAL = "NORMAL"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    FALL_SUSPECTED = "FALL_SUSPECTED"
    FALL_CONFIRMED = "FALL_CONFIRMED"
    ALERT_SENT = "ALERT_SENT"
    RECOVERED = "RECOVERED"


@dataclass
class FallStateResult:
    """Status of fall detection for a single tracked person at a specific frame."""
    track_id: int
    state: FallState
    is_emergency: bool = False
    confidence: float = 1.0
    duration_in_fall_sec: float = 0.0
    cooldown_remaining_sec: float = 0.0
    trigger_reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallIncident:
    """Evidence record for a verified or ongoing fall incident."""
    incident_id: str
    track_id: int
    start_time: float
    confirmed_time: Optional[float] = None
    peak_velocity_y: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonFallContext:
    """Internal state tracking for an individual person track."""
    track_id: int
    state: FallState = FallState.NORMAL
    state_enter_time: float = 0.0
    fall_start_time: Optional[float] = None
    horizontal_start_time: Optional[float] = None
    last_upright_time: float = 0.0
    last_alert_time: float = -999999.0
    peak_descent_velocity: float = 0.0
    incident: Optional[FallIncident] = None


class TemporalFallDetector:
    """
    Temporal fall-detection engine with per-track state machines.
    """

    def __init__(
        self,
        rapid_velocity_threshold: float = 0.45,
        transition_window_sec: float = 1.5,
        persistence_duration_sec: float = 1.0,
        alert_cooldown_sec: float = 10.0,
        recovery_duration_sec: float = 1.0,
        min_fall_confidence: float = 0.50,
        cooldown_duration_sec: Optional[float] = None,
        recovery_upright_duration_sec: Optional[float] = None,
        rapid_descent_velocity: Optional[float] = None,
    ) -> None:
        self.rapid_velocity_threshold = float(rapid_descent_velocity if rapid_descent_velocity is not None else rapid_velocity_threshold)
        self.rapid_descent_velocity = self.rapid_velocity_threshold
        self.transition_window_sec = float(transition_window_sec)
        self.persistence_duration_sec = float(persistence_duration_sec)
        self.alert_cooldown_sec = float(cooldown_duration_sec if cooldown_duration_sec is not None else alert_cooldown_sec)
        self.cooldown_duration_sec = self.alert_cooldown_sec
        self.recovery_duration_sec = float(recovery_upright_duration_sec if recovery_upright_duration_sec is not None else recovery_duration_sec)
        self.recovery_upright_duration_sec = self.recovery_duration_sec
        self.min_fall_confidence = float(min_fall_confidence)

        self._contexts: Dict[int, PersonFallContext] = {}

    def get_context(self, track_id: int, current_time: float) -> PersonFallContext:
        if track_id not in self._contexts:
            self._contexts[track_id] = PersonFallContext(
                track_id=track_id,
                state=FallState.NORMAL,
                state_enter_time=current_time,
                last_upright_time=current_time,
            )
        return self._contexts[track_id]

    def reset_track(self, track_id: int) -> None:
        if track_id in self._contexts:
            del self._contexts[track_id]

    def clear(self) -> None:
        self._contexts.clear()

    def process(
        self,
        track: PersonTrack,
        posture_res: PostureResult,
        current_time: Optional[float] = None,
    ) -> Tuple[FallState, Optional[Dict[str, Any]]]:
        """
        Process person track frame through fall-detection state machine.

        Returns:
            Tuple of (current FallState, optional event dict when confirmed)
        """
        if current_time is None:
            current_time = time.time()

        ctx = self.get_context(track.track_id, current_time)
        prev_state = ctx.state
        event_dict: Optional[Dict[str, Any]] = None

        is_upright = posture_res.is_upright
        is_horizontal = posture_res.is_horizontal
        vy = posture_res.velocity_y

        if vy > ctx.peak_descent_velocity:
            ctx.peak_descent_velocity = vy

        # State Machine Transitions
        if ctx.state == FallState.NORMAL:
            if is_upright:
                ctx.last_upright_time = current_time
                ctx.horizontal_start_time = None
                ctx.peak_descent_velocity = 0.0
            elif vy >= self.rapid_velocity_threshold:
                ctx.state = FallState.POSSIBLE_FALL
                ctx.state_enter_time = current_time
                ctx.fall_start_time = current_time
                inc_id = f"inc_{uuid.uuid4().hex[:8]}"
                ctx.incident = FallIncident(
                    incident_id=inc_id,
                    track_id=track.track_id,
                    start_time=current_time,
                    peak_velocity_y=vy,
                )
                logger.info(f"Track #{track.track_id}: POSSIBLE_FALL (vy={vy:.2f})")
            elif is_horizontal:
                if ctx.horizontal_start_time is None:
                    ctx.horizontal_start_time = current_time

        elif ctx.state == FallState.POSSIBLE_FALL:
            time_in_state = current_time - ctx.state_enter_time
            if is_upright:
                ctx.state = FallState.NORMAL
                ctx.state_enter_time = current_time
                ctx.fall_start_time = None
                ctx.incident = None
            elif is_horizontal or posture_res.posture in (POSTURE_FALL_SUSPECTED, POSTURE_LYING_DOWN):
                ctx.state = FallState.FALL_SUSPECTED
                ctx.state_enter_time = current_time
                ctx.horizontal_start_time = current_time
                logger.warning(f"Track #{track.track_id}: FALL_SUSPECTED")
            elif time_in_state > self.transition_window_sec:
                ctx.state = FallState.NORMAL
                ctx.state_enter_time = current_time
                ctx.fall_start_time = None
                ctx.incident = None

        elif ctx.state == FallState.FALL_SUSPECTED:
            if is_upright:
                ctx.state = FallState.NORMAL
                ctx.state_enter_time = current_time
                ctx.fall_start_time = None
                ctx.horizontal_start_time = None
                ctx.incident = None
            elif is_horizontal or posture_res.posture in (POSTURE_FALL_SUSPECTED, POSTURE_LYING_DOWN):
                if ctx.horizontal_start_time is None:
                    ctx.horizontal_start_time = current_time
                horiz_dur = current_time - ctx.horizontal_start_time

                if horiz_dur >= self.persistence_duration_sec:
                    # Enforce alert cooldown
                    time_since_last_alert = current_time - ctx.last_alert_time
                    if time_since_last_alert >= self.alert_cooldown_sec:
                        ctx.state = FallState.FALL_CONFIRMED
                        ctx.state_enter_time = current_time
                        ctx.last_alert_time = current_time
                        if not ctx.incident:
                            ctx.incident = FallIncident(
                                incident_id=f"inc_{uuid.uuid4().hex[:8]}",
                                track_id=track.track_id,
                                start_time=ctx.fall_start_time or current_time,
                            )
                        ctx.incident.confirmed_time = current_time
                        ctx.incident.evidence = {
                            "horizontal_duration": round(horiz_dur, 2),
                            "peak_velocity_y": round(ctx.peak_descent_velocity, 3),
                            "torso_angle_deg": round(posture_res.torso_angle_deg, 1),
                            "aspect_ratio": round(posture_res.aspect_ratio, 2),
                        }
                        event_dict = {
                            "event_type": "fall_confirmed",
                            "confidence": max(self.min_fall_confidence, posture_res.confidence),
                            "incident": ctx.incident,
                        }
                        logger.critical(f"Track #{track.track_id}: FALL CONFIRMED incident={ctx.incident.incident_id}")
                    else:
                        ctx.state = FallState.ALERT_SENT

        elif ctx.state == FallState.FALL_CONFIRMED:
            ctx.state = FallState.ALERT_SENT
            ctx.state_enter_time = current_time

        elif ctx.state == FallState.ALERT_SENT:
            if is_upright:
                ctx.state = FallState.RECOVERED
                ctx.state_enter_time = current_time
                logger.info(f"Track #{track.track_id}: RECOVERED upright")

        elif ctx.state == FallState.RECOVERED:
            time_upright = current_time - ctx.state_enter_time
            if time_upright >= self.recovery_duration_sec:
                ctx.state = FallState.NORMAL
                ctx.state_enter_time = current_time
                ctx.fall_start_time = None
                ctx.horizontal_start_time = None
                ctx.peak_descent_velocity = 0.0
                ctx.incident = None

        return ctx.state, event_dict

    def update(
        self,
        track: PersonTrack,
        posture_res: PostureResult,
        current_time: Optional[float] = None,
    ) -> FallStateResult:
        """Evaluate person track against temporal fall detection state machine."""
        t = current_time if current_time is not None else time.time()
        state, event_dict = self.process(track, posture_res, current_time=t)
        ctx = self.get_context(track.track_id, t)

        is_emergency = event_dict is not None and event_dict.get("event_type") == "fall_confirmed"
        duration_in_fall = max(0.0, t - ctx.fall_start_time) if ctx.fall_start_time is not None else 0.0
        cooldown_remaining = max(0.0, self.alert_cooldown_sec - (t - ctx.last_alert_time))

        return FallStateResult(
            track_id=track.track_id,
            state=state,
            is_emergency=is_emergency,
            confidence=event_dict.get("confidence", posture_res.confidence) if event_dict else posture_res.confidence,
            duration_in_fall_sec=round(duration_in_fall, 2),
            cooldown_remaining_sec=round(cooldown_remaining, 1),
            trigger_reason=event_dict.get("trigger", "") if event_dict else "",
            diagnostics=event_dict.get("details", {}) if event_dict else {},
        )


# Backwards compatibility alias
FallDetector = TemporalFallDetector

