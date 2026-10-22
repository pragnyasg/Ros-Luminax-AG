"""
Emergency event manager compatibility module (re-exports from luminax_vision.event_manager).
"""

from luminax_vision.event_manager import (
    EVENT_CAMERA_ERROR,
    EVENT_CAMERA_RECOVERED,
    EVENT_FALL_CONFIRMED,
    EVENT_FALL_SUSPECTED,
    EVENT_PERSON_MISSING,
    EVENT_PERSON_REAPPEARED,
    EVENT_SYSTEM_ERROR,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    EmergencyEvent,
    EmergencyEventManager,
    EventManager,
)

__all__ = [
    "EVENT_CAMERA_ERROR",
    "EVENT_CAMERA_RECOVERED",
    "EVENT_FALL_CONFIRMED",
    "EVENT_FALL_SUSPECTED",
    "EVENT_PERSON_MISSING",
    "EVENT_PERSON_REAPPEARED",
    "EVENT_SYSTEM_ERROR",
    "SEVERITY_CRITICAL",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "EmergencyEvent",
    "EmergencyEventManager",
    "EventManager",
]
