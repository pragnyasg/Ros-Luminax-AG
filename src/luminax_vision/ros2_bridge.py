"""
ROS 2 bridge adapter and message schema for LUMI Vision.

Defines transport-agnostic serialization schemas mapping LUMI vision events,
person tracks, and pose telemetry into standard ROS 2 (vision_msgs, sensor_msgs)
and custom LUMI message structures without introducing a hard ROS 2 dependency.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from luminax_vision.event_manager import EmergencyEvent
from luminax_vision.person_tracker import PersonTrack
from luminax_vision.presence_monitor import PresenceStatus

logger = logging.getLogger(__name__)


@dataclass
class RosHeader:
    """Standard ROS 2 std_msgs/Header representation."""
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str = "camera_color_optical_frame"


@dataclass
class RosDetection2D:
    """Standard ROS 2 vision_msgs/Detection2D representation."""
    header: RosHeader
    track_id: int
    class_id: str  # "person"
    score: float
    bbox_center_x: float
    bbox_center_y: float
    bbox_size_x: float
    bbox_size_y: float
    identity_name: str
    is_known_identity: bool
    posture: str
    velocity_y: float


@dataclass
class RosFallAlertMsg:
    """Custom ROS 2 message schema matching lumi_msgs/msg/FallAlert.msg."""
    header: RosHeader
    event_id: str
    event_type: str
    severity: str
    track_id: int
    person_name: str
    confidence: float
    posture: str
    torso_angle_deg: float
    peak_velocity_y: float
    snapshot_path: Optional[str]


class Ros2Bridge:
    """
    Serializes pure Python vision structures into ROS 2 message compatible payloads.
    Enables zero-friction translation for ros2_ws/src/luminax_vision_ros.
    """

    def __init__(self, frame_id: str = "camera_color_optical_frame") -> None:
        self.frame_id = frame_id

    def create_header(self, timestamp: float) -> RosHeader:
        sec = int(timestamp)
        nanosec = int((timestamp - sec) * 1e9)
        return RosHeader(stamp_sec=sec, stamp_nanosec=nanosec, frame_id=self.frame_id)

    def track_to_detection2d(
        self,
        track: PersonTrack,
        posture: str,
        timestamp: float,
    ) -> RosDetection2D:
        """Convert a PersonTrack into vision_msgs/Detection2D structure."""
        x, y, w, h = track.bbox
        cx = float(x + (w / 2.0))
        cy = float(y + (h / 2.0))

        return RosDetection2D(
            header=self.create_header(timestamp),
            track_id=track.track_id,
            class_id="person",
            score=track.pose.score if track.pose else 0.9,
            bbox_center_x=cx,
            bbox_center_y=cy,
            bbox_size_x=float(w),
            bbox_size_y=float(h),
            identity_name=track.identity_name,
            is_known_identity=track.is_known_identity,
            posture=posture,
            velocity_y=track.velocity_y,
        )

    def event_to_fall_alert(
        self,
        event: EmergencyEvent,
        timestamp: float,
    ) -> RosFallAlertMsg:
        """Convert an EmergencyEvent into lumi_msgs/FallAlert structure."""
        details = event.details or {}
        return RosFallAlertMsg(
            header=self.create_header(timestamp),
            event_id=event.event_id,
            event_type=event.event_type,
            severity=event.severity,
            track_id=event.track_id or 0,
            person_name=event.person_name or "Unknown",
            confidence=event.confidence,
            posture=event.posture,
            torso_angle_deg=float(details.get("torso_angle", details.get("final_torso_angle", 0.0))),
            peak_velocity_y=float(details.get("peak_velocity_y", 0.0)),
            snapshot_path=event.snapshot_path,
        )

    def presence_to_ros_payload(
        self,
        status: PresenceStatus,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Convert PresenceStatus into diagnostic message dictionary."""
        return {
            "header": asdict(self.create_header(timestamp)),
            "presence_state": status.state.value,
            "target_person": status.target_person,
            "monitored_track_id": status.monitored_track_id,
            "absence_duration_sec": status.absence_duration_sec,
            "last_known_bbox": status.last_known_bbox,
        }
