#!/usr/bin/env python3
"""
LUMI Vision ROS 2 Jazzy Node for Raspberry Pi 5.

Subscribes:
  - /camera/image_raw (sensor_msgs/msg/Image): Live robot video frames

Publishes:
  - /vision/person_detected (std_msgs/msg/Bool): True when person is detected in frame
  - /vision/person_count (std_msgs/msg/Int32): Number of tracked persons
  - /vision/detections (std_msgs/msg/String): Structured JSON with identities, bounding boxes, posture
  - /vision/posture (std_msgs/msg/String): Current primary posture (standing, sitting, walking, etc.)
  - /vision/fall_state (std_msgs/msg/String): Current fall state (NORMAL, FALL_SUSPECTED, FALL_CONFIRMED, etc.)
  - /vision/presence (std_msgs/msg/String): Presence state (PRESENT, TEMPORARILY_MISSING, MISSING, etc.)
  - /vision/events/emergency (std_msgs/msg/String): Emergency incident JSON alert
  - /vision/image_annotated (sensor_msgs/msg/Image): Visual overlay for RViz2 / web dashboard
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# ROS 2 imports with conditional fallback for non-ROS testing
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Int32, String
    try:
        from cv_bridge import CvBridge
        HAS_CV_BRIDGE = True
    except ImportError:
        HAS_CV_BRIDGE = False
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    HAS_CV_BRIDGE = False
    Node = object  # type: ignore

# LUMI Vision Core Module imports
try:
    from luminax_vision.activity_classifier import ActivityClassifier, PostureResult
    from luminax_vision.config import Config, load_config
    from luminax_vision.event_manager import (
        EVENT_FALL_CONFIRMED,
        EVENT_FALL_SUSPECTED,
        EVENT_PERSON_MISSING,
        EVENT_PERSON_REAPPEARED,
        EventManager,
        SEVERITY_CRITICAL,
        SEVERITY_INFO,
        SEVERITY_WARNING,
    )
    from luminax_vision.face_database import FaceDatabase
    from luminax_vision.face_detector import FaceDetector
    from luminax_vision.face_recognizer import FaceRecognizer
    from luminax_vision.fall_detector import FallState, TemporalFallDetector
    from luminax_vision.person_tracker import PersonTracker
    from luminax_vision.pose_estimator import MediaPipePoseEstimator, MockPoseEstimator
    from luminax_vision.presence_monitor import PresenceMonitor, PresenceState
    HAS_LUMI_VISION = True
except ImportError as err:
    HAS_LUMI_VISION = False
    LUMI_IMPORT_ERROR = err

logger = logging.getLogger('lumi_vision_node')


class LumiVisionNode(Node if HAS_ROS2 else object):
    """
    ROS 2 Adapter Node executing the complete LUMI vision pipeline.
    """

    def __init__(self, config_override: Optional[Any] = None) -> None:
        if HAS_ROS2:
            super().__init__('lumi_vision_node')
            self.declare_parameter('camera_topic', '/camera/image_raw')
            self.declare_parameter('monitored_target', '')
            self.declare_parameter('publish_annotated', True)
            self.declare_parameter('downsample_width', 640)

            self.camera_topic = self.get_parameter('camera_topic').value
            self.monitored_target = self.get_parameter('monitored_target').value
            self.publish_annotated = self.get_parameter('publish_annotated').value
            self.downsample_width = self.get_parameter('downsample_width').value
        else:
            self.camera_topic = '/camera/image_raw'
            self.monitored_target = ''
            self.publish_annotated = True
            self.downsample_width = 640

        # Load configuration
        if HAS_LUMI_VISION:
            self.config = config_override or load_config()
            self._init_pipeline()
        else:
            self.config = None
            if HAS_ROS2:
                self.get_logger().error(f'LUMI Vision dependencies not found: {LUMI_IMPORT_ERROR}')

        # Setup ROS 2 Publishers & Subscriber
        if HAS_ROS2:
            self.bridge = CvBridge() if HAS_CV_BRIDGE else None

            self.pub_detected = self.create_publisher(Bool, '/vision/person_detected', 10)
            self.pub_count = self.create_publisher(Int32, '/vision/person_count', 10)
            self.pub_detections = self.create_publisher(String, '/vision/detections', 10)
            self.pub_posture = self.create_publisher(String, '/vision/posture', 10)
            self.pub_fall_state = self.create_publisher(String, '/vision/fall_state', 10)
            self.pub_presence = self.create_publisher(String, '/vision/presence', 10)
            self.pub_emergency = self.create_publisher(String, '/vision/events/emergency', 10)
            self.pub_annotated = self.create_publisher(Image, '/vision/image_annotated', 10)

            self.sub_image = self.create_subscription(
                Image,
                self.camera_topic,
                self.image_callback,
                10
            )
            self.get_logger().info(f'LUMI Vision ROS 2 Node initialized. Subscribed to {self.camera_topic}')

    def _init_pipeline(self) -> None:
        cfg = self.config
        self.db = FaceDatabase(cfg.database.path)
        self.detector = FaceDetector(
            model_path=cfg.detection.model_path,
            score_threshold=cfg.detection.score_threshold,
            nms_threshold=cfg.detection.nms_threshold,
            top_k=cfg.detection.top_k,
        )
        self.recognizer = FaceRecognizer(
            model_path=cfg.recognition.model_path,
            distance_metric=cfg.recognition.distance_metric,
            cosine_threshold=cfg.recognition.cosine_threshold,
            l2_threshold=cfg.recognition.l2_threshold,
        )
        try:
            self.pose_estimator = MediaPipePoseEstimator(
                model_path=cfg.pose.model_path,
                min_detection_confidence=cfg.pose.min_detection_confidence,
                min_tracking_confidence=cfg.pose.min_tracking_confidence,
                num_poses=cfg.pose.num_poses,
            )
        except Exception:
            self.pose_estimator = MockPoseEstimator(mode='standing')

        self.tracker = PersonTracker(
            max_disappeared=cfg.tracking.max_disappeared,
            max_distance=cfg.tracking.max_distance,
            min_iou=cfg.tracking.min_iou,
        )
        self.classifier = ActivityClassifier(
            rapid_descent_velocity=cfg.fall_detection.rapid_velocity_threshold,
        )
        self.fall_detector = TemporalFallDetector(
            rapid_velocity_threshold=cfg.fall_detection.rapid_velocity_threshold,
            transition_window_sec=cfg.fall_detection.transition_window_sec,
            persistence_duration_sec=cfg.fall_detection.persistence_duration_sec,
            alert_cooldown_sec=cfg.fall_detection.alert_cooldown_sec,
            recovery_duration_sec=cfg.fall_detection.recovery_duration_sec,
        )
        self.presence_monitor = PresenceMonitor(
            target_person=self.monitored_target or cfg.monitoring.target_person,
            occlusion_tolerance_sec=cfg.monitoring.occlusion_tolerance_sec,
            missing_timeout_sec=cfg.monitoring.missing_timeout_sec,
            alert_on_missing=cfg.monitoring.alert_on_missing,
        )
        self.event_manager = EventManager(
            db_path=cfg.events.db_path,
            snapshots_dir=cfg.events.snapshots_dir,
            save_snapshots=cfg.events.save_snapshots,
            max_snapshots=cfg.events.max_snapshots,
        )

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single BGR OpenCV frame through the full LUMI vision pipeline.
        """
        current_time = time.time()
        h, w = frame.shape[:2]

        # 1. Pose estimation
        poses = self.pose_estimator.estimate(frame)

        # 2. Face detection and identity recognition
        face_detections = self.detector.detect(frame)
        rec_results = []
        for det in face_detections:
            rec_res = self.recognizer.recognize(frame, det, self.db)
            rec_results.append(rec_res)

        # 3. Person tracking & face association
        active_tracks = self.tracker.update(poses, (h, w), current_time=current_time)
        self.tracker.associate_faces(rec_results, current_time=current_time)

        # 4. Posture & Fall Detection
        primary_posture = "unknown"
        primary_fall_state = FallState.NORMAL.value
        new_fall_events = []

        for track in active_tracks:
            posture_res = self.classifier.classify(track)
            fall_state, fall_event = self.fall_detector.process(track, posture_res, current_time=current_time)

            if primary_posture == "unknown":
                primary_posture = posture_res.posture
            if fall_state != FallState.NORMAL:
                primary_fall_state = fall_state.value

            if fall_event and fall_event.get("event_type") == "fall_confirmed":
                ev = self.event_manager.create_event(
                    event_type=EVENT_FALL_CONFIRMED,
                    severity=SEVERITY_CRITICAL,
                    person_name=track.identity_name,
                    track_id=track.track_id,
                    confidence=fall_event.get("confidence", 0.95),
                    posture=posture_res.posture,
                    bbox=track.bbox,
                    frame=frame,
                    source_component="ros2_lumi_vision_node",
                )
                new_fall_events.append(ev.to_dict())

        # 5. Presence Monitoring
        presence_status = self.presence_monitor.update(active_tracks, camera_healthy=True, current_time=current_time)
        presence_str = presence_status.state.value

        if presence_status.event_triggered:
            ev_type = presence_status.event_triggered.get("event_type")
            if ev_type == "person_missing":
                ev = self.event_manager.create_event(
                    event_type=EVENT_PERSON_MISSING,
                    severity=SEVERITY_WARNING,
                    person_name=presence_status.target_person or "Monitored Person",
                    track_id=presence_status.monitored_track_id,
                    confidence=1.0,
                    source_component="ros2_lumi_vision_node",
                )
                new_fall_events.append(ev.to_dict())
            elif ev_type == "person_reappeared":
                ev = self.event_manager.create_event(
                    event_type=EVENT_PERSON_REAPPEARED,
                    severity=SEVERITY_INFO,
                    person_name=presence_status.target_person or "Monitored Person",
                    track_id=presence_status.monitored_track_id,
                    confidence=1.0,
                    source_component="ros2_lumi_vision_node",
                )
                new_fall_events.append(ev.to_dict())

        return {
            'person_detected': len(active_tracks) > 0,
            'person_count': len(active_tracks),
            'posture': primary_posture,
            'fall_state': primary_fall_state,
            'presence': presence_str,
            'active_tracks': [
                {
                    'track_id': t.track_id,
                    'identity_name': t.identity_name,
                    'bbox': list(t.bbox),
                    'velocity_y': round(t.velocity_y, 3),
                }
                for t in active_tracks
            ],
            'emergency_events': new_fall_events,
        }

    def image_callback(self, msg: Any) -> None:
        """Process incoming camera video frames from ROS 2 topic."""
        if not HAS_ROS2 or not HAS_LUMI_VISION:
            return

        try:
            if self.bridge:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        except Exception as exc:
            self.get_logger().error(f'Failed to convert ROS Image: {exc}')
            return

        results = self.process_frame(frame)

        # Publish status messages
        bool_msg = Bool()
        bool_msg.data = results['person_detected']
        self.pub_detected.publish(bool_msg)

        count_msg = Int32()
        count_msg.data = results['person_count']
        self.pub_count.publish(count_msg)

        det_msg = String()
        det_msg.data = json.dumps(results['active_tracks'])
        self.pub_detections.publish(det_msg)

        post_msg = String()
        post_msg.data = results['posture']
        self.pub_posture.publish(post_msg)

        fall_msg = String()
        fall_msg.data = results['fall_state']
        self.pub_fall_state.publish(fall_msg)

        pres_msg = String()
        pres_msg.data = results['presence']
        self.pub_presence.publish(pres_msg)

        # Publish emergency alerts if any occurred
        for ev_dict in results['emergency_events']:
            ev_msg = String()
            ev_msg.data = json.dumps(ev_dict)
            self.pub_emergency.publish(ev_msg)
            self.get_logger().warn(f'EMERGENCY ALERT PUBLISHED: {ev_dict.get("event_type")}')


def main(args: Optional[List[str]] = None) -> None:
    """Main entry point for ROS 2 node."""
    if not HAS_ROS2:
        print('ROS 2 (rclpy) is not installed in the current environment.')
        print('To run on Raspberry Pi 5 / Ubuntu 24.04: ros2 run luminax_vision_ros lumi_vision_node')
        return

    rclpy.init(args=args)
    node = LumiVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
