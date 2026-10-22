"""
Main entry point and CLI orchestration for LUMI Vision.

Coordinates real-time webcam acquisition, YuNet face detection, SFace face recognition,
MediaPipe pose estimation, person tracking, activity classification, temporal fall detection,
presence monitoring, SQLite event persistence, and caregiver notification REST/WebSocket server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from luminax_vision.activity_classifier import (
    ALL_POSTURES,
    POSTURE_FALL_CONFIRMED,
    POSTURE_FALL_SUSPECTED,
    POSTURE_LYING_DOWN,
    POSTURE_SITTING,
    POSTURE_STANDING,
    POSTURE_WALKING,
    ActivityClassifier,
    PostureResult,
)
from luminax_vision.api import create_app
from luminax_vision.camera import BaseCamera, Camera, MockCamera
from luminax_vision.camera_utils import print_camera_diagnostics
from luminax_vision.config import Config, load_config
from luminax_vision.event_manager import (
    EVENT_CAMERA_ERROR,
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
from luminax_vision.face_detector import FaceDetection, FaceDetector
from luminax_vision.face_recognizer import FaceRecognizer, RecognitionResult
from luminax_vision.fall_detector import FallState, TemporalFallDetector
from luminax_vision.person_tracker import PersonTrack, PersonTracker
from luminax_vision.pose_estimator import (
    BasePoseEstimator,
    MediaPipePoseEstimator,
    MockPoseEstimator,
    PoseResult,
)
from luminax_vision.presence_monitor import PresenceMonitor, PresenceState, PresenceStatus

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("luminax_vision")


class BackgroundApiServer:
    """Runs FastAPI/Uvicorn notification server in a dedicated background daemon thread."""

    def __init__(self, app: Any, host: str, port: int) -> None:
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
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="LumiApiServer")
        self.thread.start()
        logger.info(f"Caregiver Notification Server running at http://{self.host}:{self.port}")

    def stop(self) -> None:
        self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info("Caregiver Notification Server stopped.")


def run_live_pipeline(
    config: Config,
    camera_override: Optional[BaseCamera] = None,
    pose_override: Optional[BasePoseEstimator] = None,
    max_frames: Optional[int] = None,
    disable_api_server: bool = False,
) -> None:
    """
    Launch the full real-time LUMI computer vision and health monitoring pipeline.

    Args:
        config: Loaded and validated Config object.
        camera_override: Custom BaseCamera instance (useful for automated testing).
        pose_override: Custom BasePoseEstimator instance (useful for tests).
        max_frames: Optional frame limit.
        disable_api_server: If True, skips starting background FastAPI server.
    """
    logger.info("Initializing LUMI Healthcare Companion Vision Pipeline...")

    # 1. Initialize persistent storage and biometric components
    database = FaceDatabase(config.database.path)
    detector = FaceDetector(
        model_path=config.detection.model_path,
        score_threshold=config.detection.score_threshold,
        nms_threshold=config.detection.nms_threshold,
        top_k=config.detection.top_k,
    )
    recognizer = FaceRecognizer(
        model_path=config.recognition.model_path,
        distance_metric=config.recognition.distance_metric,
        cosine_threshold=config.recognition.cosine_threshold,
        l2_threshold=config.recognition.l2_threshold,
    )

    # 2. Initialize pose estimator
    if pose_override is not None:
        pose_estimator = pose_override
    else:
        try:
            pose_estimator = MediaPipePoseEstimator(
                model_path=config.pose.model_path,
                min_detection_confidence=config.pose.min_detection_confidence,
                min_tracking_confidence=config.pose.min_tracking_confidence,
                num_poses=config.pose.num_poses,
            )
        except Exception as exc:
            logger.warning(f"Could not load MediaPipe PoseLandmarker ({exc}). Falling back to MockPoseEstimator.")
            pose_estimator = MockPoseEstimator(mode="standing")

    # 3. Initialize tracking, classification, fall detection, and presence monitor
    tracker = PersonTracker(
        max_disappeared=config.tracking.max_disappeared,
        max_distance=config.tracking.max_distance,
        min_iou=config.tracking.min_iou,
    )
    classifier = ActivityClassifier(
        rapid_descent_velocity=config.fall_detection.rapid_velocity_threshold,
    )
    fall_detector = TemporalFallDetector(
        rapid_velocity_threshold=config.fall_detection.rapid_velocity_threshold,
        transition_window_sec=config.fall_detection.transition_window_sec,
        persistence_duration_sec=config.fall_detection.persistence_duration_sec,
        alert_cooldown_sec=config.fall_detection.alert_cooldown_sec,
        recovery_duration_sec=config.fall_detection.recovery_duration_sec,
    )
    presence_monitor = PresenceMonitor(
        target_person=config.monitoring.target_person,
        occlusion_tolerance_sec=config.monitoring.occlusion_tolerance_sec,
        missing_timeout_sec=config.monitoring.missing_timeout_sec,
        alert_on_missing=config.monitoring.alert_on_missing,
    )
    event_manager = EventManager(
        db_path=config.events.db_path,
        snapshots_dir=config.events.snapshots_dir,
        save_snapshots=config.events.save_snapshots,
        max_snapshots=config.events.max_snapshots,
    )

    # Telemetry state dictionary for FastAPI GET /status
    telemetry_state: Dict[str, Any] = {
        "monitoring_active": True,
        "presence_state": "UNKNOWN",
        "target_person": config.monitoring.target_person,
        "active_track_id": None,
        "current_posture": "unknown",
        "last_seen_timestamp": None,
        "absence_duration_sec": 0.0,
        "camera_fps": 0.0,
    }

    # Background WebSocket broadcast loop hook
    ws_broadcast_queue: List[Dict[str, Any]] = []

    def queue_broadcast(msg: Dict[str, Any]) -> None:
        ws_broadcast_queue.append(msg)

    # 4. Initialize Caregiver Notification API server
    api_server: Optional[BackgroundApiServer] = None
    if config.server.enabled and not disable_api_server:
        api_app = create_app(
            event_manager=event_manager,
            status_provider=lambda: telemetry_state,
            monitoring_control={
                "is_monitoring": True,
                "start_callback": lambda target: presence_monitor.set_target_person(target),
                "stop_callback": lambda: setattr(telemetry_state, "monitoring_active", False),
            },
            api_key=config.server.api_key,
            cors_origins=config.server.cors_origins,
        )
        try:
            api_server = BackgroundApiServer(app=api_app, host=config.server.host, port=config.server.port)
            api_server.start()
        except Exception as exc:
            logger.error(f"Failed to start Caregiver Notification Server: {exc}")

    # 5. Initialize Camera
    if camera_override is not None:
        camera = camera_override
    else:
        try:
            camera = Camera(
                index=config.camera.index,
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
            )
        except ConnectionError as exc:
            logger.error(f"Cannot open camera: {exc}")
            logger.info("Run 'luminax-vision list-cameras' to inspect available devices, or '--mock-camera' for testing.")
            if api_server:
                api_server.stop()
            sys.exit(1)

    window_name = config.display.window_name
    show_window = config.display.show_window
    show_landmarks = config.display.show_landmarks
    show_skeleton = config.pose.show_skeleton

    if show_window:
        logger.info(
            "Interactive Controls:\n"
            "  [Q/ESC] Quit\n"
            "  [E] Enrol Face\n"
            "  [L] List Enrolled Identities\n"
            "  [D] Delete Identity\n"
            "  [S] Toggle Skeleton & Landmarks\n"
            "  [M] Cycle Monitoring Target\n"
            "  [A] Acknowledge Active Alerts\n"
        )

    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            loop_start_time = time.time()

            ret, frame = camera.read()
            if not ret or frame is None:
                # Camera read failed: update presence monitor with hardware error
                presence_monitor.report_camera_error("Video frame acquisition failed", loop_start_time)
                if isinstance(camera, MockCamera):
                    break
                time.sleep(0.03)
                continue

            frame_count += 1

            # FPS calculation
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    current_fps = 15.0 / elapsed
                fps_start_time = time.time()
                telemetry_state["camera_fps"] = round(current_fps, 1)

            clean_frame = frame.copy()
            h, w = clean_frame.shape[:2]

            # -------------------------------------------------------------
            # Stage 1: Face Detection & Recognition
            # -------------------------------------------------------------
            face_detections = detector.detect(clean_frame)
            recognition_results: List[Tuple[FaceDetection, RecognitionResult]] = []
            for f_det in face_detections:
                rec_res = recognizer.process_face(clean_frame, f_det, database)
                recognition_results.append((f_det, rec_res))

            # -------------------------------------------------------------
            # Stage 2: Pose Estimation
            # -------------------------------------------------------------
            poses = pose_estimator.estimate(clean_frame)

            # -------------------------------------------------------------
            # Stage 3: Person Tracking & Identity Association
            # -------------------------------------------------------------
            active_tracks = tracker.update(poses, (h, w), current_time=loop_start_time)
            tracker.associate_faces(recognition_results, current_time=loop_start_time)

            # -------------------------------------------------------------
            # Stage 4: Posture Classification & Temporal Fall Detection
            # -------------------------------------------------------------
            track_postures: Dict[int, PostureResult] = {}
            active_emergency = False

            for track in active_tracks:
                posture_res = classifier.classify(track)
                track_postures[track.track_id] = posture_res

                # Process through temporal fall state machine
                fall_state, fall_event = fall_detector.process(track, posture_res, current_time=loop_start_time)

                if fall_state in (FallState.FALL_CONFIRMED, FallState.ALERT_SENT):
                    active_emergency = True

                # If a new confirmed fall incident was generated, record in EventManager
                if fall_event and fall_event.get("event_type") == "fall_confirmed":
                    incident = fall_event.get("incident")
                    ev = event_manager.create_event(
                        event_type=EVENT_FALL_CONFIRMED,
                        severity=SEVERITY_CRITICAL,
                        person_name=track.identity_name,
                        track_id=track.track_id,
                        confidence=fall_event.get("confidence", 0.95),
                        posture=posture_res.posture,
                        duration_seconds=incident.evidence.get("horizontal_duration", 0.0) if incident else None,
                        bbox=track.bbox,
                        frame=clean_frame,
                        source_component="temporal_fall_detector",
                        details={
                            "torso_angle": posture_res.torso_angle_deg,
                            "aspect_ratio": posture_res.aspect_ratio,
                            "velocity_y": track.velocity_y,
                            "incident_id": incident.incident_id if incident else None,
                        },
                    )
                    queue_broadcast({"type": "emergency_event", "event": ev.to_dict()})

            # -------------------------------------------------------------
            # Stage 5: Monitored Person Presence & Disappearance
            # -------------------------------------------------------------
            presence_status = presence_monitor.update(active_tracks, camera_healthy=True, current_time=loop_start_time)

            if presence_status.event_triggered:
                ev_type = presence_status.event_triggered.get("event_type")
                if ev_type == "person_missing":
                    ev = event_manager.create_event(
                        event_type=EVENT_PERSON_MISSING,
                        severity=SEVERITY_WARNING,
                        person_name=presence_status.target_person or "Monitored Person",
                        track_id=presence_status.monitored_track_id,
                        confidence=1.0,
                        duration_seconds=presence_status.absence_duration_sec,
                        bbox=presence_status.last_known_bbox,
                        source_component="presence_monitor",
                        details={"absence_seconds": presence_status.absence_duration_sec},
                    )
                    queue_broadcast({"type": "presence_alert", "event": ev.to_dict()})

                elif ev_type == "person_reappeared":
                    ev = event_manager.create_event(
                        event_type=EVENT_PERSON_REAPPEARED,
                        severity=SEVERITY_INFO,
                        person_name=presence_status.target_person or "Monitored Person",
                        track_id=presence_status.monitored_track_id,
                        confidence=1.0,
                        source_component="presence_monitor",
                    )
                    queue_broadcast({"type": "presence_info", "event": ev.to_dict()})

            # Update telemetry state for API /status
            telemetry_state["presence_state"] = presence_status.state.value
            telemetry_state["target_person"] = presence_status.target_person
            telemetry_state["active_track_id"] = presence_status.monitored_track_id
            telemetry_state["absence_duration_sec"] = round(presence_status.absence_duration_sec, 1)
            telemetry_state["last_seen_timestamp"] = (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(presence_status.last_seen_timestamp))
                if presence_status.last_seen_timestamp
                else None
            )
            if active_tracks and active_tracks[0].track_id in track_postures:
                telemetry_state["current_posture"] = track_postures[active_tracks[0].track_id].posture

            # Drain queued WebSocket messages if API server is running
            if api_server and hasattr(api_server.server_config, "app"):
                ws_mgr = getattr(api_server.server_config.app.state, "ws_manager", None)
                if ws_mgr and ws_broadcast_queue:
                    while ws_broadcast_queue:
                        item = ws_broadcast_queue.pop(0)
                        try:
                            # Fire and forget into running event loop
                            asyncio.run_coroutine_threadsafe(ws_mgr.broadcast(item), asyncio.get_event_loop())
                        except Exception:
                            pass

            # -------------------------------------------------------------
            # Stage 6: Visual Rendering & Diagnostic Overlays
            # -------------------------------------------------------------
            annotated = clean_frame.copy()

            # Render skeleton lines & landmarks for each tracked person
            if show_skeleton:
                for track in active_tracks:
                    if track.pose:
                        annotated = pose_estimator.draw_pose(annotated, track.pose)

            # Render tracked bounding boxes and identification labels
            for track in active_tracks:
                tx, ty, tw, th = track.bbox
                posture_info = track_postures.get(track.track_id)
                posture_name = posture_info.posture if posture_info else "unknown"

                # Pick color: Red for fall, Green for verified enrolled, Orange for unknown
                if posture_name in (POSTURE_FALL_CONFIRMED, POSTURE_FALL_SUSPECTED):
                    box_color = (0, 0, 255)  # Bright Red
                elif track.is_known_identity:
                    box_color = config.display.box_color_known
                else:
                    box_color = config.display.box_color_unknown

                # Draw bounding box (thicker if fall)
                thickness = 4 if posture_name == POSTURE_FALL_CONFIRMED else 2
                cv2.rectangle(annotated, (tx, ty), (tx + tw, ty + th), box_color, thickness)

                # Format label
                track_label = f"ID #{track.track_id}: {track.identity_name}"
                if track.is_known_identity and config.display.show_score:
                    track_label += f" ({track.identity_score:.2f})"

                posture_badge = f"[{posture_name.upper()}]"

                # Render label tag above bounding box
                (lw1, lh1), _ = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                (lw2, lh2), _ = cv2.getTextSize(posture_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_w = max(lw1, lw2) + 10
                tag_h = lh1 + lh2 + 12

                tag_y = max(ty - tag_h, 5)
                cv2.rectangle(annotated, (tx, tag_y), (tx + tag_w, tag_y + tag_h), box_color, -1)
                cv2.putText(
                    annotated,
                    track_label,
                    (tx + 5, tag_y + lh1 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    posture_badge,
                    (tx + 5, tag_y + lh1 + lh2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Render top diagnostic banner
            unack_alerts = event_manager.get_unacknowledged_count()
            target_str = presence_status.target_person or "General"
            presence_str = presence_status.state.value

            banner_text = (
                f"FPS: {current_fps:.1f} | People: {len(active_tracks)} | "
                f"Target ({target_str}): {presence_str} | Alerts: {unack_alerts}"
            )

            banner_bg = (0, 0, 180) if active_emergency else (30, 30, 30)
            cv2.rectangle(annotated, (0, 0), (w, 36), banner_bg, -1)
            cv2.putText(
                annotated,
                banner_text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if active_emergency else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # If emergency active, show prominent flashing emergency banner
            if active_emergency:
                cv2.rectangle(annotated, (0, h - 45), (w, h), (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    "!!! CRITICAL WARNING: FALL CONFIRMED - CAREGIVER ALERT DISPATCHED !!!",
                    (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # -------------------------------------------------------------
            # Stage 7: GUI Window & User Keyboard Input
            # -------------------------------------------------------------
            if show_window:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    logger.info("Shutdown requested by user.")
                    break
                elif key in (ord("s"), ord("S")):
                    show_skeleton = not show_skeleton
                    logger.info(f"Skeleton & Landmarks overlay: {'ON' if show_skeleton else 'OFF'}")
                elif key in (ord("m"), ord("M")):
                    # Cycle monitoring target: Toggle between Enrolled Person and General Mode
                    enrolled_names = database.list_names()
                    current_tgt = presence_monitor.target_person
                    if not enrolled_names:
                        print("\n[MONITOR] No identities enrolled. Monitoring in General Mode.\n")
                        presence_monitor.set_target_person(None)
                    elif current_tgt is None:
                        new_target = enrolled_names[0]
                        presence_monitor.set_target_person(new_target)
                        print(f"\n[MONITOR] Switched monitoring target to: '{new_target}'\n")
                    else:
                        try:
                            curr_idx = enrolled_names.index(current_tgt)
                            next_idx = (curr_idx + 1) % (len(enrolled_names) + 1)
                            new_target = enrolled_names[next_idx] if next_idx < len(enrolled_names) else None
                        except ValueError:
                            new_target = None
                        presence_monitor.set_target_person(new_target)
                        print(f"\n[MONITOR] Switched monitoring target to: '{new_target or 'General Mode'}'\n")
                elif key in (ord("a"), ord("A")):
                    # Acknowledge all active alerts
                    unack_list = event_manager.list_events(limit=10, unacknowledged_only=True)
                    if unack_list:
                        for ev_item in unack_list:
                            event_manager.acknowledge_event(ev_item.event_id, acknowledged_by="Keyboard Operator")
                        print(f"\n[ACK] Acknowledged {len(unack_list)} active alert(s).\n")
                    else:
                        print("\n[ACK] No active alerts to acknowledge.\n")
                elif key in (ord("l"), ord("L")):
                    names = database.list_names()
                    print("\n--- Enrolled Identities ---")
                    if not names:
                        print("  (Database is empty)")
                    else:
                        for n in names:
                            print(f"  * {n}")
                    print("---------------------------\n")
                elif key in (ord("e"), ord("E")):
                    # Consent-based face enrollment of currently detected face
                    if not face_detections:
                        print("\n[ENROL] No face detected. Face the camera and try again.\n")
                    else:
                        target_det = face_detections[0]
                        print("\n=== CONSENT-BASED IDENTITY ENROLMENT ===")
                        try:
                            candidate_name = input("Enter person's name (or press Enter to cancel): ").strip()
                            if candidate_name:
                                emb = recognizer.extract_embedding(clean_frame, target_det)
                                identity_id = database.enrol(candidate_name, emb)
                                print(f"[SUCCESS] Enrolled '{candidate_name}' (ID: {identity_id})\n")
                            else:
                                print("[CANCEL] Enrolment cancelled.\n")
                        except Exception as err:
                            print(f"[ERROR] Enrolment failed: {err}\n")
                elif key in (ord("d"), ord("D")):
                    print("\n=== DELETE IDENTITY ===")
                    try:
                        del_name = input("Enter name to delete: ").strip()
                        if del_name:
                            if database.delete_identity(del_name):
                                print(f"[SUCCESS] Deleted identity '{del_name}'.\n")
                            else:
                                print(f"[NOT FOUND] Identity '{del_name}' was not found in database.\n")
                    except Exception as err:
                        print(f"[ERROR] Deletion failed: {err}\n")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        camera.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        pose_estimator.close()
        if api_server:
            api_server.stop()
        logger.info("LUMI Vision pipeline shut down cleanly.")


def cli_list_events(config: Config, limit: int = 20) -> None:
    """CLI sub-command to inspect recent emergency event history."""
    em = EventManager(db_path=config.events.db_path)
    events = em.list_events(limit=limit)

    print("\n========================================================")
    print("           LUMI Vision - Recent Emergency Events        ")
    print("========================================================")
    if not events:
        print("  No emergency events recorded in database.")
    else:
        print(f"  Total Retrievable Events: {len(events)}\n")
        print(f"  {'Timestamp':<20} | {'Type':<16} | {'Severity':<8} | {'Person':<12} | {'Ack'}")
        print("  " + "-" * 66)
        for e in events:
            ack_str = "YES" if e.acknowledged else "NO"
            print(f"  {e.timestamp[:19]:<20} | {e.event_type:<16} | {e.severity:<8} | {(e.person_name or 'N/A'):<12} | {ack_str}")
    print("========================================================\n")


def cli_server(config: Config) -> None:
    """Run FastAPI Caregiver Notification Server standalone."""
    import uvicorn

    print(f"\nStarting Standalone LUMI Caregiver API at http://{config.server.host}:{config.server.port}...")
    em = EventManager(db_path=config.events.db_path, snapshots_dir=config.events.snapshots_dir)
    app = create_app(event_manager=em, api_key=config.server.api_key, cors_origins=config.server.cors_origins)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="luminax-vision",
        description="LUMI Autonomous Companion Robot - Vision, Pose Estimation & Fall Detection",
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to custom config YAML file.")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full real-time computer vision pipeline")
    run_parser.add_argument("--mock-camera", action="store_true", help="Use synthetic mock camera instead of physical webcam")
    run_parser.add_argument("--mock-pose", action="store_true", help="Use synthetic mock pose estimator for testing")
    run_parser.add_argument("--headless", action="store_true", help="Run without opening OpenCV GUI window")
    run_parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (testing)")
    run_parser.add_argument("--no-server", action="store_true", help="Disable background FastAPI notification server")
    run_parser.add_argument("--target", "-t", type=str, default=None, help="Designate specific person name to monitor")

    # Enrol command
    enrol_parser = subparsers.add_parser("enrol", help="Enrol a person's biometric face identity")
    enrol_parser.add_argument("--name", "-n", type=str, required=True, help="Person's name")
    enrol_parser.add_argument("--image", "-i", type=str, default=None, help="Path to static image file")
    enrol_parser.add_argument("--samples", "-s", type=int, default=1, help="Number of samples to capture")
    enrol_parser.add_argument("--replace", action="store_true", help="Replace existing samples for this name")

    # List identities command
    subparsers.add_parser("list", help="List all enrolled biometric identities")

    # Delete identity command
    del_parser = subparsers.add_parser("delete", help="Delete an enrolled biometric identity")
    del_parser.add_argument("--name", "-n", type=str, required=True, help="Name of identity to remove")

    # List cameras command
    subparsers.add_parser("list-cameras", help="Probe and list available video capture devices")

    # List events command
    ev_parser = subparsers.add_parser("events", help="List recent emergency events from database")
    ev_parser.add_argument("--limit", type=int, default=20, help="Max events to display")

    # Standalone server command
    subparsers.add_parser("server", help="Run caregiver notification API server standalone")

    args = parser.parse_args()

    config = load_config(args.config)
    cmd = args.command or "run"

    if cmd == "run":
        if getattr(args, "headless", False):
            config.display.show_window = False

        if getattr(args, "target", None):
            config.monitoring.target_person = args.target

        camera_override = None
        if getattr(args, "mock_camera", False):
            camera_override = MockCamera(
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
                max_frames=args.max_frames,
            )

        pose_override = None
        if getattr(args, "mock_pose", False):
            pose_override = MockPoseEstimator(mode="standing")

        run_live_pipeline(
            config=config,
            camera_override=camera_override,
            pose_override=pose_override,
            max_frames=args.max_frames,
            disable_api_server=getattr(args, "no_server", False),
        )
    elif cmd == "enrol":
        db = FaceDatabase(config.database.path)
        det = FaceDetector(config.detection.model_path)
        rec = FaceRecognizer(config.recognition.model_path)

        if args.image:
            img = cv2.imread(str(Path(args.image).resolve()))
            if img is None:
                logger.error(f"Could not load image: {args.image}")
                sys.exit(1)
            dets = det.detect(img)
            if not dets:
                logger.error("No face detected in provided image.")
                sys.exit(1)
            emb = rec.extract_embedding(img, dets[0])
            db.enrol(args.name, emb, replace=args.replace)
            logger.info(f"Successfully enrolled '{args.name}' from {args.image}")
        else:
            logger.error("Interactive enrollment via CLI requires camera or '--image <path>'.")
    elif cmd == "list":
        db = FaceDatabase(config.database.path)
        print("\n--- Enrolled Identities ---")
        for ident in db.list_identities():
            print(f"  * {ident['name']} ({ident['num_samples']} samples, enrolled {ident['enrolled_at'][:10]})")
        print("---------------------------\n")
    elif cmd == "delete":
        db = FaceDatabase(config.database.path)
        if db.delete_identity(args.name):
            print(f"[SUCCESS] Deleted identity '{args.name}'.")
        else:
            print(f"[NOT FOUND] Identity '{args.name}' not found.")
    elif cmd == "list-cameras":
        print_camera_diagnostics()
    elif cmd == "events":
        cli_list_events(config, limit=getattr(args, "limit", 20))
    elif cmd == "server":
        cli_server(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
