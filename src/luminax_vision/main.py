"""
Main entry point and CLI orchestration for LUMI Vision.

Provides live webcam detection/recognition, enrollment workflows,
database management, benchmarking, and headless testing.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from luminax_vision.camera import BaseCamera, Camera, MockCamera
from luminax_vision.config import Config, load_config
from luminax_vision.face_database import FaceDatabase
from luminax_vision.face_detector import FaceDetection, FaceDetector
from luminax_vision.face_recognizer import FaceRecognizer, RecognitionResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("luminax_vision")


def run_live_pipeline(
    config: Config,
    camera_override: Optional[BaseCamera] = None,
    max_frames: Optional[int] = None,
) -> None:
    """
    Launch the real-time camera face detection and recognition pipeline.

    Args:
        config: Loaded and validated Config object.
        camera_override: Custom BaseCamera instance (useful for testing).
        max_frames: Optional frame limit (useful for non-interactive test runs).
    """
    logger.info("Starting LUMI Vision real-time pipeline...")

    # Initialize components
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
            logger.error(f"Cannot start camera: {exc}")
            logger.info("You can test the pipeline using mock camera by running with '--mock-camera'.")
            sys.exit(1)

    window_name = config.display.window_name
    show_landmarks = config.display.show_landmarks
    show_window = config.display.show_window

    if show_window:
        logger.info("Controls: [Q] Quit | [E] Enrol Face | [L] List Identities | [D] Delete Identity | [S] Toggle Landmarks")

    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Frame read failed or video stream ended.")
                break

            frame_count += 1

            # FPS calculation
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    current_fps = 15.0 / elapsed
                fps_start_time = time.time()

            # Keep a pristine, clean frame for feature extraction
            clean_frame = frame.copy()

            # 1. Detect faces
            detections = detector.detect(clean_frame)

            # 2. Recognize each detected face
            recognition_results: List[Tuple[FaceDetection, RecognitionResult]] = []
            for det in detections:
                rec_res = recognizer.process_face(clean_frame, det, database)
                recognition_results.append((det, rec_res))

            # 3. Visual rendering on display copy
            annotated = clean_frame.copy()

            for det, rec in recognition_results:
                x, y, w, h = det.bbox
                is_known = rec.is_known

                color = config.display.box_color_known if is_known else config.display.box_color_unknown

                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

                # Format identity label
                if config.display.show_score:
                    label = f"{rec.name} ({rec.score:.2f})"
                else:
                    label = rec.name

                # Draw label background box
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y - 10, lh + 10)
                cv2.rectangle(
                    annotated,
                    (x, label_y - lh - 4),
                    (x + lw + 6, label_y + baseline - 2),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Optional landmark points
                if show_landmarks:
                    for pt in det.landmarks:
                        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

            # Display FPS counter & system status banner
            if config.display.show_fps:
                info_text = f"FPS: {current_fps:.1f} | Faces: {len(detections)} | Enrolled: {len(database.list_names())}"
                cv2.putText(
                    annotated,
                    info_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Display window
            if show_window:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    logger.info("Exit requested by user.")
                    break
                elif key in (ord("s"), ord("S")):
                    show_landmarks = not show_landmarks
                    logger.info(f"Facial landmarks display: {'ON' if show_landmarks else 'OFF'}")
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
                    # Interactive enrollment of currently detected face
                    if not detections:
                        print("\n[ENROL] No face currently detected in camera. Face the camera and try again.\n")
                    else:
                        target_det = detections[0]
                        print("\n=== CONSENT-BASED IDENTITY ENROLMENT ===")
                        print("Please verify the person has given permission to enrol their biometric data.")
                        try:
                            candidate_name = input("Enter person's name (or press Enter to cancel): ").strip()
                            if candidate_name:
                                emb = recognizer.extract_embedding(clean_frame, target_det)
                                identity_id = database.enrol(candidate_name, emb)
                                print(f"[SUCCESS] Successfully enrolled '{candidate_name}' (ID: {identity_id})\n")
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
        logger.info("Process interrupted by keyboard.")
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()
        logger.info("LUMI Vision pipeline shut down cleanly.")


def cli_enrol(args: argparse.Namespace, config: Config) -> None:
    """CLI sub-command to enrol a face from webcam or static image."""
    name = args.name.strip()
    try:
        FaceDatabase.validate_name(name)
    except ValueError as err:
        logger.error(f"Invalid name: {err}")
        sys.exit(1)

    database = FaceDatabase(config.database.path)
    detector = FaceDetector(
        model_path=config.detection.model_path,
        score_threshold=config.detection.score_threshold,
    )
    recognizer = FaceRecognizer(
        model_path=config.recognition.model_path,
        distance_metric=config.recognition.distance_metric,
    )

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            logger.error(f"Image file not found: {image_path}")
            sys.exit(1)

        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not decode image at {image_path}")
            sys.exit(1)

        detections = detector.detect(img)
        if not detections:
            logger.error(f"No face detected in {image_path}. Ensure clear lighting and facing camera.")
            sys.exit(1)

        emb = recognizer.extract_embedding(img, detections[0])
        database.enrol(name, emb, replace=args.replace)
        logger.info(f"Successfully enrolled '{name}' from {image_path.name}")
    else:
        # Camera-based enrolment
        logger.info(f"Opening camera for enrolment of '{name}' (Target samples: {args.samples})...")
        camera = Camera(index=config.camera.index, width=config.camera.width, height=config.camera.height)
        samples_collected = 0

        try:
            while samples_collected < args.samples:
                ret, frame = camera.read()
                if not ret or frame is None:
                    continue

                clean_frame = frame.copy()
                detections = detector.detect(clean_frame)

                # Show preview
                preview = clean_frame.copy()
                if detections:
                    x, y, w, h = detections[0].bbox
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        preview,
                        f"Sample {samples_collected + 1}/{args.samples} - Press SPACE to capture",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        preview,
                        "Looking for face... Please look at camera",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("LUMI Vision - Face Enrolment", preview)
                key = cv2.waitKey(1) & 0xFF

                if key == 32 and detections:  # SPACE bar
                    emb = recognizer.extract_embedding(clean_frame, detections[0])
                    database.enrol(name, emb, replace=(samples_collected == 0 and args.replace))
                    samples_collected += 1
                    logger.info(f"Collected sample {samples_collected}/{args.samples} for '{name}'")
                    time.sleep(0.3)
                elif key in (ord("q"), 27):
                    logger.info("Enrolment aborted by user.")
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


def cli_list(config: Config) -> None:
    """CLI sub-command to list all enrolled identities."""
    database = FaceDatabase(config.database.path)
    identities = database.list_identities()

    print("\n========================================================")
    print("           LUMI Vision - Enrolled Identities            ")
    print("========================================================")
    if not identities:
        print("  Database is empty. No identities currently enrolled.")
    else:
        print(f"  Total Enrolled: {len(identities)}\n")
        print(f"  {'Name':<20} | {'Samples':<8} | {'Enrolled Date':<25}")
        print("  " + "-" * 58)
        for item in identities:
            print(f"  {item['name']:<20} | {item['num_samples']:<8} | {item['enrolled_at'][:19]:<25}")
    print("========================================================\n")


def cli_delete(args: argparse.Namespace, config: Config) -> None:
    """CLI sub-command to delete an identity."""
    database = FaceDatabase(config.database.path)
    name = args.name.strip()
    if database.delete_identity(name):
        print(f"\n[SUCCESS] Deleted identity '{name}' from database.\n")
    else:
        print(f"\n[NOT FOUND] Identity '{name}' was not found in database.\n")
        sys.exit(1)


def cli_benchmark(config: Config) -> None:
    """Benchmark inference latency and throughput for YuNet and SFace."""
    print("\n========================================================")
    print("         LUMI Vision - Hardware Inference Benchmark     ")
    print("========================================================")

    detector = FaceDetector(model_path=config.detection.model_path)
    recognizer = FaceRecognizer(model_path=config.recognition.model_path)

    # Test resolution
    w, h = config.camera.width, config.camera.height
    print(f"Resolution : {w}x{h}")
    print("Warming up inference engine...")

    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw synthetic face-like structure
    cv2.circle(test_img, (w // 2, h // 2), 70, (200, 200, 200), -1)
    cv2.circle(test_img, (w // 2 - 25, h // 2 - 20), 10, (50, 50, 50), -1)
    cv2.circle(test_img, (w // 2 + 25, h // 2 - 20), 10, (50, 50, 50), -1)
    cv2.circle(test_img, (w // 2, h // 2 + 5), 8, (50, 50, 50), -1)
    cv2.ellipse(test_img, (w // 2, h // 2 + 35), (25, 12), 0, 0, 180, (50, 50, 50), -1)

    for _ in range(5):
        detector.detect(test_img)

    iterations = 50
    print(f"Benchmarking face detection (YuNet) over {iterations} iterations...")
    det_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect(test_img)
        det_times.append((time.perf_counter() - t0) * 1000.0)

    avg_det = np.mean(det_times)
    fps_det = 1000.0 / avg_det if avg_det > 0 else 0

    print(f"  YuNet Detection Latency : {avg_det:.2f} ms (Min: {np.min(det_times):.2f} ms, Max: {np.max(det_times):.2f} ms)")
    print(f"  YuNet Detection Rate    : {fps_det:.1f} FPS")

    # Benchmark recognizer
    mock_face = np.array([100, 100, 150, 150, 130, 140, 180, 140, 155, 175, 140, 210, 175, 210, 0.95], dtype=np.float32)
    mock_det = FaceDetection(bbox=(100, 100, 150, 150), landmarks=np.zeros((5, 2), dtype=np.float32), score=0.95, raw_face=mock_face)

    print(f"\nBenchmarking face recognition (SFace alignCrop + feature) over {iterations} iterations...")
    rec_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        recognizer.extract_embedding(test_img, mock_det)
        rec_times.append((time.perf_counter() - t0) * 1000.0)

    avg_rec = np.mean(rec_times)
    fps_rec = 1000.0 / avg_rec if avg_rec > 0 else 0

    print(f"  SFace Feature Latency   : {avg_rec:.2f} ms (Min: {np.min(rec_times):.2f} ms, Max: {np.max(rec_times):.2f} ms)")
    print(f"  SFace Feature Rate      : {fps_rec:.1f} FPS")

    total_pipeline_ms = avg_det + avg_rec
    total_fps = 1000.0 / total_pipeline_ms if total_pipeline_ms > 0 else 0
    print("\n--------------------------------------------------------")
    print(f"  Combined Latency        : {total_pipeline_ms:.2f} ms")
    print(f"  Estimated Full Pipeline : {total_fps:.1f} FPS")
    print("========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="luminax-vision",
        description="LUMI Autonomous Companion Robot - Standalone Vision Module",
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to custom config YAML file.")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run live webcam detection & recognition")
    run_parser.add_argument("--mock-camera", action="store_true", help="Use synthetic mock camera instead of physical webcam")
    run_parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    run_parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (testing)")

    # Enrol command
    enrol_parser = subparsers.add_parser("enrol", help="Enrol a person's biometric identity")
    enrol_parser.add_argument("--name", "-n", type=str, required=True, help="Person's name")
    enrol_parser.add_argument("--image", "-i", type=str, default=None, help="Path to static image file")
    enrol_parser.add_argument("--samples", "-s", type=int, default=1, help="Number of samples to capture")
    enrol_parser.add_argument("--replace", action="store_true", help="Replace existing samples for this name")

    # List command
    subparsers.add_parser("list", help="List all enrolled identities")

    # Delete command
    del_parser = subparsers.add_parser("delete", help="Delete an enrolled identity")
    del_parser.add_argument("--name", "-n", type=str, required=True, help="Name of identity to remove")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Measure YuNet & SFace inference latency and FPS")

    args = parser.parse_args()

    config = load_config(args.config)

    cmd = args.command or "run"

    if cmd == "run":
        if getattr(args, "headless", False):
            config.display.show_window = False

        camera_override = None
        if getattr(args, "mock_camera", False):
            camera_override = MockCamera(
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
                max_frames=args.max_frames,
            )

        run_live_pipeline(config, camera_override=camera_override, max_frames=args.max_frames)
    elif cmd == "enrol":
        cli_enrol(args, config)
    elif cmd == "list":
        cli_list(config)
    elif cmd == "delete":
        cli_delete(args, config)
    elif cmd == "benchmark":
        cli_benchmark(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
