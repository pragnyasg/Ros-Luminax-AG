# LUMI Vision & Fall Detection Engine

> **Autonomous AI-Powered Healthcare Companion Robot — Computer Vision, Pose Estimation, Fall Detection & Caregiver Notification System**

[![Ubuntu 24.04 LTS](https://img.shields.io/badge/Ubuntu-24.04%20LTS-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![Windows 11](https://img.shields.io/badge/Windows-11-0078D4?logo=windows&logoColor=white)](https://microsoft.com/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x%20%2F%204.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe Tasks](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker-00A67E?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%26%20WebSockets-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Target: Raspberry Pi 5](https://img.shields.io/badge/Target-Raspberry%20Pi%205%20(ARM64)-C51A4A?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy%20Jalisco-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. System Overview

**LUMI** is an open-source autonomous healthcare companion robot engineered to support elderly individuals, post-operative recovering patients, and differently-abled people with daily living, companionship, medication reminders, and continuous safety monitoring.

This repository implements the **LUMI Vision Core**, an integrated multi-modal perception and health safety pipeline. The system acquires video frames from webcams or robot head cameras, detects human bodies and faces, identifies recognized individuals using local on-device biometric galleries, tracks people persistently across time, calculates 3D body pose landmarks, classifies activity and posture, monitors monitored-person presence and disappearance, executes a deterministic multi-stage temporal fall-detection state machine, records emergency incidents to durable SQLite storage, and serves real-time alerts through a caregiver REST and WebSocket notification backend.

---

## 2. Implemented Features

1. **Hardware Camera Abstraction:** Unified video capture layer supporting DirectShow on Windows, V4L2 on Linux/Raspberry Pi 5, RTSP streams, and a synthetic `MockCamera` for headless CI testing.
2. **Camera Discovery & Troubleshooting:** Built-in hardware scanner (`luminax-vision list-cameras`) to probe available device indices, default resolutions, and capture rates.
3. **Face Detection (YuNet):** High-speed anchor-free face detector using OpenCV DNN `FaceDetectorYN`, extracting bounding boxes and 5-point facial landmarks.
4. **Face Recognition (SFace):** Landmark-aligned 112×112 canonical crops mapped into 128-dimensional deep feature embeddings using OpenCV `FaceRecognizerSF` with cosine similarity matching.
5. **Local Biometric Identity Database:** Consent-based multi-sample gallery with atomic file writes, checksum integrity, and local storage in `data/identities.json` (biometrics never leave the device).
6. **Multi-Person Tracking:** Centroid and IoU tracker maintaining persistent `track_id` assignments, calculating horizontal and vertical velocities, and managing occlusion tolerance.
7. **Spatial Identity Association:** Associates face recognition results with body tracks only when face detections spatially fall within the upper-body region of the tracked person. Unrecognized persons remain strictly `Unknown`.
8. **Pose Estimation (MediaPipe Tasks):** 33-point normalized 3D human body landmarks extracted via CPU XNNPACK delegate (`pose_landmarker_lite.task`), with support for synthetic `MockPoseEstimator`.
9. **Geometric Feature Calculation:** Computes torso inclination angles (vertical and horizontal), bounding-box aspect ratios ($H/W$), joint midpoints, vertical velocity $v_y$, and landmark visibility confidence.
10. **Activity & Posture Classification:** Real-time geometric and kinematic classifier identifying `standing`, `sitting`, `walking`, `lying_down`, `fall_suspected`, `fall_confirmed`, and `unknown`.
11. **Differentiates Non-Falls:** Distinguishes ordinary sitting, walking, floor exercises, bending, and intentional lying down from accidental uncontrolled falls.
12. **Temporal Fall-Detection State Machine:** Multi-stage state machine enforcing:
    $$\text{NORMAL} \xrightarrow[\text{rapid drop}]{\Delta y/\Delta t > v_{\text{thresh}}} \text{POSSIBLE\_FALL} \xrightarrow[\text{orientation shift}]{\text{horizontal}} \text{FALL\_SUSPECTED} \xrightarrow[\Delta t \ge 2.0\text{s}]{\text{persistent horizontal}} \text{FALL\_CONFIRMED} \rightarrow \text{ALERT\_SENT} \xrightarrow[\text{standing}]{\Delta t \ge 1.0\text{s}} \text{RECOVERED}$$
13. **Incident Deduplication & Cooldown:** One incident produces exactly one primary emergency alert; ongoing frames do not flood duplicate alerts; recovery enables future independent incident detection.
14. **Monitored-Person Presence & Disappearance:** Supervised tracking for a designated individual (e.g. `target_person: "Pragnya"`) or general monitoring mode:
    - `PRESENT`: Subject detected in current frame.
    - `TEMPORARILY_MISSING`: Brief occlusion ($< 2.0\text{s}$) tolerated without alerts.
    - `MISSING`: Sustained departure ($> 5.0\text{s}$) triggers `person_missing` caregiver alert.
    - `CAMERA_ERROR`: Camera failure strictly distinguished from patient departure.
15. **Emergency Event Manager:** Generates stable UUIDv4 event IDs, UTC ISO-8601 timestamps, posture, confidence, bounding boxes, diagnostic metrics, and persists events in SQLite (`data/events.db`).
16. **Incident Frame Snapshots:** Automatically saves annotated high-resolution JPEG frames for emergency incidents to `data/snapshots/{event_id}.jpg` with automatic retention pruning.
17. **Caregiver Notification Backend (FastAPI):**
    - `GET /health` — Subsystem operational health.
    - `GET /status` — Live telemetry (presence state, posture, active track, FPS, alerts).
    - `GET /events` — Query emergency events with filtering (`limit`, `event_type`, `unacknowledged_only`).
    - `GET /events/{event_id}` — Retrieve detailed event payload.
    - `POST /events/{event_id}/acknowledge` — Acknowledge event with caregiver name and notes.
    - `POST /monitoring/start` & `POST /monitoring/stop` — Runtime monitoring control.
    - `GET /snapshots/{filename}` — Protected incident snapshot retrieval.
    - `WS /ws/events` — Real-time bi-directional WebSocket event streaming.
18. **Webcam Visual Diagnostics:** High-contrast OpenCV GUI window showing bounding boxes, track IDs, recognized names, posture badges, anatomical skeleton lines, FPS counters, and flashing emergency alerts.
19. **ROS 2 Compatibility Bridge:** Decoupled data adapter (`ros2_bridge.py`) converting pure Python events and tracks to standard `vision_msgs/Detection2D` and custom `lumi_msgs` schemas.

---

## 3. System Architecture & Pipeline Data Flow

```text
                                  +-----------------------+
                                  | Physical / Mock Camera |
                                  +-----------+-----------+
                                              |
                                     (Clean BGR Frame)
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
         +-----------------------+                         +-----------------------+
         |    YuNet Detector     |                         |  MediaPipe Landmarker |
         |   (FaceDetectorYN)    |                         | (pose_landmarker.task)|
         +-----------+-----------+                         +-----------+-----------+
                     |                                                 |
         Face Bboxes & Landmarks                            33 3D Body Landmarks
                     |                                                 |
                     v                                                 v
         +-----------------------+                         +-----------------------+
         |   SFace Recognizer    |                         |   Person Tracker &    |
         |  (FaceRecognizerSF)   |                         |  Identity Association |
         +-----------+-----------+                         +-----------+-----------+
                     |                                                 |
             Identity Vector                                   Persistent Tracks
                     |                                                 |
                     +------------------------>+<----------------------+
                                               |
                                     (Track + Identity)
                                               |
                                               v
                                   +-----------------------+
                                   |  Activity & Posture   |
                                   |      Classifier       |
                                   +-----------+-----------+
                                               |
                                        Classified Posture
                                               |
                                               v
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
         +-----------------------+                           +-----------------------+
         | Temporal Fall Detector|                           |   Presence Monitor    |
         | (6-Stage State Machine|                           | (Present / Missing /  |
         |  & Deduplication)     |                           |  Camera Hardware Err) |
         +-----------+-----------+                           +-----------+-----------+
                     |                                                   |
                     +------------------------>+<------------------------+
                                               |
                                     Emergency / Telemetry
                                               |
                                               v
                                   +-----------------------+
                                   | Emergency Event Mgr   |
                                   | SQLite (events.db)    |
                                   | Snapshots (.jpg)      |
                                   +-----------+-----------+
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
         +-----------------------+                           +-----------------------+
         |  Webcam Diagnostic GUI|                           | FastAPI & WebSockets  |
         | (Overlays, Skeletons) |                           | (Caregiver App API)   |
         +-----------------------+                           +-----------------------+
```

---

## 4. Project Directory Structure

```text
Ros Luminax - AG/
├── README.md                     # Comprehensive architecture and operation guide
├── LICENSE                       # MIT License
├── THIRD_PARTY_NOTICES.md        # Upstream model licenses, citations, and notices
├── pyproject.toml                # Build configuration and dependencies
├── requirements.txt              # Pinned Python package requirements
├── run_windows.bat               # Windows quick-launch launcher
├── config/
│   ├── config.example.yaml       # Template configuration file
│   └── config.yaml               # Active configuration (git-ignored)
├── models/
│   ├── README.md                 # Model cards, checksums, and license details
│   ├── face_detection_yunet_2023mar.onnx    # YuNet detector weights (232.6 KB)
│   ├── face_recognition_sface_2021dec.onnx  # SFace recognizer weights (38.7 MB)
│   └── pose_landmarker_lite.task            # MediaPipe Pose Landmarker (5.7 MB)
├── data/
│   ├── identities.json           # Local biometric database (git-ignored)
│   ├── events.db                 # SQLite durable emergency event history
│   └── snapshots/                # Incident JPEG frames ({event_id}.jpg)
├── scripts/
│   ├── setup_ubuntu.sh           # Automated setup for Ubuntu 24.04 LTS & Pi 5
│   ├── download_models.sh        # Shell model downloader with SHA-256 validation
│   ├── download_models.py        # Python cross-platform model downloader
│   └── check_environment.sh      # System diagnostic & readiness checker
├── src/
│   └── luminax_vision/
│       ├── __init__.py           # Public module API exports
│       ├── __main__.py           # Entry point for 'python -m luminax_vision'
│       ├── camera.py             # BaseCamera, Camera, and MockCamera
│       ├── camera_utils.py       # Camera probing and device diagnostic enumeration
│       ├── config.py             # YAML configuration loader and validators
│       ├── face_detector.py      # YuNet FaceDetectorYN wrapper
│       ├── face_recognizer.py    # SFace FaceRecognizerSF wrapper
│       ├── face_database.py      # Local JSON multi-sample biometric gallery
│       ├── pose_estimator.py     # MediaPipe Pose Landmarker and MockPoseEstimator
│       ├── person_tracker.py     # Multi-person tracker with spatial face association
│       ├── activity_classifier.py# Posture classification (standing, sitting, fallen, etc.)
│       ├── fall_detector.py      # 6-stage temporal fall-detection state machine
│       ├── presence_monitor.py   # Person presence, disappearance, and camera error monitor
│       ├── event_manager.py      # SQLite event persistence and snapshot retention
│       ├── emergency_manager.py  # Compatibility re-export module
│       ├── api.py                # FastAPI REST endpoints and WebSocket broadcaster
│       ├── notification_api.py   # Compatibility re-export module
│       ├── ros2_bridge.py        # Transport-independent ROS 2 message serializer
│       └── main.py               # Live orchestration pipeline and CLI interface
├── tests/
│   ├── test_camera.py            # Camera abstraction and connection tests
│   ├── test_config.py            # Config validation and environment variable tests
│   ├── test_face_database.py     # Biometric database CRUD and atomic write tests
│   ├── test_recognition.py       # SFace embedding distance and threshold tests
│   ├── test_pose_estimator.py    # Pose feature geometry and invalid landmark tests
│   ├── test_activity_classifier.py# Posture classification rules and velocity tests
│   ├── test_fall_detector.py     # Temporal state machine, persistence, and recovery tests
│   ├── test_presence_monitor.py  # Absence, occlusion, and camera failure tests
│   ├── test_event_manager.py     # SQLite persistence, retrieval, and ack tests
│   ├── test_tracker.py           # Tracking continuity and spatial face association tests
│   ├── test_api.py               # FastAPI endpoints, auth, and WebSocket client tests
│   └── test_integration.py       # End-to-end pipeline execution with MockCamera
└── ros2_ws/
    └── src/
        └── luminax_vision_ros/   # ROS 2 Jazzy thin wrapper package
            ├── README.md         # ROS 2 node architecture and message definitions
            └── package.xml       # ROS 2 package manifest
```

---

## 5. Prerequisites & Operating System Requirements

- **Supported Operating Systems:**
  - Windows 11 / Windows 10 (64-bit)
  - Ubuntu 24.04 LTS (x86_64 or aarch64 / ARM64)
  - Raspberry Pi OS (64-bit Debian 12 Bookworm)
- **Python Version:** Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.10).
- **Hardware Requirements:**
  - **Development Laptop:** Intel Core i5 / AMD Ryzen 5 or higher, 8 GB RAM, built-in or USB webcam.
  - **Deployment Target (Raspberry Pi 5):** 4 GB or 8 GB RAM, active cooler/heatsink, USB UVC camera or Raspberry Pi Camera Module 3 via MIPI-CSI.

---

## 6. Environment Setup & Dependency Installation

### Step 1: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / Ubuntu / Raspberry Pi 5:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Pinned Core Dependencies

| Package | Version Range | Purpose |
| :--- | :--- | :--- |
| `opencv-python` | `>=4.8.0` | Video frame capture, image processing, YuNet, SFace |
| `numpy` | `>=1.24.0,<2.3.0` | Fast multidimensional vector and landmark mathematics |
| `mediapipe` | `>=0.10.14` | Pose Landmarker Tasks on CPU via XNNPACK delegate |
| `fastapi` | `>=0.110.0` | Caregiver HTTP REST API framework |
| `uvicorn` | `>=0.28.0` | Asynchronous ASGI production server |
| `pydantic` | `>=2.6.0` | Request and response schema validation |
| `websockets` | `>=12.0` | Real-time bi-directional event streaming |
| `httpx` | `>=0.27.0` | Async HTTP client for automated API integration tests |
| `PyYAML` | `>=6.0.1` | YAML configuration file parsing and writing |
| `pytest` | `>=7.4.0` | Automated unit, regression, and integration testing |

---

## 7. Model Asset Verification & Download

LUMI Vision uses 3 pre-trained neural network assets:

| Model | Purpose | Framework | Input Resolution | Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YuNet** | Face Detection | ONNX (OpenCV Zoo) | Dynamic ($640\times 480$) | 232.6 KB | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| **SFace** | Face Recognition | ONNX (OpenCV Zoo) | $112\times 112$ RGB | 38.7 MB | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| **Pose Landmarker** | 33-point Body Pose | TFLite Task (Google) | Dynamic ($256\times 256$) | 5.7 MB | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

Download and verify all model files with a single command:
```bash
python scripts/download_models.py
```

*Output:*
```text
====================================================
      LUMI Vision - Model Download & Verification   
====================================================
Checking YuNet Face Detector (face_detection_yunet_2023mar.onnx)...
  [OK] Already present and verified: face_detection_yunet_2023mar.onnx

Checking SFace Face Recognizer (face_recognition_sface_2021dec.onnx)...
  [OK] Already present and verified: face_recognition_sface_2021dec.onnx

Checking MediaPipe Pose Landmarker (Lite) (pose_landmarker_lite.task)...
  [OK] Already present and verified: pose_landmarker_lite.task

All required models downloaded and verified successfully!
```

---

## 8. Camera Selection & Troubleshooting

Discover connected camera devices and negotiated capabilities:
```bash
python -m luminax_vision list-cameras
```

*Sample Output:*
```text
========================================================
           LUMI Vision - Camera Device Discovery        
========================================================
Scanning available video capture devices on system...

  Found 1 accessible camera device(s):

  Index/Path   | Backend        | Resolution   | FPS    | Status
  ------------------------------------------------------------------
  0            | DirectShow     | 640x480      | 30.0   | Ready (Frame capture verified)
========================================================
```

### Camera Troubleshooting Matrix

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `ConnectionError: Camera device '0' could not be opened` | Another application (Zoom, Teams, Chrome) is locking camera | Close other apps locking the camera, or set `camera.index: 1` in `config/config.yaml`. |
| Windows MSMF async grab error `-1072875772` | Windows Media Foundation asynchronous deadlock | The `Camera` class automatically forces DirectShow (`cv2.CAP_DSHOW`) on Windows. |
| Linux `Permission denied: '/dev/video0'` | User account missing `video` group permissions | Run `sudo usermod -a -G video $USER`, then log out and back in. |
| Low framerate on laptop | Resolution set higher than sensor capability | Ensure resolution in `config/config.yaml` is set to `width: 640`, `height: 480`. |

---

## 9. Configuration Guide

System configuration is defined in `config/config.yaml` with automatic fallback to `config/config.example.yaml`:

```yaml
camera:
  index: 0          # Device index (0: built-in webcam, 1/2: USB camera)
  width: 640        # Capture width
  height: 480       # Capture height
  fps: 30           # Target frame rate

detection:
  model_path: "models/face_detection_yunet_2023mar.onnx"
  score_threshold: 0.80   # YuNet confidence threshold (0.0 - 1.0)
  nms_threshold: 0.30     # Non-maximum suppression threshold
  top_k: 5000

recognition:
  model_path: "models/face_recognition_sface_2021dec.onnx"
  distance_metric: "cosine"   # "cosine" (recommended) or "l2"
  cosine_threshold: 0.363     # Similarity >= 0.363 is considered a match
  l2_threshold: 1.128         # Distance <= 1.128 is considered a match

database:
  path: "data/identities.json"

pose:
  model_path: "models/pose_landmarker_lite.task"
  min_detection_confidence: 0.50
  min_tracking_confidence: 0.50
  num_poses: 4
  show_skeleton: true

tracking:
  max_disappeared: 15     # Consecutive missed frames before deregistering track
  max_distance: 0.35      # Max normalized centroid displacement for matching
  min_iou: 0.15

fall_detection:
  enabled: true
  rapid_velocity_threshold: 0.55   # Downward velocity (norm Y/sec) triggering suspicion
  transition_window_sec: 1.5       # Window between descent and horizontal settling
  persistence_duration_sec: 2.0    # Required horizontal duration to confirm fall
  alert_cooldown_sec: 30.0         # Alert cooldown for continuous incident
  recovery_duration_sec: 1.0       # Upright duration required to confirm recovery

monitoring:
  target_person: "Pragnya"         # Specific enrolled identity to monitor, or null for general mode
  occlusion_tolerance_sec: 2.0     # Brief occlusion tolerance
  missing_timeout_sec: 5.0         # Duration absent triggering person_missing alert
  alert_on_missing: true

events:
  db_path: "data/events.db"
  snapshots_dir: "data/snapshots"
  save_snapshots: true
  max_snapshots: 500

server:
  enabled: true
  host: "127.0.0.1"        # 127.0.0.1 for local development; 0.0.0.0 for LAN
  port: 8000
  api_key: null            # Optional security token (e.g. "lumi-secret-key")
  cors_origins: ["*"]

display:
  show_window: true
  window_name: "LUMI Vision - Healthcare Companion"
  show_fps: true
  show_score: true
  show_landmarks: false
```

### Environment Variable Overrides

Configuration settings can be dynamically overridden in containerized or headless robotic environments:

| Environment Variable | Description | Example |
| :--- | :--- | :--- |
| `LUMI_VISION_CONFIG` | Path to custom YAML configuration file | `LUMI_VISION_CONFIG=/etc/lumi/vision.yaml` |
| `LUMI_CAMERA_INDEX` | Camera device index or video device path | `LUMI_CAMERA_INDEX=1` or `LUMI_CAMERA_INDEX=/dev/video2` |
| `LUMI_TARGET_PERSON` | Designated person identity to track | `LUMI_TARGET_PERSON="Pragnya"` |
| `LUMI_API_HOST` | Caregiver notification API listen host | `LUMI_API_HOST="0.0.0.0"` |
| `LUMI_API_PORT` | Caregiver notification API listen port | `LUMI_API_PORT=8080` |
| `LUMI_API_KEY` | Secret authentication token | `LUMI_API_KEY="super-secret-token"` |
| `LUMI_SAVE_SNAPSHOTS`| Boolean toggle for incident frame snapshots | `LUMI_SAVE_SNAPSHOTS="true"` |

---

## 10. Running the Application

### A. Run Live Vision Pipeline (Webcam + Fall Detection + API Server)

**Windows:**
Double-click `run_windows.bat` or run in terminal:
```powershell
.\.venv\Scripts\python -m luminax_vision run
```

**Linux / Ubuntu / Raspberry Pi 5:**
```bash
python3 -m luminax_vision run
```

### B. Interactive Keyboard Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| `Q` or `ESC` | **Quit** | Cleanly terminates pipeline, closes camera, shuts down API server |
| `E` | **Enrol Face** | Prompts in terminal for name to enrol currently visible face |
| `L` | **List Identities** | Prints all currently enrolled identities to terminal |
| `D` | **Delete Identity** | Prompts in terminal for name of an identity to delete |
| `S` | **Toggle Skeleton** | Toggles anatomical body skeleton lines and landmarks overlay |
| `M` | **Cycle Target** | Toggles monitoring between enrolled individuals and General Mode |
| `A` | **Acknowledge** | Acknowledges active emergency alerts directly from keyboard |

### C. Run Headless / Background Mode (Robotic Daemon)

To run without opening an OpenCV GUI window:
```bash
python -m luminax_vision run --headless
```

### D. Run with Synthetic Mock Camera & Pose

To test the complete processing loop without a physical camera connected:
```bash
python -m luminax_vision run --mock-camera --mock-pose --headless --max-frames 100
```

### E. Run Standalone Caregiver Notification Server

To launch the FastAPI / WebSocket server independently:
```bash
python -m luminax_vision server
```

---

## 11. Caregiver Notification API & WebSocket Interface

The notification backend runs on `http://127.0.0.1:8000` (or configured host/port).

### Interactive OpenAPI Documentation
Visit `http://127.0.0.1:8000/docs` in your browser for the full Swagger interactive API documentation.

### 1. Health Status: `GET /health`
```bash
curl -X GET http://127.0.0.1:8000/health
```
*Response (`200 OK`):*
```json
{
  "status": "healthy",
  "timestamp": "2026-10-22T09:15:30.123456+00:00",
  "version": "0.1.0",
  "subsystems": {
    "database": "healthy",
    "event_manager": "healthy",
    "notification_server": "healthy",
    "camera": "healthy",
    "pose_estimator": "healthy"
  }
}
```

### 2. Live Telemetry: `GET /status`
```bash
curl -X GET http://127.0.0.1:8000/status
```
*Response (`200 OK`):*
```json
{
  "monitoring_active": true,
  "presence_state": "PRESENT",
  "target_person": "Pragnya",
  "active_track_id": 1,
  "current_posture": "standing",
  "last_seen_timestamp": "2026-10-22T09:15:28Z",
  "absence_duration_sec": 0.0,
  "unacknowledged_alerts": 0,
  "camera_fps": 29.4,
  "system_time": "2026-10-22T09:15:30.450120+00:00"
}
```

### 3. List Emergency Events: `GET /events`
Query parameters:
- `limit`: int (default `50`, range `1-500`)
- `event_type`: string (optional: `fall_confirmed`, `fall_suspected`, `person_missing`, `person_reappeared`, `camera_error`)
- `severity`: string (optional: `critical`, `warning`, `info`)
- `unacknowledged_only`: bool (optional: `true`/`false`)

```bash
curl -X GET "http://127.0.0.1:8000/events?limit=5&unacknowledged_only=true"
```
*Response (`200 OK`):*
```json
[
  {
    "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "event_type": "fall_confirmed",
    "severity": "critical",
    "timestamp": "2026-10-22T09:14:10.000000+00:00",
    "person_name": "Pragnya",
    "track_id": 1,
    "confidence": 0.95,
    "posture": "lying_down",
    "last_seen_timestamp": null,
    "duration_seconds": 2.1,
    "bbox": [140, 280, 360, 150],
    "snapshot_path": "snapshots/a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg",
    "robot_location": null,
    "source_component": "temporal_fall_detector",
    "details": {
      "torso_angle": 86.4,
      "aspect_ratio": 0.42,
      "velocity_y": 0.78
    },
    "acknowledged": false,
    "acknowledged_at": null,
    "acknowledged_by": null
  }
]
```

### 4. Acknowledge Event: `POST /events/{event_id}/acknowledge`
```bash
curl -X POST http://127.0.0.1:8000/events/a1b2c3d4-e5f6-7890-abcd-ef1234567890/acknowledge \
  -H "Content-Type: application/json" \
  -d '{"acknowledged_by": "Dr. Sarah Caregiver", "note": "Checked on patient. Patient is safe and unhurt."}'
```
*Response (`200 OK`):*
```json
{
  "success": true,
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "acknowledged_by": "Dr. Sarah Caregiver",
  "acknowledged_at": "2026-10-22T09:16:02.123456+00:00"
}
```

### 5. Runtime Monitoring Controls: `POST /monitoring/start` & `stop`
```bash
# Switch monitored person dynamically
curl -X POST http://127.0.0.1:8000/monitoring/start \
  -H "Content-Type: application/json" \
  -d '{"target_person": "Pragnya"}'

# Pause monitoring
curl -X POST http://127.0.0.1:8000/monitoring/stop
```

### 6. Real-Time WebSocket Streaming: `WS /ws/events`

Clients connect via WebSocket to `ws://127.0.0.1:8000/ws/events`.

**Welcome Handshake:**
```json
{
  "type": "connection_established",
  "timestamp": "2026-10-22T09:15:00.000000+00:00",
  "unacknowledged_alerts": 1
}
```

**Real-Time Broadcast Payload on Confirmed Fall:**
```json
{
  "type": "emergency_event",
  "event": {
    "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "event_type": "fall_confirmed",
    "severity": "critical",
    "timestamp": "2026-10-22T09:14:10.000000+00:00",
    "person_name": "Pragnya",
    "track_id": 1,
    "confidence": 0.95,
    "posture": "lying_down",
    "snapshot_path": "snapshots/a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg",
    "details": {
      "torso_angle": 86.4,
      "aspect_ratio": 0.42,
      "velocity_y": 0.78
    }
  }
}
```

**WebSocket Client Example (Python):**
```python
import asyncio
import json
import websockets

async def listen_caregiver_stream():
    uri = "ws://127.0.0.1:8000/ws/events"
    async with websockets.connect(uri) as ws:
        print("Connected to LUMI Caregiver WebSocket Stream.")
        while True:
            raw_msg = await ws.recv()
            event = json.loads(raw_msg)
            print(f"[CAREGIVER ALERT RECEIVED]: {event}")

asyncio.run(listen_caregiver_stream())
```

---

## 12. Automated Testing Suite

Automated testing is mandatory and strictly verified. The project includes a complete test suite covering:

- **Pose & Posture Geometry (`test_pose_estimator.py`):** Complete landmark calculation, invalid landmark rejection, aspect ratio, angle calculations.
- **Activity Classification (`test_activity_classifier.py`):** Standing, sitting, walking, intentional resting, and velocity triggers.
- **Temporal Fall Detection (`test_fall_detector.py`):** Rapid drop, horizontal persistence duration, single-frame rejection, incident deduplication, and recovery.
- **Presence Monitoring (`test_presence_monitor.py`):** Occlusion tolerance, sustained absence, reappearance, camera failure isolation.
- **Event Manager & SQLite (`test_event_manager.py`):** UUID generation, timestamp formatting, SQL queries, acknowledgement, snapshot retention pruning.
- **Caregiver API & WebSockets (`test_api.py`):** Health, telemetry, event CRUD, authentication rejection, WebSocket streaming.
- **Person Tracking (`test_tracker.py`):** Persistent track IDs, IoU overlap, spatial face association, unknown identity preservation.
- **Camera Abstraction (`test_camera.py`):** Physical camera connection error handling, synthetic `MockCamera` streaming.
- **End-to-End Pipeline (`test_integration.py`):** Full multi-frame execution with mock sensors and simulated incidents.

### Run Test Suite

```bash
pytest -v
```

**Actual Execution Result:**
```text
tests/test_activity_classifier.py ......                                 [  9%]
tests/test_api.py .....                                                  [ 17%]
tests/test_camera.py ....                                                [ 23%]
tests/test_config.py ......                                              [ 32%]
tests/test_event_manager.py ....                                         [ 39%]
tests/test_face_database.py ........                                     [ 51%]
tests/test_fall_detector.py ......                                       [ 60%]
tests/test_integration.py ..                                             [ 64%]
tests/test_pose_estimator.py ......                                      [ 73%]
tests/test_presence_monitor.py .......                                   [ 84%]
tests/test_recognition.py ......                                         [ 93%]
tests/test_tracker.py ....                                               [100%]

======================== 64 passed in 4.10s ========================
```

---

## 13. Inspecting Logs & Event History

### Inspect Recorded Emergency Events via CLI
```bash
python -m luminax_vision events --limit 20
```

*Output:*
```text
========================================================
           LUMI Vision - Recent Emergency Events        
========================================================
  Total Retrievable Events: 2

  Timestamp            | Type             | Severity | Person       | Ack
  ------------------------------------------------------------------
  2026-10-22T09:14:10  | fall_confirmed   | critical | Pragnya      | NO
  2026-10-22T08:52:14  | person_missing   | warning  | Pragnya      | YES
========================================================
```

### Inspect Database Directly via SQLite CLI
```bash
sqlite3 data/events.db "SELECT event_id, event_type, severity, person_name, timestamp, acknowledged FROM events ORDER BY timestamp DESC LIMIT 5;"
```

---

## 14. Raspberry Pi 5 Migration & Optimization

The entire vision pipeline was developed on a laptop with strict modularity for direct deployment onto a **Raspberry Pi 5 (8 GB RAM)**:

### Optimization Strategy for Raspberry Pi 5
1. **CPU Acceleration (XNNPACK / ARM NEON):**
   - MediaPipe PoseLandmarker automatically creates and leverages the ARM NEON TFLite XNNPACK delegate on aarch64.
   - OpenCV DNN YuNet leverages ARM NEON SIMD optimizations built into Ubuntu 24.04 aarch64.
2. **Decoupled Processing Rates:**
   - Face detection and body pose estimation run on every frame at $640\times 480$.
   - SFace biometric feature embedding extraction (128-d vector) is only computed when a new track is initialized or when an unidentified face is detected, conserving CPU cycles.
3. **Bounded Queues & Drop Frames:**
   - Camera frame reads operate synchronously with transient retries to avoid unbounded memory buffering.
4. **Thermal Stability:**
   - Active cooling is recommended on the Raspberry Pi 5. If CPU temperatures exceed $75^\circ\text{C}$, framerate can be adjusted in `config/config.yaml` (`camera.fps: 20`).

---

## 15. ROS 2 Integration Architecture

The LUMI Vision pipeline maintains a zero-dependency boundary from ROS 2 while providing the exact message translations in `src/luminax_vision/ros2_bridge.py`.

When deployed inside `ros2_ws/src/luminax_vision_ros`, the thin wrapper node subscribes to and publishes standard ROS 2 topics:

### Subscribed Topics
- `/camera/image_raw` (`sensor_msgs/msg/Image`) — Head camera feed.
- `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`) — Intrinsic camera parameters.

### Published Topics
- `/vision/tracks` (`vision_msgs/msg/Detection2DArray`) — Tracked person bounding boxes, identities, and postures.
- `/vision/events/fall` (`lumi_msgs/msg/FallAlert`) — Confirmed emergency fall events.
- `/vision/presence` (`lumi_msgs/msg/PresenceStatus`) — Patient presence and visibility status.
- `/vision/image_annotated` (`sensor_msgs/msg/Image`) — Diagnostic video stream for robot web UI.

---

## 16. Medical Safety, Reliability & Limitations

> [!WARNING]
> **Research and Healthcare Prototype Notice:**
> LUMI is an autonomous robotic companion prototype, NOT a certified medical device (FDA Class II/III or CE-MDR).
> Monocular RGB webcam computer vision is subject to optical limitations:
> - **Occlusion:** Large furniture, thick blankets, or walls blocking the camera view will impede landmark detection.
> - **Perspective & Lens Distortion:** Camera angles looking sharply downward from high ceilings or angled obliquely from low tables alter vertical projection geometry.
> - **Lighting Conditions:** Severe backlighting, complete darkness, or strobe lighting degrade landmark visibility scores.
> - **Medical Diagnosis:** The system detects kinematic movement and geometry, NOT medical conditions, concussions, or vitals.
> - **Human Supervision:** Caregiver notification outputs must complement, never replace, trained human nursing oversight and emergency services.

---

## 17. Biometric Privacy & Data Retention

- **Local-Only Biometrics:** Face embeddings and personal identities are stored exclusively on the robot's local filesystem (`data/identities.json`). No biometric vectors are ever transmitted across external cloud networks.
- **Consent-Driven Enrollment:** The system provides consent-based enrollment tools (`luminax-vision enrol`) and explicit deletion commands (`luminax-vision delete`).
- **Snapshot Retention Limit:** Incident frame captures are stored locally in `data/snapshots/` and strictly bounded by `events.max_snapshots: 500`. Old snapshots are automatically purged when the threshold is reached.
- **API Security:** The caregiver notification server binds to `127.0.0.1` by default and supports configurable `api_key` authentication.

---

## 18. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Third-party models and components are licensed under Apache 2.0 as detailed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
