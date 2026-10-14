# LUMI Vision

> **Portable, High-Performance Computer Vision Module for the LUMI Autonomous Healthcare Companion Robot**

[![Ubuntu 24.04 LTS](https://img.shields.io/badge/Ubuntu-24.04%20LTS-E95420?logo=ubuntu\&logoColor=white)](https://ubuntu.com/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv\&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Target: Raspberry Pi 5](https://img.shields.io/badge/Target-Raspberry%20Pi%205%20(ARM64)-C51A4A?logo=raspberrypi\&logoColor=white)](https://www.raspberrypi.com/)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy%20Jalisco-22314E?logo=ros\&logoColor=white)](https://docs.ros.org/en/jazzy/)

---

## 1. Project Overview

**LUMI** is an open-source autonomous healthcare companion robot engineered to assist elderly individuals, post-operative recovering patients, and differently-abled people with daily living, companionship, medication reminders, and safety monitoring.

This repository implements **LUMI Vision**, the foundational perception and person-recognition module. It provides:

- **Anchor-free Face Detection (YuNet):** Real-time multi-face bounding box and 5-point facial landmark detection via OpenCV's `FaceDetectorYN`.
- **Deep Feature Embedding & Recognition (SFace):** Landmark-aligned face cropping and 128-dimensional L2-normalized feature representation via OpenCV's `FaceRecognizerSF`.
- **Local Biometric Identity Database:** Consent-based multi-sample enrollment with atomic writes, corrupted-state recovery, and strict on-device data retention.
- **Hardware-Decoupled Camera Abstraction:** Unified interface supporting laptop webcams, external USB video devices (`/dev/video*`), RTSP/video files, and a synthetic `MockCamera` for headless CI/CD.
- **Pure Standalone Architecture:** Runs directly without requiring ROS 2 or physical robot hardware, while providing a clear architecture and package skeleton for future ROS 2 Jazzy deployment on Raspberry Pi 5.

---

## 2. System Architecture

```
                                  +-----------------------+
                                  | Physical / Mock Camera |
                                  +-----------+-----------+
                                              |
                                     (Clean BGR Frame)
                                              |
                                              v
                                  +-----------------------+
                                  |      YuNet Detector   |
                                  |   (FaceDetectorYN)    |
                                  +-----------+-----------+
                                              |
                             Detections + 5-point Landmarks
                                              |
                                              v
+-----------------------+         +-----------------------+
|  Local Face Database  |<------->|    SFace Recognizer   |
| (data/identities.json)|         |   (FaceRecognizerSF)  |
+-----------------------+         +-----------+-----------+
                                              |
                                      Matched Identity
                                    or Explicit 'Unknown'
                                              |
                                              v
                                  +-----------------------+
                                  |   Display / ROS 2     |
                                  +-----------------------+
```

---

## 3. Directory Structure

```text
luminax-vision/
├── README.md                     # Comprehensive project documentation
├── LICENSE                       # MIT License
├── THIRD_PARTY_NOTICES.md        # Upstream notices, model cards, and citations
├── .gitignore                    # Privacy & binary exclusion rules
├── .gitattributes                # Enforces POSIX LF line endings
├── pyproject.toml                # Modern Python build & tool configuration
├── requirements.txt              # Pinned Python package dependencies
├── config/
│   ├── config.example.yaml       # Template configuration file
│   └── config.yaml               # Active local configuration (git-ignored)
├── models/
│   ├── README.md                 # Model cards, checksums, and license details
│   ├── face_detection_yunet_2023mar.onnx    # YuNet detector weights (downloaded)
│   └── face_recognition_sface_2021dec.onnx  # SFace recognizer weights (downloaded)
├── data/
│   ├── .gitkeep                  # Preserves directory in git
│   └── identities.json           # Local biometric database (git-ignored)
├── scripts/
│   ├── setup_ubuntu.sh           # Automated setup for Ubuntu 24.04 LTS & Pi 5
│   ├── download_models.sh        # Shell model downloader with SHA-256 validation
│   ├── download_models.py        # Python cross-platform model downloader
│   └── check_environment.sh      # System diagnostic & readiness checker
├── src/
│   └── luminax_vision/
│       ├── __init__.py           # Package exports
│       ├── __main__.py           # Entry point for 'python -m luminax_vision'
│       ├── camera.py             # BaseCamera, Camera, and MockCamera
│       ├── config.py             # YAML loader, validator, and dataclasses
│       ├── face_detector.py      # YuNet FaceDetectorYN wrapper
│       ├── face_recognizer.py    # SFace FaceRecognizerSF wrapper
│       ├── face_database.py      # Multi-sample local JSON identity database
│       └── main.py               # CLI interface and real-time execution loop
├── tests/
│   ├── test_camera.py            # Camera abstraction and error tests
│   ├── test_config.py            # Configuration loader and validator tests
│   ├── test_face_database.py     # Database CRUD, validation, and recovery tests
│   └── test_recognition.py       # Threshold, embedding, and inference tests
└── ros2_ws/
    └── src/
        └── luminax_vision_ros/
            ├── README.md         # ROS 2 Jazzy node architecture & message specs
            └── package.xml       # ROS 2 ament_python package specification
```

---

## 4. Quick Start Guide (Ubuntu 24.04 LTS)

### Prerequisites

- Ubuntu 24.04 LTS (x86_64 or aarch64 / Raspberry Pi 5)
- Python 3.12 (`python3.12` and `python3.12-venv`)
- Built-in webcam or external USB camera

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd luminax-vision
```

### Step 2: Automated Environment Setup
Run the setup script. It checks your system, creates an isolated virtual environment (`.venv`), installs pinned dependencies, generates `config/config.yaml`, and downloads & verifies the vision models:
```bash
bash scripts/setup_ubuntu.sh
```

### Step 3: Run Environment Diagnostics
Verify that your operating system, Python libraries, vision models, and camera permissions are operational:
```bash
bash scripts/check_environment.sh
```

### Step 4: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 5: Launch Real-Time Vision Application
```bash
python -m luminax_vision
```

*(On systems without a desktop display or webcam, test using mock camera: `python -m luminax_vision run --mock-camera`)*

---

## 5. Interactive Real-Time Controls

When running the live camera feed window:

| Key | Action | Description |
| :---: | :--- | :--- |
| `Q` or `ESC` | **Quit** | Cleanly terminates the pipeline and releases camera hardware |
| `E` | **Enrol Face** | Prompts in terminal for name to enrol the currently visible face |
| `L` | **List Identities** | Prints all currently enrolled identities to terminal |
| `D` | **Delete Identity** | Prompts in terminal for name of an identity to delete |
| `S` | **Toggle Landmarks** | Toggles 5-point facial landmark dots on/off |

---

## 6. Command-Line Interface (CLI)

The module provides full CLI management for headless servers, scripts, and robotic service wrappers:

### A. Live Camera Detection & Recognition
```bash
# Default live webcam feed
python -m luminax_vision run

# Run headless (no GUI window, logs to stdout)
python -m luminax_vision run --headless

# Run with synthetic mock camera (useful for automated testing)
python -m luminax_vision run --mock-camera --max-frames 60
```

### B. Consent-Based Identity Enrolment
```bash
# Interactive camera capture (captures 3 samples for pose/lighting diversity)
python -m luminax_vision enrol --name "Grandma" --samples 3

# Enrol from a static photograph
python -m luminax_vision enrol --name "Caregiver" --image /path/to/photo.jpg

# Replace existing biometric samples for an identity
python -m luminax_vision enrol --name "Grandma" --samples 3 --replace
```

### C. List Enrolled Identities
```bash
python -m luminax_vision list
```
*Output:*
```text
========================================================
           LUMI Vision - Enrolled Identities            
========================================================
  Total Enrolled: 2

  Name                 | Samples  | Enrolled Date            
  ----------------------------------------------------------
  Grandma              | 3        | 2026-10-14T07:45:10      
  Caregiver            | 1        | 2026-10-14T07:46:22      
========================================================
```

### D. Delete an Enrolled Identity
```bash
python -m luminax_vision delete --name "Grandma"
```

### E. Hardware Inference Benchmark
Measure inference latency (ms) and throughput (FPS) for YuNet and SFace on the host CPU:
```bash
python -m luminax_vision benchmark
```
*Sample Output:*
```text
YuNet Detection Latency : 19.46 ms (51.4 FPS)
SFace Feature Latency   : 23.37 ms (42.8 FPS)
Combined Latency        : 42.83 ms
Estimated Full Pipeline : 23.3 FPS
```

---

## 7. Configuration Guide

Configuration is managed via YAML (`config/config.yaml`). If `config/config.yaml` does not exist, it is automatically cloned from `config/config.example.yaml`:

```yaml
camera:
  index: 0          # Device index (0: built-in, 1: USB) or Linux path "/dev/video0"
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

display:
  show_window: true
  window_name: "LUMI Vision - Healthcare Companion"
  show_fps: true
  show_score: true
  show_landmarks: false
  box_color_known: [0, 220, 100]    # BGR Green for verified identities
  box_color_unknown: [0, 80, 240]   # BGR Red/Orange for unidentified persons
```

---

## 8. Deep Learning Models & Verification

The vision pipeline uses pre-trained ONNX models from the official OpenCV Zoo:

| Model | Purpose | Input | Output | Size | SHA-256 Checksum | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YuNet** | Face Detection | Dynamic BGR | Bounding boxes + 5 landmarks | 232.6 KB | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | Apache-2.0 |
| **SFace** | Face Recognition | 112×112 crop | 128-d L2 unit vector | 38.7 MB | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

Models are automatically fetched by `scripts/download_models.sh` or `scripts/download_models.py` and validated against these exact checksums before saving.

---

## 9. Automated Testing Suite

The project includes unit and integration tests covering:
- Configuration validation and range boundary checks
- Identity database persistence, multi-sample gallery, duplicate resolution, and corrupt-state recovery
- Recognition threshold verification using controlled synthetic embeddings (Cosine and L2)
- Camera abstraction and connection failure modes
- End-to-end YuNet and SFace model inference integration

Run the complete test suite:
```bash
pytest -v
```

All 24 automated tests pass in under 3 seconds.

---

## 10. Future Raspberry Pi 5 & ROS 2 Jazzy Roadmap

### Raspberry Pi 5 Target Specifications
- **Operating System:** Ubuntu 24.04 LTS (ARM64)
- **Middleware:** ROS 2 Jazzy Jalisco
- **Camera Device:** USB UVC Camera or Raspberry Pi Camera Module 3 (libcamera / v4l2)
- **Optimization Strategy:**
  - Leverage OpenCV NEON SIMD vectorization on the Cortex-A76 cores.
  - Maintain detection at 25–30 FPS on 640×480 frames.
  - Rate-limit feature extraction (SFace) to 5–10 Hz to prevent thermal throttling.

### ROS 2 Package Architecture
See [ros2_ws/src/luminax_vision_ros/README.md](file:///c:/Users/pragn/OneDrive/Desktop/All%20Desktop/S5/Ros%20Luminax%20-%20AG/ros2_ws/src/luminax_vision_ros/README.md) for detailed node architecture and message definitions.

---

## 11. Privacy, Ethics & Safety Guidelines

- **Strictly Local Biometrics:** Biometric embeddings and face data are stored exclusively in the local `data/identities.json` file. Data is never transmitted over the network or committed to Git.
- **Explicit Consent Required:** Only individuals who have given explicit verbal or written consent should be enrolled into the robot's identity database.
- **Informational Purpose Only:** Face recognition events are designed strictly to facilitate pleasant social interactions (e.g. customized greetings, reminding a known user about scheduled medication). Face recognition must **never** be treated as conclusive evidence for medical emergencies, fall events, or physical access control.

---

## 12. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Third-party components and models are subject to their respective licenses detailed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
