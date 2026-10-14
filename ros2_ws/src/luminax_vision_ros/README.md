# LUMI Vision - ROS 2 Jazzy Architecture & Integration Plan

This directory outlines the future integration of the standalone LUMI Vision module into the ROS 2 Jazzy ecosystem running on the Raspberry Pi 5.

---

## 1. Architectural Philosophy

- **Decoupled Architecture:** The core computer vision module (`luminax_vision`) remains a standalone, pure Python library with zero mandatory ROS 2 dependencies.
- **Thin ROS 2 Wrapper:** The `luminax_vision_ros` package acts purely as an adapter, translating ROS 2 camera messages (`sensor_msgs/msg/Image`) into `numpy.ndarray` frames, passing them through `FaceDetector` and `FaceRecognizer`, and publishing standard or custom ROS 2 messages.
- **Safety Boundary:** Biometric recognition is an informational input for social interaction (greeting, medication prompts). It must **never** be directly wired to motor actuation, medical decisions, or physical access locks without multi-factor authorization.

---

## 2. Node Specification: `lumi_vision_node`

### Subscribed Topics
| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Live video frames from robot's head camera (USB / MIPI-CSI) |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Intrinsic calibration parameters |

### Published Topics
| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/vision/faces/detections` | `vision_msgs/msg/Detection2DArray` | Standard ROS 2 face bounding boxes & landmarks |
| `/vision/faces/recognized` | `lumi_msgs/msg/RecognizedPersonArray` (or JSON string) | Recognized identity events with confidence scores |
| `/vision/faces/image_annotated` | `sensor_msgs/msg/Image` | Annotated visual stream for diagnostic GUI or web dashboard |

### Provided Services
| Service | Service Type | Description |
| :--- | :--- | :--- |
| `/vision/faces/enrol` | `lumi_msgs/srv/EnrolPerson` | Dynamically enrol current face into local identity database |
| `/vision/faces/delete` | `lumi_msgs/srv/DeletePerson` | Delete identity from database |
| `/vision/faces/list` | `lumi_msgs/srv/ListPersons` | Return list of enrolled names and sample counts |

---

## 3. Recommended ROS 2 Message Definitions

### `lumi_msgs/msg/RecognizedPerson.msg`
```text
# Timestamp and coordinate frame
std_msgs/Header header

# Unique detection tracker ID
int32 detection_id

# Bounding box in pixel coordinates (x, y, width, height)
int32 x
int32 y
int32 width
int32 height

# Identity label ("Grandma", "Caregiver", or "Unknown")
string name

# Similarity metric ("cosine" or "l2")
string metric

# Similarity score (latent space distance, NOT calibrated probability)
float32 score

# Boolean flag whether score satisfied acceptance threshold
bool is_known
```

### `lumi_msgs/msg/RecognizedPersonArray.msg`
```text
std_msgs/Header header
lumi_msgs/RecognizedPerson[] persons
```

---

## 4. Raspberry Pi 5 Optimization Strategy

1. **Downsampled Inference:**
   - Robot camera operates at 1080p for navigation / streaming.
   - Downsample frames to 640×480 or 320×240 for YuNet face detection to maintain 25+ FPS without thermal throttling.
2. **Decoupled Recognition Rate:**
   - Detect faces on every frame (30 Hz).
   - Compute SFace 128-d embeddings at a lower frequency (e.g. 5–10 Hz) or only upon new track acquisition to save ARM CPU cycles.
3. **OpenCV DNN Acceleration:**
   - Leverage ARM NEON optimizations in OpenCV on Ubuntu 24.04 aarch64.
