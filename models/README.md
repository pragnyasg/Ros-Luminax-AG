# LUMI Vision Pretrained Models

This directory stores the deep learning model weights used by LUMI Vision for real-time face detection and recognition.

Model weights (`*.onnx`) are excluded from Git to prevent repository bloat and ensure clean cloning. They are automatically downloaded and verified by `scripts/download_models.sh`.

---

## 1. YuNet (Face Detection)

- **Filename:** `face_detection_yunet_2023mar.onnx`
- **Architecture:** YuNet (lightweight anchor-free CNN from `libfacedetection`)
- **Author:** Shiqi Yu and contributors
- **Upstream Repository:** [OpenCV Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Hugging Face Model Card:** [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet)
- **License:** Apache License 2.0
- **Size:** ~232.6 KB (232,589 bytes)
- **SHA-256 Checksum:** `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- **Input:** BGR image (dynamically sized)
- **Output:** 15 elements per detected face:
  - `[0:4]`: Bounding box `[x, y, w, h]`
  - `[4:14]`: 5 facial landmarks `[right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y]`
  - `[14]`: Detection confidence score

---

## 2. SFace (Face Recognition)

- **Filename:** `face_recognition_sface_2021dec.onnx`
- **Architecture:** SFace (ResNet-like deep feature extractor with sphere/cosine margin)
- **Paper:** *SphereFace2: Binary Classification is All You Need for Deep Face Recognition* (Zhong et al.)
- **Upstream Repository:** [OpenCV Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- **Hugging Face Model Card:** [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface)
- **License:** Apache License 2.0
- **Size:** ~38.7 MB (38,696,353 bytes)
- **SHA-256 Checksum:** `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- **Input:** 112×112 aligned RGB face patch
- **Output:** 128-dimensional normalized feature embedding vector
- **Matching Metric:**
  - Cosine Similarity: `match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)` (threshold: ~0.363)
  - Norm L2 Distance: `match(feat1, feat2, cv2.FaceRecognizerSF_FR_NORM_L2)` (threshold: ~1.128)

---

## Manual Download Instructions

If the automated script cannot reach Hugging Face directly, download manually into this directory:

```bash
# YuNet Face Detector
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx

# SFace Face Recognizer
curl -L -o models/face_recognition_sface_2021dec.onnx \
  https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx
```

Verify checksums:
```bash
sha256sum models/*.onnx
```
