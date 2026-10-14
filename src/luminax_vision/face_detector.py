"""
Face detector module wrapping OpenCV's YuNet (FaceDetectorYN).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Represents a detected face with bounding box, landmarks, and confidence score."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: np.ndarray             # 5 landmark points: [[x1, y1], [x2, y2], ...] (right eye, left eye, nose, right mouth, left mouth)
    score: float                      # Confidence score in [0.0, 1.0]
    raw_face: np.ndarray              # Raw 15-element array required by SFace alignCrop


class FaceDetector:
    """
    Lightweight, high-performance face detector using YuNet via cv2.FaceDetectorYN.

    Features:
    - Completely decoupled from camera implementation; processes any valid BGR image.
    - Dynamically updates input size whenever image dimensions change.
    - Extracts 5 facial landmarks for alignment and normalization.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        score_threshold: float = 0.80,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YuNet face detection model not found at: {self.model_path}. "
                "Run 'bash scripts/download_models.sh' to download it."
            )

        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)

        self._current_size: Tuple[int, int] = (0, 0)  # (width, height)

        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(320, 320),  # Initial placeholder size, dynamically updated
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize cv2.FaceDetectorYN with {self.model_path}: {exc}") from exc

        logger.info(
            f"FaceDetector initialized with YuNet ({self.model_path.name}), "
            f"score_thresh={self.score_threshold}, nms_thresh={self.nms_threshold}"
        )

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a given BGR image.

        Args:
            image: BGR image as a uint8 NumPy array with shape (H, W, 3).

        Returns:
            List of FaceDetection objects, sorted by confidence descending.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            logger.warning("FaceDetector.detect received empty or invalid image.")
            return []

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be 3-channel BGR image, got shape {image.shape}")

        height, width = image.shape[:2]

        # Dynamically update detector input size when frame dimensions change
        if (width, height) != self._current_size:
            self._detector.setInputSize((width, height))
            self._current_size = (width, height)

        _, faces = self._detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        results: List[FaceDetection] = []
        for raw_face in faces:
            # Format of raw_face: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, w, h = int(raw_face[0]), int(raw_face[1]), int(raw_face[2]), int(raw_face[3])
            score = float(raw_face[14])

            # Landmarks (5 points: right eye, left eye, nose, right mouth corner, left mouth corner)
            landmarks = np.array([
                [raw_face[4], raw_face[5]],
                [raw_face[6], raw_face[7]],
                [raw_face[8], raw_face[9]],
                [raw_face[10], raw_face[11]],
                [raw_face[12], raw_face[13]],
            ], dtype=np.float32)

            results.append(
                FaceDetection(
                    bbox=(x, y, w, h),
                    landmarks=landmarks,
                    score=score,
                    raw_face=raw_face,
                )
            )

        # Sort detections by score descending
        results.sort(key=lambda d: d.score, reverse=True)
        return results

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[FaceDetection],
        draw_landmarks: bool = False,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes and optional landmarks on a copy of the image.

        Args:
            image: Source image (unmodified).
            detections: List of FaceDetection objects.
            draw_landmarks: Whether to draw the 5 facial landmark dots.
            color: BGR tuple for box color.
            thickness: Line thickness.

        Returns:
            Annotated copy of image.
        """
        output = image.copy()
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

            label = f"Face: {det.score:.2f}"
            cv2.putText(
                output,
                label,
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            if draw_landmarks:
                for pt in det.landmarks:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(output, (px, py), 2, (0, 255, 255), -1)

        return output
