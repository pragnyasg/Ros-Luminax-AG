"""
Face recognizer module wrapping OpenCV's SFace (FaceRecognizerSF).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from luminax_vision.face_database import FaceDatabase
from luminax_vision.face_detector import FaceDetection

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """
    Result of a face recognition evaluation.

    NOTE: Similarity scores represent vector distances in latent space, NOT calibrated probabilities.
    Higher cosine score (closer to 1.0) indicates greater similarity.
    """
    name: str                        # Enrolled identity name, or "Unknown"
    score: float                     # Best match score (Cosine similarity or L2 distance)
    is_known: bool                   # True if best match satisfies threshold, False otherwise
    metric: str                      # "cosine" or "l2"
    threshold: float                 # Configured decision threshold applied
    embedding: Optional[np.ndarray] = None  # 128-d normalized feature vector


class FaceRecognizer:
    """
    Feature extractor and identity matcher using SFace via cv2.FaceRecognizerSF.

    Features:
    - 5-point landmark-guided face alignment (alignCrop) to 112x112 canonical pose.
    - Computes 128-dimensional L2-normalized deep feature embeddings.
    - Matches against multiple samples per person in FaceDatabase.
    - Supports both Cosine similarity and Euclidean L2 distance metrics.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        distance_metric: str = "cosine",
        cosine_threshold: float = 0.363,
        l2_threshold: float = 1.128,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"SFace face recognition model not found at: {self.model_path}. "
                "Run 'bash scripts/download_models.sh' to download it."
            )

        self.metric_name = distance_metric.lower().strip()
        if self.metric_name not in ("cosine", "l2"):
            raise ValueError(f"Invalid distance metric '{distance_metric}'. Must be 'cosine' or 'l2'.")

        self.cosine_threshold = float(cosine_threshold)
        self.l2_threshold = float(l2_threshold)

        try:
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.model_path),
                config="",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize cv2.FaceRecognizerSF with {self.model_path}: {exc}") from exc

        logger.info(
            f"FaceRecognizer initialized with SFace ({self.model_path.name}), "
            f"metric='{self.metric_name}', cosine_thresh={self.cosine_threshold}, l2_thresh={self.l2_threshold}"
        )

    def extract_embedding(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Align face patch and extract 128-dimensional normalized feature embedding.

        CRITICAL NOTE:
        Always pass the pristine, UNANNOTATED image frame to this method.
        Never draw bounding boxes or text on the image prior to extracting embeddings!

        Args:
            image: Clean source BGR image.
            detection: FaceDetection containing raw 15-element face landmark vector.

        Returns:
            (128,) float32 normalized NumPy embedding vector.
        """
        if image is None or image.size == 0:
            raise ValueError("Cannot extract embedding from empty image.")

        # SFace alignCrop uses the 5 landmarks in raw_face to crop and rotate to 112x112
        aligned_face = self._recognizer.alignCrop(image, detection.raw_face)
        feature = self._recognizer.feature(aligned_face)

        # Ensure flat float32 array
        feat_arr = np.asarray(feature, dtype=np.float32).flatten()
        # Normalize vector to unit sphere
        norm = np.linalg.norm(feat_arr)
        if norm > 0:
            feat_arr = feat_arr / norm

        return feat_arr

    def compute_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Compute similarity score between two embeddings according to configured metric.

        For 'cosine': returns cosine similarity (higher is more similar, typically in [-1, 1]).
        For 'l2': returns Euclidean distance (lower is more similar, >= 0).
        """
        feat_a = np.asarray(embedding_a, dtype=np.float32).flatten()
        feat_b = np.asarray(embedding_b, dtype=np.float32).flatten()

        if self.metric_name == "cosine":
            # For unit-normalized vectors: cosine_sim = dot(a, b)
            norm_a = np.linalg.norm(feat_a)
            norm_b = np.linalg.norm(feat_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(feat_a, feat_b) / (norm_a * norm_b))
        else:
            # L2 Euclidean distance
            return float(np.linalg.norm(feat_a - feat_b))

    def is_match(self, score: float) -> bool:
        """Determine if a score meets the recognition acceptance threshold."""
        if self.metric_name == "cosine":
            return score >= self.cosine_threshold
        else:
            return score <= self.l2_threshold

    def recognize(
        self,
        embedding: np.ndarray,
        database: Union[FaceDatabase, Dict[str, List[np.ndarray]]],
    ) -> RecognitionResult:
        """
        Compare candidate embedding against all enrolled identities in database.

        Args:
            embedding: Candidate (128,) float32 embedding vector.
            database: FaceDatabase instance or pre-extracted dictionary of names to embeddings.

        Returns:
            RecognitionResult with best match name or 'Unknown'.
        """
        if isinstance(database, FaceDatabase):
            known_dict = database.get_all_embeddings()
        else:
            known_dict = database

        if not known_dict:
            # Empty database: All detections are unknown
            default_score = 0.0 if self.metric_name == "cosine" else 999.0
            threshold = self.cosine_threshold if self.metric_name == "cosine" else self.l2_threshold
            return RecognitionResult(
                name="Unknown",
                score=default_score,
                is_known=False,
                metric=self.metric_name,
                threshold=threshold,
                embedding=embedding,
            )

        best_name = "Unknown"
        threshold = self.cosine_threshold if self.metric_name == "cosine" else self.l2_threshold

        if self.metric_name == "cosine":
            best_score = -1.0
            for name, samples in known_dict.items():
                for sample_emb in samples:
                    sim = self.compute_similarity(embedding, sample_emb)
                    if sim > best_score:
                        best_score = sim
                        if sim >= self.cosine_threshold:
                            best_name = name
            is_known = best_score >= self.cosine_threshold
        else:
            best_score = float("inf")
            for name, samples in known_dict.items():
                for sample_emb in samples:
                    dist = self.compute_similarity(embedding, sample_emb)
                    if dist < best_score:
                        best_score = dist
                        if dist <= self.l2_threshold:
                            best_name = name
            is_known = best_score <= self.l2_threshold

        if not is_known:
            best_name = "Unknown"

        return RecognitionResult(
            name=best_name,
            score=best_score,
            is_known=is_known,
            metric=self.metric_name,
            threshold=threshold,
            embedding=embedding,
        )

    def process_face(
        self,
        clean_image: np.ndarray,
        detection: FaceDetection,
        database: Union[FaceDatabase, Dict[str, List[np.ndarray]]],
    ) -> RecognitionResult:
        """
        Helper to extract embedding and recognize an identity in a single call.
        Always operates on clean_image before visual bounding boxes are drawn.
        """
        embedding = self.extract_embedding(clean_image, detection)
        return self.recognize(embedding, database)
