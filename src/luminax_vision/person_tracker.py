"""
Person tracking and identity association module for LUMI Vision.

Maintains persistent track IDs across frames using spatial proximity and IoU matching,
computes dynamic velocity and displacement, and associates biometric face recognition
results with the appropriate tracked person based on spatial containment.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from luminax_vision.face_detector import FaceDetection
from luminax_vision.face_recognizer import RecognitionResult
from luminax_vision.pose_estimator import PoseFeatures, PoseResult

logger = logging.getLogger(__name__)


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) of two boxes in (x, y, w, h) format."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    yb = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = float(area_a + area_b - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class TrackSnapshot:
    """Historical snapshot of a person track at a single timestamp."""
    timestamp: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    torso_center: Optional[Tuple[float, float]]
    torso_angle_deg: Optional[float]
    aspect_ratio: float


@dataclass
class PersonTrack:
    """
    State and historical record for a tracked individual.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixels
    bbox_normalized: Tuple[float, float, float, float]  # (nx, ny, nw, nh)
    centroid: Tuple[float, float]  # (normalized_cx, normalized_cy)
    pose: Optional[PoseResult] = None

    # Biometric identity details
    identity_name: str = "Unknown"
    identity_score: float = 0.0
    is_known_identity: bool = False
    identity_updated_at: float = 0.0

    # Tracking metrics
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    consecutive_frames_seen: int = 1
    consecutive_misses: int = 0

    # Motion and velocity (normalized coordinates per second, positive Y downwards)
    velocity_y: float = 0.0
    velocity_x: float = 0.0

    # Short history buffer for temporal fall analysis
    history: Deque[TrackSnapshot] = field(default_factory=lambda: collections.deque(maxlen=60))

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        bbox_normalized: Tuple[float, float, float, float],
        centroid: Tuple[float, float],
        pose: Optional[PoseResult],
        current_time: float,
    ) -> None:
        """Update track with new frame observations and calculate velocity."""
        dt = current_time - self.last_seen_timestamp

        if dt > 1e-4:
            # Velocity = delta_pos / dt
            self.velocity_x = (centroid[0] - self.centroid[0]) / dt
            self.velocity_y = (centroid[1] - self.centroid[1]) / dt

        self.bbox = bbox
        self.bbox_normalized = bbox_normalized
        self.centroid = centroid
        self.pose = pose
        self.last_seen_timestamp = current_time
        self.consecutive_frames_seen += 1
        self.consecutive_misses = 0

        torso_c = pose.features.torso_center if (pose and pose.features) else None
        torso_a = pose.features.torso_angle_deg if (pose and pose.features) else None
        ar = pose.features.aspect_ratio if (pose and pose.features) else 1.0

        self.history.append(
            TrackSnapshot(
                timestamp=current_time,
                centroid=centroid,
                bbox=bbox,
                torso_center=torso_c,
                torso_angle_deg=torso_a,
                aspect_ratio=ar,
            )
        )

    def mark_missed(self) -> None:
        """Increment miss counter on frame where person was not matched."""
        self.consecutive_misses += 1

    def associate_identity(
        self,
        name: str,
        score: float,
        is_known: bool,
        current_time: float,
    ) -> None:
        """Update recognized identity if match confidence is higher or matches verified identity."""
        if is_known:
            # Verified enrolled identity takes precedence
            self.identity_name = name
            self.identity_score = score
            self.is_known_identity = True
            self.identity_updated_at = current_time
        elif not self.is_known_identity:
            # Unidentified person stays Unknown
            self.identity_name = name
            self.identity_score = score
            self.is_known_identity = False
            self.identity_updated_at = current_time


class PersonTracker:
    """
    Associative tracker managing continuous person tracks across camera frames.
    Matches incoming pose detections with existing tracks using centroid distance & IoU,
    and associates face recognition detections with corresponding body tracks.
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        max_distance: float = 0.35,
        min_iou: float = 0.15,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_iou = min_iou
        self._next_track_id = 1
        self._tracks: Dict[int, PersonTrack] = {}

    @property
    def tracks(self) -> Dict[int, PersonTrack]:
        return self._tracks

    def update(
        self,
        poses: List[PoseResult],
        image_shape: Tuple[int, int],
        current_time: Optional[float] = None,
    ) -> List[PersonTrack]:
        """
        Update tracker with detected poses from the current frame.

        Args:
            poses: Detected pose instances.
            image_shape: (height, width) of the frame.
            current_time: Optional explicit timestamp (seconds).

        Returns:
            List of active PersonTrack objects for this frame.
        """
        t = current_time if current_time is not None else time.time()
        h, w = image_shape[:2]

        # Extract candidates from valid poses
        candidates = []
        for pose in poses:
            if pose.features and pose.features.is_valid:
                bbox = pose.features.bbox
                norm_bbox = pose.features.bbox_normalized
                cx = norm_bbox[0] + (norm_bbox[2] / 2.0)
                cy = norm_bbox[1] + (norm_bbox[3] / 2.0)
                candidates.append((bbox, norm_bbox, (cx, cy), pose))

        # If no active tracks, register all candidates as new tracks
        if not self._tracks:
            for bbox, norm_bbox, centroid, pose in candidates:
                self._register_track(bbox, norm_bbox, centroid, pose, t)
            return list(self._tracks.values())

        # If no candidates detected in frame, increment misses for all existing tracks
        if not candidates:
            for track_id in list(self._tracks.keys()):
                track = self._tracks[track_id]
                track.mark_missed()
                if track.consecutive_misses > self.max_disappeared:
                    self._deregister_track(track_id)
            return [t for t in self._tracks.values() if t.consecutive_misses == 0]

        # Associate existing tracks with candidates using Euclidean distance and IoU
        track_ids = list(self._tracks.keys())
        track_centroids = [self._tracks[tid].centroid for tid in track_ids]

        cost_matrix = np.zeros((len(track_ids), len(candidates)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t_box = self._tracks[tid].bbox
            tc = track_centroids[i]
            for j, (c_box, c_norm_box, cc, _) in enumerate(candidates):
                dist = math.hypot(tc[0] - cc[0], tc[1] - cc[1])
                iou = compute_iou(t_box, c_box)
                # Combined cost: lower distance and higher IoU produce lower cost
                cost = dist - (0.5 * iou)
                cost_matrix[i, j] = cost

        # Greedy association
        matched_tracks = set()
        matched_candidates = set()

        rows = cost_matrix.min(axis=1).argsort()
        for row in rows:
            col = cost_matrix[row].argmin()
            if col in matched_candidates:
                continue

            tid = track_ids[row]
            dist = math.hypot(
                track_centroids[row][0] - candidates[col][2][0],
                track_centroids[row][1] - candidates[col][2][1],
            )
            iou = compute_iou(self._tracks[tid].bbox, candidates[col][0])

            # Accept match if distance is reasonable or IoU is non-zero
            if dist < self.max_distance or iou >= self.min_iou:
                bbox, norm_bbox, centroid, pose = candidates[col]
                self._tracks[tid].update(bbox, norm_bbox, centroid, pose, t)
                matched_tracks.add(tid)
                matched_candidates.add(col)

        # Unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid].mark_missed()
                if self._tracks[tid].consecutive_misses > self.max_disappeared:
                    self._deregister_track(tid)

        # Unmatched new candidates
        for j, (bbox, norm_bbox, centroid, pose) in enumerate(candidates):
            if j not in matched_candidates:
                self._register_track(bbox, norm_bbox, centroid, pose, t)

        return [tr for tr in self._tracks.values() if tr.consecutive_misses == 0]

    def associate_faces(
        self,
        face_results: List[Tuple[FaceDetection, RecognitionResult]],
        current_time: Optional[float] = None,
    ) -> None:
        """
        Associate face recognition detections with active person tracks.
        A face belongs to a person if its center is within the upper half of the person's bbox.
        """
        t = current_time if current_time is not None else time.time()

        for det, rec in face_results:
            fx, fy, fw, fh = det.bbox
            fcx = fx + (fw / 2.0)
            fcy = fy + (fh / 2.0)

            best_track: Optional[PersonTrack] = None
            best_dist = float("inf")

            for track in self._tracks.values():
                px, py, pw, ph = track.bbox
                # Check spatial containment in person bounding box (with 20% margin for head top)
                margin = int(ph * 0.20)
                upper_py = max(0, py - margin)
                # Upper 60% of person box
                lower_py = py + int(ph * 0.60)

                if (px <= fcx <= px + pw) and (upper_py <= fcy <= lower_py):
                    # Face center is inside upper body region
                    dist = math.hypot(fcx - (px + pw / 2.0), fcy - (py + ph * 0.25))
                    if dist < best_dist:
                        best_dist = dist
                        best_track = track

            if best_track is not None:
                best_track.associate_identity(rec.name, rec.score, rec.is_known, t)
                logger.debug(
                    f"Associated face '{rec.name}' ({rec.score:.2f}) with Track #{best_track.track_id}"
                )

    def _register_track(
        self,
        bbox: Tuple[int, int, int, int],
        bbox_normalized: Tuple[float, float, float, float],
        centroid: Tuple[float, float],
        pose: Optional[PoseResult],
        current_time: float,
    ) -> PersonTrack:
        track = PersonTrack(
            track_id=self._next_track_id,
            bbox=bbox,
            bbox_normalized=bbox_normalized,
            centroid=centroid,
            pose=pose,
            first_seen_timestamp=current_time,
            last_seen_timestamp=current_time,
        )
        self._tracks[self._next_track_id] = track
        self._next_track_id += 1
        return track

    def _deregister_track(self, track_id: int) -> None:
        if track_id in self._tracks:
            del self._tracks[track_id]

    def reset(self) -> None:
        """Reset all tracking states."""
        self._tracks.clear()
        self._next_track_id = 1
