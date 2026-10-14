"""
Unit and integration tests for FaceRecognizer and FaceDetector in LUMI Vision.
"""

from pathlib import Path
import numpy as np
import pytest

from luminax_vision.face_database import FaceDatabase
from luminax_vision.face_detector import FaceDetection, FaceDetector
from luminax_vision.face_recognizer import FaceRecognizer, RecognitionResult


@pytest.fixture
def recognizer_cosine():
    # Use actual model if available, otherwise initialized
    model_path = Path("models/face_recognition_sface_2021dec.onnx")
    return FaceRecognizer(
        model_path=model_path,
        distance_metric="cosine",
        cosine_threshold=0.363,
    )


@pytest.fixture
def recognizer_l2():
    model_path = Path("models/face_recognition_sface_2021dec.onnx")
    return FaceRecognizer(
        model_path=model_path,
        distance_metric="l2",
        l2_threshold=1.128,
    )


def test_cosine_similarity_computation(recognizer_cosine: FaceRecognizer):
    """Test mathematical correctness of cosine similarity computation."""
    vec1 = np.zeros(128, dtype=np.float32)
    vec1[0] = 1.0

    vec2 = np.zeros(128, dtype=np.float32)
    vec2[0] = 1.0

    # Identical vectors: cosine similarity = 1.0
    sim_identical = recognizer_cosine.compute_similarity(vec1, vec2)
    assert pytest.approx(sim_identical, 1e-5) == 1.0

    # Orthogonal vectors: cosine similarity = 0.0
    vec_ortho = np.zeros(128, dtype=np.float32)
    vec_ortho[1] = 1.0
    sim_ortho = recognizer_cosine.compute_similarity(vec1, vec_ortho)
    assert pytest.approx(sim_ortho, 1e-5) == 0.0

    # Opposite vectors: cosine similarity = -1.0
    vec_opp = -vec1
    sim_opp = recognizer_cosine.compute_similarity(vec1, vec_opp)
    assert pytest.approx(sim_opp, 1e-5) == -1.0


def test_recognition_threshold_behavior(recognizer_cosine: FaceRecognizer, tmp_path: Path):
    """Verify recognition decision thresholding with controlled embeddings."""
    db = FaceDatabase(tmp_path / "identities.json")

    base_emb = np.random.randn(128).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)
    db.enrol("Grandma", base_emb)

    # 1. Candidate is identical -> Matches Grandma
    res_exact = recognizer_cosine.recognize(base_emb, db)
    assert res_exact.is_known is True
    assert res_exact.name == "Grandma"
    assert pytest.approx(res_exact.score, 1e-4) == 1.0

    # 2. Candidate has subtle noise (guaranteed cosine similarity > 0.8 > 0.363) -> Matches Grandma
    noise = np.random.randn(128).astype(np.float32)
    noise /= np.linalg.norm(noise)
    noisy_emb = 0.95 * base_emb + 0.05 * noise
    noisy_emb /= np.linalg.norm(noisy_emb)
    sim_noise = recognizer_cosine.compute_similarity(base_emb, noisy_emb)
    assert sim_noise > recognizer_cosine.cosine_threshold
    res_noise = recognizer_cosine.recognize(noisy_emb, db)
    assert res_noise.is_known is True
    assert res_noise.name == "Grandma"

    # 3. Candidate is orthogonal (cosine similarity ~0.0 < 0.363) -> Unknown
    ortho_emb = np.random.randn(128).astype(np.float32)
    # Project out base_emb
    ortho_emb -= np.dot(ortho_emb, base_emb) * base_emb
    ortho_emb /= np.linalg.norm(ortho_emb)
    res_unknown = recognizer_cosine.recognize(ortho_emb, db)
    assert res_unknown.is_known is False
    assert res_unknown.name == "Unknown"
    assert res_unknown.score < recognizer_cosine.cosine_threshold


def test_empty_database_recognition(recognizer_cosine: FaceRecognizer, tmp_path: Path):
    """Verify that an empty database always returns 'Unknown' without raising exceptions."""
    db = FaceDatabase(tmp_path / "identities.json")
    query_emb = np.random.randn(128).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    res = recognizer_cosine.recognize(query_emb, db)
    assert res.is_known is False
    assert res.name == "Unknown"
    assert res.score == 0.0


def test_multi_identity_selection(recognizer_cosine: FaceRecognizer, tmp_path: Path):
    """Verify that the recognizer selects the closest match among multiple enrolled people."""
    db = FaceDatabase(tmp_path / "identities.json")

    emb_alice = np.zeros(128, dtype=np.float32)
    emb_alice[0] = 1.0
    db.enrol("Alice", emb_alice)

    emb_bob = np.zeros(128, dtype=np.float32)
    emb_bob[1] = 1.0
    db.enrol("Bob", emb_bob)

    # Query vector very close to Bob (angle 15 degrees)
    query = np.zeros(128, dtype=np.float32)
    query[1] = 0.95
    query[2] = 0.31
    query /= np.linalg.norm(query)

    res = recognizer_cosine.recognize(query, db)
    assert res.is_known is True
    assert res.name == "Bob"


def test_l2_metric_recognition(recognizer_l2: FaceRecognizer, tmp_path: Path):
    """Verify recognition behavior using Euclidean L2 distance."""
    db = FaceDatabase(tmp_path / "identities.json")

    emb = np.zeros(128, dtype=np.float32)
    emb[0] = 1.0
    db.enrol("Caregiver", emb)

    # Same vector: distance 0.0 <= 1.128 -> Match
    res_match = recognizer_l2.recognize(emb, db)
    assert res_match.is_known is True
    assert res_match.name == "Caregiver"
    assert pytest.approx(res_match.score, 1e-4) == 0.0

    # Distant vector: distance sqrt(2) ≈ 1.414 > 1.128 -> Unknown
    far_vec = np.zeros(128, dtype=np.float32)
    far_vec[1] = 1.0
    res_far = recognizer_l2.recognize(far_vec, db)
    assert res_far.is_known is False
    assert res_far.name == "Unknown"
    assert res_far.score > recognizer_l2.l2_threshold


def test_face_detector_and_recognizer_integration():
    """Verify actual model loading and inference with YuNet and SFace."""
    det_model = Path("models/face_detection_yunet_2023mar.onnx")
    rec_model = Path("models/face_recognition_sface_2021dec.onnx")

    assert det_model.is_file(), f"Model file missing: {det_model}"
    assert rec_model.is_file(), f"Model file missing: {rec_model}"

    detector = FaceDetector(det_model, score_threshold=0.5)
    recognizer = FaceRecognizer(rec_model)

    # Create synthetic test frame
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_img)
    # Background should have 0 faces
    assert isinstance(detections, list)

    # Test recognizer feature extraction on synthetic landmark coordinates
    mock_raw_face = np.array([
        100.0, 100.0, 120.0, 120.0,  # bbox: x, y, w, h
        130.0, 130.0, 170.0, 130.0,  # right eye, left eye
        150.0, 160.0,                # nose
        135.0, 190.0, 165.0, 190.0,  # right mouth, left mouth
        0.98                         # score
    ], dtype=np.float32)

    mock_det = FaceDetection(
        bbox=(100, 100, 120, 120),
        landmarks=np.array([[130, 130], [170, 130], [150, 160], [135, 190], [165, 190]], dtype=np.float32),
        score=0.98,
        raw_face=mock_raw_face,
    )

    emb = recognizer.extract_embedding(test_img, mock_det)
    assert emb.shape == (128,)
    assert emb.dtype == np.float32
    # Verify unit norm
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, 1e-4) == 1.0
