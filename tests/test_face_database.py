"""
Unit tests for FaceDatabase in LUMI Vision.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from luminax_vision.face_database import FaceDatabase


def test_empty_database(tmp_path: Path):
    """Verify behavior of a newly created, empty database."""
    db_file = tmp_path / "identities.json"
    db = FaceDatabase(db_file)
    assert db.is_empty()
    assert db.list_names() == []
    assert db.list_identities() == []
    assert db.get_identity("NonExistent") is None


def test_enrol_and_retrieve(tmp_path: Path):
    """Verify single-sample enrollment and retrieval."""
    db_file = tmp_path / "identities.json"
    db = FaceDatabase(db_file)

    dummy_embedding = np.random.randn(128).astype(np.float32)
    dummy_embedding /= np.linalg.norm(dummy_embedding)

    identity_id = db.enrol("Grandma", dummy_embedding)
    assert identity_id is not None
    assert not db.is_empty()
    assert db.list_names() == ["Grandma"]

    record = db.get_identity("Grandma")
    assert record is not None
    assert record["name"] == "Grandma"
    assert len(record["samples"]) == 1

    # Case-insensitive lookup
    record_lower = db.get_identity("grandma")
    assert record_lower is not None
    assert record_lower["id"] == identity_id


def test_multi_sample_enrolment(tmp_path: Path):
    """Verify adding multiple samples to an existing identity."""
    db_file = tmp_path / "identities.json"
    db = FaceDatabase(db_file, max_samples_per_identity=5)

    emb1 = np.random.randn(128).astype(np.float32)
    emb2 = np.random.randn(128).astype(np.float32)

    db.enrol("Caregiver", emb1)
    db.enrol("Caregiver", emb2, replace=False)

    record = db.get_identity("Caregiver")
    assert record is not None
    assert len(record["samples"]) == 2

    # Verify replacement mode
    emb3 = np.random.randn(128).astype(np.float32)
    db.enrol("Caregiver", emb3, replace=True)
    record_replaced = db.get_identity("Caregiver")
    assert len(record_replaced["samples"]) == 1


def test_deletion_workflow(tmp_path: Path):
    """Verify deleting an enrolled identity."""
    db_file = tmp_path / "identities.json"
    db = FaceDatabase(db_file)

    emb = np.random.randn(128).astype(np.float32)
    db.enrol("Patient Bob", emb)

    assert "Patient Bob" in db.list_names()

    # Successful deletion
    assert db.delete_identity("Patient Bob") is True
    assert db.is_empty()
    assert db.get_identity("Patient Bob") is None

    # Deleting non-existent identity
    assert db.delete_identity("Nobody") is False


def test_name_validation(tmp_path: Path):
    """Verify validation rules on identity names."""
    db_file = tmp_path / "identities.json"
    db = FaceDatabase(db_file)
    emb = np.random.randn(128).astype(np.float32)

    # Empty name
    with pytest.raises(ValueError, match="empty or whitespace"):
        db.enrol("", emb)

    # Whitespace only
    with pytest.raises(ValueError, match="empty or whitespace"):
        db.enrol("   ", emb)

    # Invalid special characters
    with pytest.raises(ValueError, match="Invalid identity name"):
        db.enrol("Bad<Name>!", emb)


def test_invalid_embedding_dimension(tmp_path: Path):
    """Verify that embeddings with invalid shape raise ValueError."""
    db_file = tmp_path / "identities.json"
    db = FaceDatabase(db_file)

    bad_emb = np.random.randn(64).astype(np.float32)  # Wrong length
    with pytest.raises(ValueError, match="128-dimensional"):
        db.enrol("Valid Name", bad_emb)


def test_persistence_across_instances(tmp_path: Path):
    """Verify that enrolled data is saved and reloaded by another instance."""
    db_file = tmp_path / "identities.json"
    db1 = FaceDatabase(db_file)

    emb = np.random.randn(128).astype(np.float32)
    db1.enrol("Dr. Alice", emb)

    # Instantiate second instance pointing to same file
    db2 = FaceDatabase(db_file)
    assert not db2.is_empty()
    assert "Dr. Alice" in db2.list_names()

    embeddings_dict = db2.get_all_embeddings()
    assert "Dr. Alice" in embeddings_dict
    assert len(embeddings_dict["Dr. Alice"]) == 1
    np.testing.assert_allclose(
        embeddings_dict["Dr. Alice"][0],
        emb / np.linalg.norm(emb),
        rtol=1e-5,
    )


def test_corrupt_file_recovery(tmp_path: Path):
    """Verify that a corrupted JSON file triggers backup and starts empty."""
    db_file = tmp_path / "identities.json"
    db_file.write_text("{ this is corrupted json content !@@#$")

    db = FaceDatabase(db_file)
    assert db.is_empty()

    # Verify backup was created
    backups = list(tmp_path.glob("identities.corrupt_*.bak"))
    assert len(backups) == 1
