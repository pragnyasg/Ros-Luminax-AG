"""
Persistent local face identity database for LUMI Vision.

Stores enrolled identities and multi-sample 128-dimensional embeddings locally in JSON.
Follows local-first privacy: biometrics never leave the device.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Valid name pattern: alphanumeric, spaces, dashes, periods, apostrophes, 1-50 characters
NAME_REGEX = re.compile(r"^[A-Za-z0-9\s\.\-']{1,50}$")


class FaceDatabase:
    """
    Manages local storage, retrieval, and updating of enrolled biometric identities.

    Features:
    - Multi-sample support: Stores multiple embedding vectors per identity for robust
      matching across lighting conditions and facial angles.
    - Atomic writes: Uses temporary files to prevent corruption during unexpected shutdowns.
    - Corrupt file recovery: Automatically backs up unreadable JSON files.
    - Local privacy: Strict local-only persistence.
    """

    def __init__(self, db_path: Union[str, Path], max_samples_per_identity: int = 10) -> None:
        self.db_path = Path(db_path).resolve()
        self.max_samples = max_samples_per_identity
        self._identities: Dict[str, Dict[str, Any]] = {}  # Keyed by lowercased name for fast collision checks

        self._ensure_dir()
        self.load()

    def _ensure_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_name(name: str) -> str:
        """Validate and clean an identity name. Raises ValueError on invalid format."""
        if not isinstance(name, str):
            raise ValueError("Identity name must be a string.")
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Identity name cannot be empty or whitespace.")
        if not NAME_REGEX.match(cleaned):
            raise ValueError(
                f"Invalid identity name: '{cleaned}'. Names must be 1-50 characters, "
                "containing only letters, digits, spaces, hyphens, periods, or apostrophes."
            )
        return cleaned

    def is_empty(self) -> bool:
        """Check if database contains any enrolled identities."""
        return len(self._identities) == 0

    def list_names(self) -> List[str]:
        """Return list of original enrolled identity names."""
        return [entry["name"] for entry in self._identities.values()]

    def list_identities(self) -> List[Dict[str, Any]]:
        """Return summary of all enrolled identities."""
        summary = []
        for entry in self._identities.values():
            summary.append({
                "id": entry["id"],
                "name": entry["name"],
                "enrolled_at": entry["enrolled_at"],
                "updated_at": entry.get("updated_at", entry["enrolled_at"]),
                "num_samples": len(entry.get("samples", [])),
            })
        return summary

    def get_identity(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch identity record by name (case-insensitive)."""
        clean_name = name.strip().lower()
        return self._identities.get(clean_name)

    def enrol(
        self,
        name: str,
        embedding: Union[np.ndarray, List[float]],
        replace: bool = False,
    ) -> str:
        """
        Enrol an identity embedding.

        Args:
            name: Person's name (e.g. "Grandma", "Caregiver").
            embedding: 128-dimensional normalized embedding vector.
            replace: If True, overwrites any existing embeddings for this person.
                     If False, appends to their multi-sample gallery up to max_samples.

        Returns:
            UUID of the enrolled identity.
        """
        clean_name = self.validate_name(name)
        key = clean_name.lower()

        emb_array = np.asarray(embedding, dtype=np.float32).flatten()
        if emb_array.shape != (128,):
            raise ValueError(f"Embedding must be a 128-dimensional vector, got shape {emb_array.shape}")

        # Ensure embedding is L2-normalized
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm

        emb_list = emb_array.tolist()
        now_iso = datetime.now(timezone.utc).isoformat()

        if key in self._identities and not replace:
            # Append sample to existing identity
            identity = self._identities[key]
            samples = identity.setdefault("samples", [])
            if len(samples) >= self.max_samples:
                # Remove oldest sample to keep gallery fresh
                samples.pop(0)
            samples.append({
                "embedding": emb_list,
                "timestamp": now_iso,
            })
            identity["updated_at"] = now_iso
            identity_id = identity["id"]
            logger.info(f"Added sample to identity '{clean_name}' (Total samples: {len(samples)})")
        else:
            # Create new identity or replace existing
            identity_id = str(uuid.uuid4())
            self._identities[key] = {
                "id": identity_id,
                "name": clean_name,
                "enrolled_at": now_iso,
                "updated_at": now_iso,
                "samples": [
                    {
                        "embedding": emb_list,
                        "timestamp": now_iso,
                    }
                ],
            }
            action = "Re-enrolled" if replace else "Enrolled"
            logger.info(f"{action} identity '{clean_name}' with ID {identity_id}")

        self.save()
        return identity_id

    def delete_identity(self, name: str) -> bool:
        """
        Delete an enrolled identity by name.

        Returns:
            True if deleted, False if name was not found.
        """
        clean_name = name.strip().lower()
        if clean_name in self._identities:
            del self._identities[clean_name]
            self.save()
            logger.info(f"Deleted identity '{name}' from database.")
            return True
        logger.warning(f"Cannot delete identity '{name}': not found in database.")
        return False

    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """
        Retrieve all enrolled embeddings organized by identity name.

        Returns:
            Dict mapping identity display name to list of (128,) float32 NumPy arrays.
        """
        result: Dict[str, List[np.ndarray]] = {}
        for entry in self._identities.values():
            disp_name = entry["name"]
            sample_list = []
            for s in entry.get("samples", []):
                arr = np.array(s["embedding"], dtype=np.float32)
                sample_list.append(arr)
            if sample_list:
                result[disp_name] = sample_list
        return result

    def save(self) -> None:
        """
        Atomically save identity database to JSON file.
        Uses a temporary file in the same directory, then renames to avoid corruption.
        """
        self._ensure_dir()
        temp_path = self.db_path.with_suffix(".tmp")
        payload = {
            "version": "1.0",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "identities": list(self._identities.values()),
        }

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (replace target)
            os.replace(temp_path, self.db_path)
            logger.debug(f"Saved {len(self._identities)} identities to {self.db_path}")
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise IOError(f"Failed to save face database to {self.db_path}: {exc}") from exc

    def load(self) -> None:
        """
        Load identity database from JSON file.
        Safely recovers from corrupt files by backing them up.
        """
        self._identities.clear()
        if not self.db_path.is_file():
            logger.debug(f"No existing database file at {self.db_path}. Starting with empty database.")
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"Database file {self.db_path} is empty.")
                    return
                data = json.loads(content)

            identities_list = data.get("identities", [])
            for item in identities_list:
                name = item.get("name")
                if name and isinstance(name, str):
                    self._identities[name.strip().lower()] = item

            logger.info(f"Loaded {len(self._identities)} identities from {self.db_path}")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            backup_path = self.db_path.with_suffix(f".corrupt_{int(datetime.now().timestamp())}.bak")
            shutil.copyfile(self.db_path, backup_path)
            logger.error(
                f"Face database {self.db_path} is corrupt ({exc}). "
                f"A backup was saved to {backup_path}. Initializing empty database."
            )
            self._identities.clear()
