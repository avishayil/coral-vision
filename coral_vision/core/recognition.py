"""Face recognition and matching logic using L2 distance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from coral_vision.core.types import Match

if TYPE_CHECKING:
    from coral_vision.core.storage_backend import StorageBackend


def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate squared L2 distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Squared L2 distance.
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.sum((a - b) ** 2))


@dataclass(frozen=True)
class PersonEmbeddings:
    """Storage for a person's face embeddings.

    Attributes:
        person_id: Unique identifier for the person.
        name: Display name of the person.
        embeddings: Array of face embeddings with shape (N, D).
    """

    person_id: str
    name: str
    embeddings: np.ndarray  # (N, D)


class EmbeddingDB:
    """Database of person embeddings for face recognition.

    Attributes:
        people: List of PersonEmbeddings objects.
    """

    def __init__(self, people: list[PersonEmbeddings]) -> None:
        """Initialize embedding database.

        Args:
            people: List of PersonEmbeddings to store.
        """
        self.people = people

    @staticmethod
    def load(scanned_people_dir: Path, people_index: dict[str, str]) -> "EmbeddingDB":
        """Load all person embeddings from disk.

        DEPRECATED: Use load_from_backend() instead.

        Args:
            scanned_people_dir: Directory containing person data.
            people_index: Mapping of person_id to name.

        Returns:
            EmbeddingDB instance with loaded embeddings.
        """
        people: list[PersonEmbeddings] = []

        for person_id, name in people_index.items():
            if person_id == "unknown":
                continue
            emb_dir = scanned_people_dir / person_id / "embeddings"
            if not emb_dir.exists():
                continue

            embs = []
            for f in sorted(emb_dir.glob("*.npy")):
                arr = np.load(f)
                embs.append(arr.reshape(-1))
            if not embs:
                continue

            mat = np.stack(embs, axis=0)  # (N,D)
            people.append(
                PersonEmbeddings(person_id=person_id, name=name, embeddings=mat)
            )

        return EmbeddingDB(people)

    @staticmethod
    def load_from_backend(storage: "StorageBackend") -> "EmbeddingDB":
        """Load all person embeddings from storage backend.

        Args:
            storage: Storage backend to load from.

        Returns:
            EmbeddingDB instance with loaded embeddings.
        """
        people: list[PersonEmbeddings] = []
        people_index = storage.load_people_index()
        all_embeddings = storage.get_all_embeddings()

        for person_id, name in people_index.items():
            if person_id == "unknown":
                continue

            embs_list = all_embeddings.get(person_id, [])
            if not embs_list:
                continue

            # Stack embeddings into matrix (N, D)
            mat = np.stack([e.reshape(-1) for e in embs_list], axis=0)
            people.append(
                PersonEmbeddings(person_id=person_id, name=name, embeddings=mat)
            )

        return EmbeddingDB(people)

    def match(self, emb: np.ndarray, per_person_k: int, top_k: int) -> list[Match]:
        """Find best matching persons for a face embedding.

        Args:
            emb: Face embedding vector.
            per_person_k: Number of best embeddings to average per person.
            top_k: Number of top matches to return.

        Returns:
            List of Match objects sorted by distance (best first).
        """
        emb = emb.reshape(-1)

        matches: list[Match] = []
        for p in self.people:
            dists = [l2_sq(emb, e) for e in p.embeddings]
            dists.sort()
            top = dists[: max(1, min(per_person_k, len(dists)))]
            mean_dist = float(np.mean(top))
            matches.append(
                Match(person_id=p.person_id, name=p.name, distance=mean_dist)
            )

        matches.sort(key=lambda m: m.distance)
        return matches[: max(1, top_k)]
