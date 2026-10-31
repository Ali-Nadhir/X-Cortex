from __future__ import annotations

import threading
from typing import TYPE_CHECKING, cast

import numpy as np

from ..core.execution import check_cancelled
from .metadata import normalize_region

if TYPE_CHECKING:
    from chromadb.api.types import Metadata


class VectorStore:
    def __init__(self, settings):
        self.settings = settings
        self._client = None
        self._collection = None
        self._lock = threading.RLock()

    def _ensure(self):
        if self._collection is not None:
            return self._collection
        with self._lock:
            if self._collection is not None:
                return self._collection
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.settings.paths.db))
            self._collection = self._client.get_or_create_collection(
                name=self.settings.rag.collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        return self._ensure().count()

    def upsert(self, ids: list[str], embeddings: list, metadatas: list[dict]) -> None:
        if not (len(ids) == len(embeddings) == len(metadatas)):
            raise ValueError("reference IDs, embeddings and metadata must have equal lengths")
        if not ids:
            return
        with self._lock:
            check_cancelled()
            self._ensure().upsert(
                ids=ids,
                embeddings=[self._embedding(e) for e in embeddings],
                metadatas=cast("list[Metadata]", metadatas),
            )

    @staticmethod
    def _embedding(value) -> list[float]:
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError("reference embedding must be a finite nonzero vector")
        return vector.tolist()

    def query(
        self, embedding: np.ndarray, k: int, *, region: str | None = None,
        exclude_patient_id: str | None = None, exclude_study_id: str | None = None,
        exclude_image_sha256: str | None = None,
    ) -> list[dict]:
        if k <= 0:
            return []
        clauses = []
        if region and normalize_region(region) != "unknown":
            clauses.append({"region": {"$eq": normalize_region(region)}})
        for key, value in (
            ("patient_id", exclude_patient_id), ("study_id", exclude_study_id),
            ("image_sha256", exclude_image_sha256),
        ):
            if value:
                clauses.append({key: {"$ne": str(value)}})
        options = {}
        if clauses:
            options["where"] = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        with self._lock:
            check_cancelled()
            collection = self._ensure()
            count = collection.count()
            if not count:
                return []
            result = collection.query(
                query_embeddings=[self._embedding(embedding)],
                n_results=min(int(k), count), **options,
            )
            check_cancelled()
        matches = []
        ids, metadatas, distances = result.get("ids"), result.get("metadatas"), result.get("distances")
        if not ids or not ids[0]:
            return matches
        if not metadatas or not distances or len(metadatas[0]) != len(ids[0]) or len(distances[0]) != len(ids[0]):
            raise RuntimeError("Reference index returned incomplete metadata or similarity scores")
        for i in range(len(ids[0])):
            matches.append(
                {
                    "case_id": ids[0][i],
                    "meta": metadatas[0][i],
                    "distance": float(distances[0][i]),
                }
            )
        return matches
