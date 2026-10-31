from __future__ import annotations

import threading

from PIL import Image

from ..core.execution import check_cancelled
from ..rag.encoder import MedSigLIPEncoder
from ..rag.metadata import normalize_region
from ..rag.store import VectorStore
from ..utils.dicom import load_study_image
from .base import BaseTool, ToolOutput


class RAGRetrievalTool(BaseTool):
    name = "rag"
    label = "Retrieval"

    def __init__(self, settings):
        super().__init__(settings)
        self._encoder = None
        self._store = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._encoder is not None and self._store is not None:
            return
        with self._lock:
            if self._store is None:
                self._store = VectorStore(self.settings)
            if self._encoder is None:
                self._encoder = MedSigLIPEncoder.shared(self.settings)

    def _unavailable(self, status: str, note: str) -> ToolOutput:
        return ToolOutput(summary=note, payload={
            "matches": [], "count": 0, "collection": self.settings.rag.collection,
            "status": status, "note": note, "experimental": self.settings.rag.experimental,
        })

    def release_models(self):
        # The shared encoder cache is weak. Other live owners retain their own
        # model; this run need not pin it while MedGemma is interpreting images.
        with self._lock:
            self._encoder = None

    def run(self, image, meta: dict, history: str) -> ToolOutput:
        check_cancelled()
        if not getattr(self.settings.rag, "enabled", True):
            return self._unavailable("disabled", "reference retrieval disabled for this run")
        self._ensure_loaded()
        assert self._store is not None and self._encoder is not None
        if self._store.count() == 0:
            return self._unavailable("empty_index", "reference database is empty; run 'xcortex index' with training/reference cases first")
        region = normalize_region(meta.get("region"))
        if region == "unknown":
            return self._unavailable("anatomy_unknown", "reference retrieval requires study region metadata to avoid comparing different anatomy")
        embedding = self._encoder.encode(image)
        matches = self._store.query(
            embedding, self.settings.rag.top_k, region=region,
            exclude_patient_id=meta.get("patient_id"), exclude_study_id=meta.get("study_id"),
            exclude_image_sha256=meta.get("image_sha256"),
        )
        if not matches:
            return self._unavailable("no_eligible_references", "no same-anatomy reference cases remain after patient, study and duplicate exclusions")
        payload_matches = []
        images = []
        unavailable_images = []
        for match in matches:
            check_cancelled()
            cosine = max(-1.0, min(1.0, 1.0 - match["distance"]))
            similarity = round(cosine * 100.0)
            info = match["meta"]
            image_path = info.get("image_path")
            try:
                if not image_path:
                    raise FileNotFoundError("reference has no image source")
                reference_image, _ = load_study_image(image_path)
            except Exception as exc:
                # A text-only hit is not a successfully returned image reference.
                unavailable_images.append({"case_id": match["case_id"], "status": "image_unavailable", "error_type": type(exc).__name__})
                continue
            i = len(payload_matches)
            image_name = f"ref_{i}.png"
            payload_matches.append(
                {
                    "case_id": match["case_id"],
                    "region": info.get("region", "unknown"),
                    "similarity_percent": similarity,
                    "similarity_cosine": round(cosine, 6),
                    "score_type": "cosine_similarity_not_probability",
                    "label": info.get("label", "unknown"),
                    "reference": info.get("reference", ""),
                    "reference_kind": info.get("reference_kind", "legacy_unverified_metadata"),
                    "source": info.get("source", "unspecified"),
                    "dataset_split": info.get("dataset_split", "unspecified"),
                    "patient_identity_available": bool(info.get("patient_id")),
                    "image_artifact": image_name,
                    "image_dimensions": [int(reference_image.shape[1]), int(reference_image.shape[0])],
                }
            )
            # Preserve the whole image at its normalized source resolution.
            # The medical model's processor performs its own resize/patching.
            images.append((image_name, Image.fromarray(reference_image).convert("RGB")))
        payload = {
            "matches": payload_matches, "count": len(payload_matches),
            "collection": self.settings.rag.collection,
            "status": "partial_images" if unavailable_images and payload_matches else ("images_unavailable" if unavailable_images else "available"),
            "requested_count": self.settings.rag.top_k,
            "unavailable_images": unavailable_images,
            "experimental": self.settings.rag.experimental, "region_filter": region,
            "note": "reference labels describe other cases; cosine similarity is not diagnostic confidence",
            "query_patient_identity_available": bool(meta.get("patient_id")),
        }
        if not meta.get("patient_id") or any(not m["patient_identity_available"] for m in payload_matches):
            payload["limitations"] = ["patient-disjoint retrieval cannot be guaranteed without patient identifiers for every case"]
        similarities = " / ".join(f"{m['similarity_percent']}%" for m in payload_matches)
        summary = f"{len(payload_matches)} same-anatomy reference images ({similarities} cosine similarity; not probability)"
        return ToolOutput(summary=summary, payload=payload, images=images)
