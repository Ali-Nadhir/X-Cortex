from __future__ import annotations

import threading

from ..rag.encoder import MedSigLIPEncoder
from .base import BaseTool, ToolOutput

BAND_HIGH = 0.66
TEMPERATURE = 0.1

NORMAL_PROMPTS = [
    "a normal {} radiograph",
    "a healthy {} x-ray",
    "an unremarkable {} radiograph without pathology",
]

ABNORMAL_PROMPTS = [
    "a {} radiograph with a fracture",
    "an abnormal {} x-ray showing pathology",
    "a {} radiograph with a visible bone lesion",
]


class ZeroShotAnomalyTool(BaseTool):
    name = "triage"
    label = "Anomaly"

    def __init__(self, settings):
        super().__init__(settings)
        self._encoder = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    self._encoder = MedSigLIPEncoder.shared(self.settings)

    def run(self, image, meta: dict, history: str) -> ToolOutput:
        import math

        self._ensure_loaded()
        assert self._encoder is not None
        region = str(meta.get("region") or "musculoskeletal").lower()
        texts = [p.format(region) for p in NORMAL_PROMPTS] + [p.format(region) for p in ABNORMAL_PROMPTS]
        image_embedding = self._encoder.encode(image)
        text_embeddings = self._encoder.encode_texts(texts)
        similarities = text_embeddings @ image_embedding
        n = len(NORMAL_PROMPTS)
        score_normal = float(similarities[:n].mean())
        score_abnormal = float(similarities[n:].mean())
        probability = 1.0 / (1.0 + math.exp((score_normal - score_abnormal) / TEMPERATURE))
        probability = round(probability, 4)

        threshold = self.settings.models.triage_threshold
        band = "high" if probability >= max(BAND_HIGH, threshold) else ("indeterminate" if probability >= threshold else "low")
        payload = {
            "abnormality_probability": probability,
            "threshold": threshold,
            "prediction": "abnormal" if probability >= threshold else "normal",
            "band": band,
            "method": "zero-shot MedSigLIP",
            "calibrated": False,
            "score_type": "uncalibrated_similarity_score",
            "region": region,
            "similarity_normal": round(score_normal, 4),
            "similarity_abnormal": round(score_abnormal, 4),
        }
        return ToolOutput(
            summary=f"Abnormality score = {probability} ({band}) — uncalibrated zero-shot", payload=payload
        )
