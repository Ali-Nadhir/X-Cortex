from __future__ import annotations

import numpy as np
import threading
from typing import Any

from .base import BaseTool, ToolOutput
from ..core.execution import check_cancelled, inference_lock

BAND_HIGH = 0.66


class TriageTool(BaseTool):
    name = "triage"
    label = "Triage"

    def __init__(self, settings):
        super().__init__(settings)
        self._model: Any = None
        self._lock = threading.RLock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from torch import nn
        from torchvision import models

        with self._lock:
            if self._model is not None:
                return
            model: Any = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
            state = torch.load(self.settings.models.triage_checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self._model = model

    def run(self, image: np.ndarray, meta: dict, history: str) -> ToolOutput:
        import torch

        check_cancelled()
        self._ensure_loaded()
        from PIL import Image
        image = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))
        img = np.stack([image] * 3, axis=-1).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))[None]
        with inference_lock(self._lock), torch.inference_mode():
            logit = self._model(torch.from_numpy(img))
            probability = float(torch.sigmoid(logit).item())
        threshold = self.settings.models.triage_threshold
        band = "high" if probability >= max(BAND_HIGH, threshold) else ("indeterminate" if probability >= threshold else "low")
        payload = {
            "abnormality_probability": round(probability, 4),
            "threshold": threshold,
            "prediction": "abnormal" if probability >= threshold else "normal",
            "band": band,
            "calibrated": False,
            "score_type": "uncalibrated_classifier_score",
            "method": "MobileNetV3 binary classifier",
        }
        return ToolOutput(summary=f"Abnormality score = {payload['abnormality_probability']} ({band}, uncalibrated)", payload=payload)
