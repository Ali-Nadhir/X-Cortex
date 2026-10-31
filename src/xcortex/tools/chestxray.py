"""CPU-capable chest finding classifier for integration smoke tests.

The maximum pathology score is an uncalibrated triage heuristic, not a validated
probability of any disease. This tool is deliberately restricted to chest inputs.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseTool, ToolOutput
from ..core.execution import check_cancelled, inference_lock
from ..downloads import DETECTOR_FILENAME
from ..model_files import inspect_checkpoint


class ChestXRayTool(BaseTool):
    name = "triage"
    label = "Chest findings"

    def __init__(self, settings):
        super().__init__(settings)
        self._model: Any = None
        self._lock = threading.RLock()
        self._tested = False

    @property
    def checkpoint_path(self):
        return Path(self.settings.models.detector_cache) / DETECTOR_FILENAME

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not inspect_checkpoint(self.checkpoint_path)["installed"]:
                raise FileNotFoundError(f"Detector weights missing/incomplete at {self.checkpoint_path}. Run xcortex download detector.")
            import torch
            import torchxrayvision as xrv
            torch.set_num_threads(max(1, self.settings.models.cpu_threads))
            model: Any = xrv.models.DenseNet(weights=self.settings.models.detector_weights, cache_dir=self.settings.models.detector_cache)
            # Disable library operating-point renormalization; return sigmoid
            # class scores with explicit, unvalidated deployment threshold.
            model.op_threshs = None
            model.apply_sigmoid = True
            model.eval().to("cpu")
            self._model = model

    def run(self, image, meta, history):
        check_cancelled()
        region = str(meta.get("region") or "").strip().lower()
        if region not in {"chest", "thorax", "cxr", "lung", "lungs"}:
            raise ValueError("Chest detector requires an explicitly identified chest radiograph; it cannot assess musculoskeletal studies.")
        self._ensure_loaded()
        import torch
        import torchxrayvision as xrv
        normalized = xrv.datasets.normalize(np.asarray(image, dtype=np.float32), 255)[None, ...]
        normalized = xrv.datasets.XRayCenterCrop()(normalized)
        normalized = xrv.datasets.XRayResizer(224)(normalized)
        tensor = torch.from_numpy(normalized.astype(np.float32))[None]
        with inference_lock(self._lock), torch.inference_mode():
            scores = self._model(tensor)[0].cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("Detector produced nonfinite scores")
        threshold = float(self.settings.models.triage_threshold)
        score_map = {name: round(float(value), 6) for name, value in zip(self._model.pathologies, scores)}
        maximum = round(float(scores.max()), 4)
        band = "high" if maximum >= max(0.66, threshold) else ("indeterminate" if maximum >= threshold else "low")
        detections = [{"label": name, "score": value, "threshold": threshold} for name, value in score_map.items() if value >= threshold]
        self._tested = True
        return ToolOutput(summary=f"Maximum chest finding score = {maximum} ({band}; uncalibrated screening heuristic)", payload={
            "abnormality_probability": maximum, "threshold": threshold, "prediction": "abnormal" if maximum >= threshold else "normal",
            "band": band, "method": f"TorchXRayVision {self.settings.models.detector_weights}", "calibrated": False,
            "score_type": "max_pathology_score", "region": "chest", "pathology_scores": score_map, "detections": detections,
            "limitations": ["Chest radiographs only", "Threshold is not clinically validated", "No guaranteed sensitivity", "Scores are not calibrated disease probabilities"],
        })
