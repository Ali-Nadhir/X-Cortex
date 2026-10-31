"""Image AnomalyCLIP as fallible chest-pilot visual context.

The score is a normal/abnormal text-similarity softmax, not a calibrated disease
probability. The patch map is not a validated lesion mask. Rendering uses a fixed
scale across images; the unannotated source array is never modified.
"""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
from PIL import Image

from .base import BaseTool, ToolOutput
from ..core.execution import check_cancelled, inference_lock
from ..models.anomalyclip.artifacts import BACKBONE, PROMPT, UPSTREAM_REVISION


def render_anomaly_overlay(image: np.ndarray, response: np.ndarray, *, scale_max: float = 1.0, alpha: float = 0.35) -> Image.Image:
    """Return a separate RGB copy; no per-image min/max normalization."""
    image = np.asarray(image)
    response = np.asarray(response, dtype=np.float32)
    if image.ndim != 2 or image.dtype != np.uint8 or response.shape != image.shape:
        raise ValueError("Anomaly overlay requires normalized uint8 grayscale pixels and an aligned 2-D response")
    if not np.isfinite(response).all() or not np.isfinite(scale_max) or scale_max <= 0:
        raise ValueError("Anomaly map and scale must be finite, with a positive scale maximum")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be between zero and one")
    if float(response.min()) < -1e-5 or float(response.max()) > scale_max + 1e-5:
        raise ValueError("Anomaly map exceeds its declared fixed scale")
    value = np.clip(response / scale_max, 0, 1)
    # Fixed blue -> green -> red palette. The numeric scale is documented in
    # metadata, and low/constant response maps remain visibly low/constant.
    palette = np.stack((np.clip(2 * value - 1, 0, 1), 1 - np.abs(2 * value - 1), np.clip(1 - 2 * value, 0, 1)), axis=-1)
    original = np.repeat(image[..., None], 3, axis=-1).astype(np.float32)
    result = ((1 - alpha) * original + alpha * palette * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


class AnomalyCLIPTool(BaseTool):
    name = "triage"
    label = "Auxiliary anomaly map"

    def __init__(self, settings):
        super().__init__(settings)
        directory = Path(settings.paths.weights) / "anomalyclip"
        self.checkpoint_path = Path(getattr(settings.models, "anomalyclip_checkpoint", directory / PROMPT.filename))
        self.backbone_path = Path(getattr(settings.models, "anomalyclip_backbone", directory / BACKBONE.filename))
        self.image_size = int(getattr(settings.models, "anomalyclip_image_size", 518))
        self.features = tuple(int(v) for v in getattr(settings.models, "anomalyclip_features", [24]))
        self.sigma = float(getattr(settings.models, "anomalyclip_sigma", 4.0))
        self.alpha = float(getattr(settings.models, "anomalyclip_overlay_alpha", 0.35))
        self._runtime = None
        self._lock = threading.RLock()
        self._tested = False

    def _ensure_loaded(self):
        if self._runtime is not None:
            return
        with inference_lock(self._lock):
            if self._runtime is not None:
                return
            import torch
            from ..models.anomalyclip.runtime import AnomalyCLIPRuntime

            torch.set_num_threads(max(1, int(self.settings.models.cpu_threads)))
            device = str(self.settings.models.device)
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            runtime = AnomalyCLIPRuntime(
                self.backbone_path, self.checkpoint_path, device=device,
                image_size=self.image_size, features=self.features, sigma=self.sigma,
            )
            check_cancelled()
            self._runtime = runtime

    def release_models(self):
        """Release auxiliary weights after collection in memory-bounded CPU runs."""
        with inference_lock(self._lock):
            self._runtime = None

    def run(self, image: np.ndarray, meta: dict, history: str) -> ToolOutput:
        check_cancelled()
        region = str(meta.get("region") or "").strip().lower()
        if region not in {"chest", "thorax", "cxr", "lung", "lungs"}:
            raise ValueError("The initial AnomalyCLIP adapter is scoped to explicitly identified chest radiographs")
        source = np.asarray(image)
        if source.ndim != 2 or source.dtype != np.uint8 or source.size == 0:
            raise ValueError("AnomalyCLIP requires normalized uint8 grayscale study pixels")
        threshold = float(self.settings.models.triage_threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Legacy display threshold must be finite and between zero and one")
        self._ensure_loaded()
        assert self._runtime is not None
        with inference_lock(self._lock):
            score, native_map = self._runtime.infer(Image.fromarray(source).convert("RGB"))
        check_cancelled()
        native_map = np.asarray(native_map, dtype=np.float32)
        scale_max = float(len(self.features))
        if (not np.isfinite(score) or not 0 <= score <= 1 or native_map.ndim != 2
                or native_map.size == 0 or not np.isfinite(native_map).all()):
            raise RuntimeError("AnomalyCLIP produced invalid scores or map geometry")
        if native_map.shape != (self.image_size, self.image_size):
            raise RuntimeError("AnomalyCLIP output map dimensions differ from the configured inference image")
        # The model resizes the entire image to a square. Map its response back
        # onto the original normalized image without introducing crop/flip.
        aligned = np.asarray(Image.fromarray(native_map).resize(
            (source.shape[1], source.shape[0]), resample=Image.Resampling.BILINEAR,
        ), dtype=np.float32).copy()
        overlay = render_anomaly_overlay(source, aligned, scale_max=scale_max, alpha=self.alpha)
        payload = {
            "status": "completed", "region": "chest", "method": "AnomalyCLIP ViT-L/14@336px",
            "model_revision": UPSTREAM_REVISION,
            "checkpoint": str(self.checkpoint_path), "backbone": str(self.backbone_path),
            "anomaly_score": float(score),
            # Retain the existing UI field while declaring its actual meaning.
            "abnormality_probability": float(score), "calibrated": False,
            "score_type": "anomalyclip_normal_abnormal_softmax",
            "threshold": threshold, "band": "unvalidated",
            "threshold_role": "Legacy display reference only; not used to declare normality or disease",
            "output_type": "spatial_anomaly_response", "detections": [],
            "detections_semantics": "No disease-specific bounding boxes or validated lesion masks are produced",
            "prompts": {"normal": "learned context + object", "abnormal": "learned context + damaged object"},
            "image_score_temperature": 0.07,
            "map": {
                "artifact": "anomaly_map.npy", "native_artifact": "anomaly_map_native.npy",
                "overlay_artifact": "overlay_anomalyclip.png", "feature_levels": list(self.features),
                "smoothing_sigma_model_pixels": self.sigma,
                "aggregation": "sum of normal/abnormal patch response over selected feature levels",
                "score_range": [0.0, scale_max], "calibrated": False,
                "native_shape": list(native_map.shape), "aligned_shape": list(source.shape),
                "transform": "whole-image square resize, then bilinear response resize to source dimensions; no crop or flip",
                "display_scale": {"minimum": 0.0, "maximum": scale_max, "per_image_minmax": False,
                                  "palette": "fixed blue-green-red", "alpha": self.alpha},
                "minimum": float(aligned.min()), "maximum": float(aligned.max()), "mean": float(aligned.mean()),
            },
            "limitations": [
                "Experimental auxiliary context, not a final diagnosis",
                "Chest lesion localization is not validated by the image-level chest results in the AnomalyCLIP paper",
                "Scores and patch responses are not calibrated disease probabilities",
                "An absent or weak response does not rule out a visual finding",
                "No guaranteed sensitivity or universal abnormality coverage",
            ],
        }
        check_cancelled()
        self._tested = True
        # Pipeline already preserves arrays as .npy; overlay retains source size.
        return ToolOutput(
            summary=f"AnomalyCLIP auxiliary response = {float(score):.4f} (uncalibrated; no diagnosis implied)",
            payload=payload,
            images=[("overlay_anomalyclip.png", overlay), ("anomaly_map.npy", aligned), ("anomaly_map_native.npy", native_map)],
        )
