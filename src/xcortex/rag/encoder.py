from __future__ import annotations

import threading
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from ..core.execution import check_cancelled, inference_lock


class MedSigLIPEncoder:
    _shared_instances = weakref.WeakValueDictionary()
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls, settings):
        # Constructing a lazy encoder must not require installed weights: an
        # empty retrieval index can be reported honestly without loading a model.
        local = settings.model_path(settings.models.siglip_id) if hasattr(settings, "model_path") else Path(settings.paths.data) / "models" / "hf" / settings.models.siglip_id.replace("/", "__")
        key = (settings.models.siglip_id, str(local.resolve()), settings.models.device)
        with cls._shared_lock:
            instance = cls._shared_instances.get(key)
            if instance is None:
                instance = cls(settings)
                cls._shared_instances[key] = instance
            return instance

    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.models.siglip_id
        self._resolve = settings.resolve_model
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.RLock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with inference_lock(self._lock):
            check_cancelled()
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoProcessor

            if self.settings.models.device != "auto":
                device = self.settings.models.device
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            resolved = self._resolve(self.model_id)
            try:
                local_only = getattr(self.settings.models, "local_files_only", False)
                self._processor = AutoProcessor.from_pretrained(resolved, local_files_only=local_only)
                self._model = AutoModel.from_pretrained(resolved, torch_dtype=torch.float32, local_files_only=local_only).to(device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load MedSigLIP model from '{resolved}' ({self.model_id}): {exc}. "
                    f"Ensure weights are downloaded via 'xcortex download siglip' or check HuggingFace access."
                ) from exc
            self._model.eval()
            self._device = device
            check_cancelled()

    def encode(self, image: np.ndarray) -> np.ndarray:
        import torch

        from PIL import Image

        with inference_lock(self._lock):
            check_cancelled()
            self._ensure_loaded()
            pil = Image.fromarray(image).convert("RGB")
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                embedding = self._model.get_image_features(**inputs)
            # Transformers 5 may return a pooled model output instead of a tensor.
            if not isinstance(embedding, torch.Tensor):
                embedding = embedding.pooler_output
            norm = embedding.norm(dim=-1, keepdim=True)
            if not torch.isfinite(embedding).all() or torch.any(norm <= 0):
                raise RuntimeError("MedSigLIP returned an invalid image embedding")
            embedding = embedding / norm
            check_cancelled()
            return embedding.squeeze(0).float().cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        if not texts:
            raise ValueError("at least one text is required for MedSigLIP encoding")
        with inference_lock(self._lock):
            check_cancelled()
            self._ensure_loaded()
            inputs = self._processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                embeddings = self._model.get_text_features(**inputs)
            if not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.pooler_output
            norm = embeddings.norm(dim=-1, keepdim=True)
            if not torch.isfinite(embeddings).all() or torch.any(norm <= 0):
                raise RuntimeError("MedSigLIP returned invalid text embeddings")
            embeddings = embeddings / norm
            check_cancelled()
            return embeddings.float().cpu().numpy()
