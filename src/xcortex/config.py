from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_type_hints

import yaml
from pydantic import TypeAdapter


@dataclass
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 8417
    mode: str = "models"


@dataclass
class ModelSettings:
    reasoner: str = "medgemma"
    medgemma_id: str = "google/medgemma-1.5-4b-it"
    dtype: str = "bfloat16"
    device: str = "auto"
    siglip_id: str = "google/medsiglip-448"
    sam2_variant: str = "sam2.1_hiera_large"
    sam2_checkpoint: str = "data/models/sam2.1_hiera_large.pt"
    sam2_points_per_side: int = 32
    sam2_points_per_batch: int = 64
    triage_backend: str = "zeroshot"
    triage_checkpoint: str = "data/models/mobilenetv3_mura.pth"
    triage_threshold: float = 0.33
    detector_weights: str = "densenet121-res224-all"
    detector_cache: str = "weights/torchxrayvision"
    anomalyclip_checkpoint: str = "weights/anomalyclip/epoch_15.pth"
    anomalyclip_backbone: str = "weights/anomalyclip/ViT-L-14-336px.pt"
    anomalyclip_image_size: int = 518
    anomalyclip_features: list[int] = field(default_factory=lambda: [24])
    anomalyclip_sigma: float = 4.0
    anomalyclip_overlay_alpha: float = 0.35
    segmentation_enabled: bool = True
    max_new_tokens: int = 768
    structured_reports: bool = True
    generation_timeout_seconds: float = 1200.0
    cpu_memory_gib: int = 10
    cpu_threads: int = 6
    cpu_quantization: str = "none"
    cpu_release_auxiliary_models: bool = False
    local_files_only: bool = True
    default_region: str = ""


@dataclass
class ApiSettings:
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_model: str = "deepseek-chat"
    vision_enabled: bool = False
    timeout_seconds: float = 180.0


@dataclass
class PathsSettings:
    data: str = "data"
    weights: str = "weights"
    db: str = "data/db"
    artifacts: str = "data/artifacts"
    state: str = "data/state/sessions.json"


@dataclass
class RagSettings:
    collection: str = "mura_references"
    top_k: int = 5
    experimental: bool = True
    enabled: bool = True
    attach_reference_images: bool = True
    max_reference_images: int = 5


@dataclass
class AgentSettings:
    # Compatibility field for historical configs; the fixed graph generates once.
    max_attempts: int = 1
    request_timeout_seconds: float = 1800.0
    stream_idle_timeout_seconds: float = 1200.0
    max_concurrent_runs: int = 1


@dataclass
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    paths: PathsSettings = field(default_factory=PathsSettings)
    rag: RagSettings = field(default_factory=RagSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Settings":
        settings = cls()
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {path}")
        candidates = [path] if path else [os.environ.get("XCORTEX_CONFIG"), "configs/default.yaml"]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                with open(candidate, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                settings._apply(data)
                break
        return settings

    def _apply(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a mapping of settings sections")
        for section, values in data.items():
            target = getattr(self, section, None)
            if section not in self.__dataclass_fields__:
                raise ValueError(f"Unknown configuration section: {section}")
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section {section} must be a mapping")
            fields = getattr(type(target), "__dataclass_fields__", {})
            hints = get_type_hints(type(target))
            for key, value in values.items():
                if key not in fields:
                    raise ValueError(f"Unknown configuration setting: {section}.{key}")
                setattr(target, key, TypeAdapter(hints[key]).validate_python(value))

    def resolve_model(self, repo_id: str) -> str:
        local = self.model_path(repo_id)
        from .model_files import inspect_model
        if inspect_model(local)["installed"]:
            return str(local)
        if self.models.local_files_only:
            raise FileNotFoundError(f"Model '{repo_id}' is not completely installed at '{local}'. Run xcortex download medgemma siglip after enabling model access.")
        return repo_id

    def model_path(self, repo_id: str) -> Path:
        local = Path(self.paths.weights) / "hf" / repo_id.replace("/", "__")
        legacy = Path(self.paths.data) / "models" / "hf" / repo_id.replace("/", "__")
        from .model_files import inspect_model
        return legacy if not inspect_model(local)["installed"] and inspect_model(legacy)["installed"] else local

    def ensure_dirs(self) -> None:
        for attr in ("data", "weights", "db", "artifacts"):
            Path(getattr(self.paths, attr)).mkdir(parents=True, exist_ok=True)
        Path(self.paths.state).parent.mkdir(parents=True, exist_ok=True)
