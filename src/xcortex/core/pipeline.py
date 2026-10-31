from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .execution import check_cancelled
from .messages import Artifact, EvidenceBundle, EvidenceItem, Study
from ..tools.registry import ToolRegistry
from ..utils.dicom import load_study_image

EVIDENCE_PREFIX = {"triage": "TRI", "segmentation": "SEG", "rag": "RAG"}


def artifact_kind(name: str) -> str:
    if name.startswith("overlay"):
        return "overlay"
    if name.startswith("ref_"):
        return "reference"
    if name.startswith("masks"):
        return "masks"
    if name.startswith("preview"):
        return "preview"
    return "image"


class AnalysisPipeline:
    def __init__(self, registry: ToolRegistry, settings):
        self.registry = registry
        self.settings = settings

    def run(self, study: Study, run_dir: Path, on_event) -> EvidenceBundle:
        run_dir.mkdir(parents=True, exist_ok=True)
        check_cancelled()
        image, meta = load_study_image(study.image_path)
        # Physical calibration and pixel dimensions must describe the loaded source,
        # rather than stale or caller-supplied metadata from another image.
        caller_meta = {
            key: value for key, value in study.metadata.items()
            if key not in {"pixel_spacing_mm", "pixel_spacing_source"}
        }
        meta = {**caller_meta, **meta}
        meta.setdefault("study_id", study.id)
        normalized_path = run_dir / "normalized.png"
        Image.fromarray(image).save(normalized_path)
        bundle = EvidenceBundle(
            run_id=study.id,
            normalized_image_path=str(normalized_path),
            study_metadata=meta,
        )
        for tool in self.registry:
            check_cancelled()
            label = tool.label
            on_event("stage", {"tool": tool.name, "label": label, "status": "running"})
            output = tool.run(image, meta, study.history)
            check_cancelled()
            artifacts = []
            for name, data in output.images:
                if Path(name).name != name or name in {"normalized.png", "preview.png"}:
                    raise ValueError(f"tool returned an invalid artifact name: {name}")
                target = run_dir / name
                if isinstance(data, np.ndarray):
                    np.save(target, data)
                else:
                    data.save(target)
                artifacts.append(
                    Artifact(name=name, url=f"/a/{study.id}/{name}", kind=artifact_kind(name))
                )
            prefix = EVIDENCE_PREFIX.get(tool.name, tool.name.upper()[:3])
            item = EvidenceItem(
                id=f"{prefix}-1",
                tool=tool.name,
                summary=output.summary,
                payload=output.payload,
                artifacts=artifacts,
            )
            bundle.items.append(item)
            on_event("stage", {"tool": tool.name, "label": label, "status": "done", "summary": output.summary})
            on_event(
                "evidence",
                {
                    "id": item.id,
                    "tool": item.tool,
                    "summary": item.summary,
                    "payload": item.payload,
                    "artifacts": [asdict(a) for a in item.artifacts],
                },
            )
        if getattr(self.settings.models, "cpu_release_auxiliary_models", False):
            import gc
            for tool in self.registry:
                release = getattr(tool, "release_models", None)
                if release is not None:
                    release()
            gc.collect()
        return bundle
