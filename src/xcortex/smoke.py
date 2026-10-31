"""Explicit real-model CPU checks. Missing weights are blocked, never mocked."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

from .model_files import inspect_model


def run_smoke(settings, image_path, components, output, *, region="chest", max_new_tokens=64, full_report=False):
    from .utils.dicom import load_study_image
    if not components:
        raise ValueError("Select at least one real model component")
    settings.app.mode = "models"
    settings.models.reasoner = "medgemma"
    report_token_budget = settings.models.max_new_tokens
    settings.models.device = "cpu"
    settings.models.max_new_tokens = max_new_tokens
    image, meta = load_study_image(image_path)
    if meta.get("region") and str(meta["region"]).lower() not in {region, "thorax"}:
        raise ValueError("Smoke-test region conflicts with image metadata")
    meta.setdefault("region", region)
    results = {"device": "cpu", "image": str(Path(image_path).resolve()), "components": {}, "clinical_validation": False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    for component in components:
        started = time.monotonic()
        backend = None
        result: dict[str, Any]
        print(f"CPU smoke: {component}", flush=True)
        try:
            if component == "detector":
                if settings.models.triage_backend == "anomalyclip":
                    from .tools.anomalyclip import AnomalyCLIPTool
                    tool = AnomalyCLIPTool(settings)
                else:
                    from .tools.chestxray import ChestXRayTool
                    tool = ChestXRayTool(settings)
                first = tool.run(image, meta, "")
                second = tool.run(image, meta, "")
                if first.payload != second.payload:
                    raise RuntimeError("Repeated detector output differed")
                result = {"status": "passed", "repeat_identical": True, "evidence": first.payload}
                for name, value in first.images:
                    target = output.parent / f"smoke-{name}"
                    import numpy as np
                    if isinstance(value, np.ndarray):
                        np.save(target, value)
                    else:
                        value.save(target)
                del tool
            elif component == "segmentation":
                from .tools.segmentation import SegmentationTool
                settings.models.sam2_points_per_side = 4
                settings.models.sam2_points_per_batch = 4
                tool = SegmentationTool(settings)
                evidence = tool.run(image, meta, "")
                from PIL import Image
                for name, value in evidence.images:
                    if name.endswith(".png"):
                        rendered = value if isinstance(value, Image.Image) else Image.fromarray(value)
                        rendered.save(output.parent / f"smoke-{name}")
                result = {"status": "passed", "sampling": "4x4 points (smoke only)", "evidence": evidence.payload}
                del tool
            elif component == "siglip":
                from .rag.encoder import MedSigLIPEncoder
                if not inspect_model(settings.model_path(settings.models.siglip_id))["installed"]:
                    raise FileNotFoundError("MedSigLIP weights not installed; accept Google model access, sign in and run xcortex download siglip")
                encoder = MedSigLIPEncoder(settings)
                embedding = encoder.encode(image)
                result = {"status": "passed", "embedding_dimensions": len(embedding), "embedding_norm": float((embedding ** 2).sum() ** 0.5)}
                del encoder
            elif component == "reasoner":
                from .core.backends.medgemma import MedGemmaBackend
                if not inspect_model(settings.model_path(settings.models.medgemma_id))["installed"]:
                    raise FileNotFoundError("MedGemma weights not installed; accept Google model access, sign in and run xcortex download medgemma")
                from PIL import Image
                normalized = output.parent / "normalized.png"
                Image.fromarray(image).save(normalized)
                backend = MedGemmaBackend(settings)
                text = backend.generate([{"role": "user", "content": [
                    {"type": "image", "path": str(normalized)},
                    {"type": "text", "text": "Describe the visible image in one short sentence. This is an integration test."},
                ]}], task="chat")
                result = {"status": "passed", "model": settings.models.medgemma_id, "output": text, "max_new_tokens": max_new_tokens}
            else:
                raise ValueError(f"Unknown component: {component}")
        except FileNotFoundError as exc:
            result = {"status": "blocked", "error": str(exc)}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if backend is not None:
                # A timed-out prefill may still own native tensors. Drain it
                # before loading another model or starting the full report.
                backend.wait_until_idle()
                backend = None
        result["seconds"] = round(time.monotonic() - started, 3)
        results["components"][component] = result
        print(f"  {result['status']} ({result['seconds']}s)", flush=True)
        persist()
        gc.collect()

    if full_report:
        settings.models.max_new_tokens = report_token_budget
        from .core.agent import Agent
        from .core.messages import Study
        agent = None
        try:
            if not inspect_model(settings.model_path(settings.models.medgemma_id))["installed"]:
                raise FileNotFoundError("MedGemma weights are required for a real full-report smoke test")
            agent = Agent(settings)
            report = agent.analyze(Study(id="cpu-smoke", image_path=str(image_path), metadata=meta), output.parent / "full-report")
            results["full_report"] = {"status": "passed", **report}
        except FileNotFoundError as exc:
            results["full_report"] = {"status": "blocked", "error": str(exc)}
        except Exception as exc:
            results["full_report"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            wait_until_idle = getattr(getattr(agent, "backend", None), "wait_until_idle", None)
            if callable(wait_until_idle):
                wait_until_idle()
        persist()
    results["status"] = "passed" if all(r["status"] == "passed" for r in results["components"].values()) and results.get("full_report", {}).get("status", "passed") == "passed" else "incomplete"
    persist()
    return results
