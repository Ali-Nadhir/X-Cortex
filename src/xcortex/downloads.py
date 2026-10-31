from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from .model_files import inspect_checkpoint, inspect_model

SAM2_CHECKPOINTS = {
    "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
}
HF_REPOS = {"siglip": "google/medsiglip-448", "medgemma": "google/medgemma-1.5-4b-it"}
DETECTOR_FILENAME = "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
DETECTOR_URL = f"https://github.com/mlmed/torchxrayvision/releases/download/v1/{DETECTOR_FILENAME}"


def _record(settings, key, target, **fields):
    manifest = Path(settings.paths.weights) / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    data[key] = {"path": str(Path(target).resolve()), "timestamp": time.time(), **fields}
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(manifest)


def _download_file(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    print(f"downloading {url} -> {target}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "X-Cortex-model-setup"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        done, last = 0, time.monotonic()
        while block := response.read(1024 * 1024):
            handle.write(block)
            done += len(block)
            if time.monotonic() - last >= 10:
                print(f"  {done // 1048576} / {total // 1048576} MiB", flush=True)
                last = time.monotonic()
    if total and temporary.stat().st_size != total:
        raise RuntimeError(f"Incomplete download: {target}")
    temporary.replace(target)
    return target


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sam2(settings, variant=None):
    variant = variant or settings.models.sam2_variant
    target = Path(settings.models.sam2_checkpoint) if variant == settings.models.sam2_variant else Path(settings.paths.weights) / f"{variant}.pt"
    if not inspect_checkpoint(target)["installed"]:
        _download_file(SAM2_CHECKPOINTS[variant], target)
    if not inspect_checkpoint(target)["installed"]:
        raise RuntimeError(f"Invalid SAM 2 checkpoint: {target}")
    _record(settings, variant, target, source=SAM2_CHECKPOINTS[variant], bytes=target.stat().st_size, sha256=_file_digest(target))
    return target


def download_detector(settings):
    if settings.models.detector_weights != "densenet121-res224-all":
        raise ValueError("Explicit downloader currently supports densenet121-res224-all")
    target = Path(settings.models.detector_cache) / DETECTOR_FILENAME
    if not inspect_checkpoint(target)["installed"]:
        _download_file(DETECTOR_URL, target)
    if not inspect_checkpoint(target)["installed"]:
        raise RuntimeError(f"Invalid detector checkpoint: {target}")
    _record(settings, "detector", target, source=DETECTOR_URL, bytes=target.stat().st_size, sha256=_file_digest(target), model=settings.models.detector_weights)
    return target


def download_hf(settings, key):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError
    repo_id = settings.models.medgemma_id if key == "medgemma" else settings.models.siglip_id
    target = settings.model_path(repo_id)
    info = HfApi().model_info(repo_id, files_metadata=True)
    print(f"downloading {repo_id} revision {info.sha} -> {target}", flush=True)
    # snapshot_download resumes interrupted files and checks every shard rather
    # than treating one safetensors file as a completed multi-shard model.
    try:
        snapshot_download(repo_id, revision=info.sha, local_dir=str(target),
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "LICENSE*", "README.md"], max_workers=2)
    except GatedRepoError as exc:
        raise RuntimeError(f"Access blocked for {repo_id}. Accept Google's terms at https://huggingface.co/{repo_id}, then run 'hf auth login' locally. No weights were substituted.") from exc
    status = inspect_model(target)
    if not status["installed"]:
        raise RuntimeError(f"Model download incomplete: {status['reason']}")
    _record(settings, key, target, repo_id=repo_id, revision=info.sha, **status)
    print(f"installed {repo_id}: {status['total_bytes']} tensor bytes", flush=True)
    return target


def run_downloads(settings, which):
    done, errors = [], []
    for item in which:
        try:
            if item == "sam2":
                done.append(download_sam2(settings))
            elif item == "detector":
                done.append(download_detector(settings))
            elif item == "anomalyclip":
                from .models.anomalyclip.artifacts import download_anomalyclip
                done.append(download_anomalyclip(settings.paths.weights))
            elif item in HF_REPOS:
                done.append(download_hf(settings, item))
            else:
                raise ValueError(f"Unknown model key: {item}")
        except Exception as exc:
            errors.append(f"{item}: {exc}")
            print(f"download incomplete: {item}: {exc}", flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return done
