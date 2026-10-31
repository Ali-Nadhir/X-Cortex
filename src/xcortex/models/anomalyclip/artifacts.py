"""Official AnomalyCLIP artifacts; no implicit downloads during inference.

The prompt weights are pinned to an ordinary Git blob in the author's repository.
The backbone checksum is published in OpenAI's official CLIP download URL.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable
import urllib.error
import urllib.request

UPSTREAM_REVISION = "3911738c0867544f545a076ad78f3f11d9ecbfdf"
UPSTREAM_REPOSITORY = "https://github.com/zqhang/AnomalyCLIP"
BACKBONE_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
PROMPT_SHA256 = "94ce202da3e6486a864b904fdfed5057de75846c5834e446fd1d2fe7f97acb44"


@dataclass(frozen=True)
class WeightArtifact:
    filename: str
    url: str
    size_bytes: int
    sha256: str
    git_blob_sha1: str | None = None


PROMPT = WeightArtifact(
    "epoch_15.pth",
    f"https://raw.githubusercontent.com/zqhang/AnomalyCLIP/{UPSTREAM_REVISION}/"
    "checkpoints/9_12_4_multiscale/epoch_15.pth",
    22_631_975, PROMPT_SHA256, "c90f1d9d92d158b436a8fc193f984d4cec8b1fc8",
)
BACKBONE = WeightArtifact(
    "ViT-L-14-336px.pt",
    f"https://openaipublic.azureedge.net/clip/models/{BACKBONE_SHA256}/ViT-L-14-336px.pt",
    934_088_680, BACKBONE_SHA256,
)


def verify_artifact(path: Path, artifact: WeightArtifact) -> bool:
    """Stream checksums without deserializing model code or tensors."""
    if not path.is_file() or path.stat().st_size != artifact.size_bytes:
        return False
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()
    blob.update(f"blob {artifact.size_bytes}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest() == artifact.sha256 and (
        artifact.git_blob_sha1 is None or blob.hexdigest() == artifact.git_blob_sha1
    )


def inspect_anomalyclip(weights_root: str | Path, *, checkpoint: str | Path | None = None, backbone: str | Path | None = None) -> dict:
    """Lightweight existence/size readiness; actual inference is a separate state."""
    directory = Path(weights_root) / "anomalyclip"
    files = []
    for artifact, configured in ((PROMPT, checkpoint), (BACKBONE, backbone)):
        path = Path(configured) if configured is not None else directory / artifact.filename
        size = path.stat().st_size if path.is_file() else 0
        files.append({"path": str(path), "bytes": size, "expected_bytes": artifact.size_bytes,
                      "complete_size": size == artifact.size_bytes})
    return {"installed": all(f["complete_size"] for f in files), "files": files,
            "inference_verified": False,
            "reason": "file sizes only; loader verifies SHA256 before deserialization"}


def _download(path: Path, artifact: WeightArtifact, progress: Callable[[int, int], None] | None) -> None:
    if path.exists():
        if verify_artifact(path, artifact):
            return
        raise RuntimeError(f"Existing AnomalyCLIP artifact failed verification: {path}; preserve/remove it before downloading")
    partial = path.with_name(path.name + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > artifact.size_bytes:
        raise RuntimeError(f"Oversized partial artifact: {partial}")
    if offset == artifact.size_bytes:
        if not verify_artifact(partial, artifact):
            raise RuntimeError(f"Partial artifact checksum mismatch: {partial}")
        partial.replace(path)
        return
    request = urllib.request.Request(artifact.url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as source:
        # Some hosts ignore Range and return a complete body. Never append it.
        resumed = offset > 0 and source.status == 206
        if resumed and not source.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Download returned an unexpected Content-Range")
        if not resumed:
            offset = 0
        with partial.open("ab" if resumed else "wb") as output:
            downloaded = offset
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded > artifact.size_bytes:
                    raise RuntimeError("Download exceeded the pinned artifact size")
                if progress is not None:
                    progress(downloaded, artifact.size_bytes)
    if not verify_artifact(partial, artifact):
        raise RuntimeError(f"Incomplete or checksum-mismatched AnomalyCLIP download: {partial}")
    partial.replace(path)


def download_anomalyclip(weights_root: str | Path, progress: Callable[[int, int], None] | None = None) -> dict:
    directory = Path(weights_root) / "anomalyclip"
    directory.mkdir(parents=True, exist_ok=True)
    for artifact in (PROMPT, BACKBONE):
        _download(directory / artifact.filename, artifact, progress)
    record = {
        "model": "AnomalyCLIP (image, ICLR 2024)", "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION, "license": "MIT",
        "prompt_training": "VisA industrial dataset, per pinned train.sh non-_visa configuration",
        "checkpoint_selection": "official fixed epoch 15; not selected using ChestAgentBench labels",
        "artifacts": [{**asdict(a), "path": str(directory / a.filename), "verified": True} for a in (PROMPT, BACKBONE)],
        "clinical_validation": False,
    }
    temporary = directory / "manifest.json.partial"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    return record
