"""Local, immutable experiment records; never added to a model prompt."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO

from ..config import Settings
from .chestagentbench import CHESTAGENTBENCH_REPO, CHESTAGENTBENCH_REVISION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Durably publish JSON, retaining the previous snapshot on write failure."""
    encoded = json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def append_record(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def experiment_provenance(settings: Settings, inputs: dict[str, Path], *, experiment_id: str,
                          condition: str, artifact_directory: Path) -> dict[str, Any]:
    """Hash inputs, configuration, source and available local manifests.

    Hashes identify a local snapshot, not clinical eligibility or unknown model
    pretraining overlap. Credentials are not read; API settings contain only an
    environment-variable name. Answer keys are hashed only and remain scoring data.
    """
    source_root = Path(__file__).resolve().parents[1]
    effective_settings = asdict(settings)
    canonical_config = json.dumps(effective_settings, sort_keys=True, separators=(",", ":"), default=str)
    sources = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in sorted(source_root.rglob("*.py"))
    }
    manifests = {}
    weights = Path(settings.paths.weights)
    for path in [weights / "download_manifest.json", weights / "anomalyclip" / "manifest.json"]:
        if path.is_file():
            manifests[str(path.resolve())] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    index = Path(settings.paths.db)
    index_files = {
        path.relative_to(index).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(index.rglob("*")) if path.is_file()
    } if index.exists() else {}
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "condition": condition,
        "created_at": utc_now(),
        "artifact_directory": str(artifact_directory.resolve()),
        "effective_settings": effective_settings,
        "effective_settings_sha256": hashlib.sha256(canonical_config.encode()).hexdigest(),
        "dataset": {"repository": CHESTAGENTBENCH_REPO, "revision": CHESTAGENTBENCH_REVISION,
                    "scope": "development pilot; eligibility is recorded per task"},
        "input_files": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                        for name, path in inputs.items()},
        "source_sha256": sources,
        "model_manifests": manifests,
        "reference_index": {"collection": settings.rag.collection, "files": index_files,
                            "snapshot_method": "file hashes before this experiment; no concurrent writer permitted"},
    }
