from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from ..core.execution import check_cancelled
from ..utils.dicom import SUPPORTED_SUFFIXES, load_study_image
from .encoder import MedSigLIPEncoder
from .metadata import normalize_region
from .store import VectorStore

REGION_PATTERN = re.compile(r"XR_([A-Z]+)", re.IGNORECASE)
POSITIVE_LABELS = {"positive", "abnormal", "1", "1.0", "true"}
NEGATIVE_LABELS = {"negative", "normal", "0", "0.0", "false"}
REFERENCE_SPLITS = {"train", "training", "reference"}


def region_of(path: str) -> str:
    match = REGION_PATTERN.search(path)
    return normalize_region(match.group(1)) if match else "unknown"


def _label(value: str) -> str | None:
    value = value.strip().lower()
    if value in POSITIVE_LABELS:
        return "abnormal"
    if value in NEGATIVE_LABELS:
        return "normal"
    if value == "unspecified":
        # An explicitly unknown reference diagnosis is not a negative label.
        return "unspecified"
    return None


def read_manifest_records(manifest_path: str) -> list[dict]:
    """Read path,label CSVs or explicit image_path,label,region,patient_id,
    study_id,reference,source,split columns. Unknown labels are never made normal.
    """
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as handle:
        raw = [row for row in csv.reader(handle) if row and any(value.strip() for value in row)]
    if not raw:
        return []
    headers = [value.strip().lower() for value in raw[0]]
    has_header = headers[0] in {"path", "image_path", "image", "study_path"}
    records = []
    for row in raw[1:] if has_header else raw:
        if len(row) < 2:
            continue
        if has_header:
            record: dict[str, Any] = dict(zip(headers, (value.strip() for value in row)))
            path = next((record.get(key) for key in ("image_path", "path", "image", "study_path") if record.get(key)), "")
            label = _label(record.get("label", ""))
        else:
            path, label = row[0].strip(), _label(row[1])
            record = {}
        if not path or label is None:
            continue
        record.update(path=path, label=label)
        if has_header and "split" in headers:
            record["split"] = record.get("split", "").strip().lower()
            record["has_split_column"] = True
        records.append(record)
    return records


def read_manifest(manifest_path: str) -> list[tuple[str, str]]:
    return [(record["path"], record["label"]) for record in read_manifest_records(manifest_path)]


def _reference_split(record: dict) -> str:
    if record.get("has_split_column"):
        split = record.get("split", "")
    else:
        parts = {part.lower() for part in record["path"].replace("\\", "/").split("/")}
        split = "train" if "train" in parts else ""
    if split not in REFERENCE_SPLITS:
        raise ValueError(
            f"reference indexing rejected '{record['path']}': use only training/reference cases. "
            "Supply a CSV split column with train or reference; validation/test cases must not be indexed."
        )
    return split


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    direct = base / candidate
    if direct.exists():
        return direct.resolve()
    # Official MURA manifests include their root folder in each relative path.
    if candidate.parts and candidate.parts[0].lower() == base.name.lower():
        return base.joinpath(*candidate.parts[1:]).resolve()
    return direct.resolve()


def run_indexing(settings, manifest_path: str, root: str | None = None, batch: int = 16) -> int:
    if batch <= 0:
        raise ValueError("index batch size must be positive")
    records = read_manifest_records(manifest_path)
    if not records:
        raise ValueError("reference manifest contains no supported labeled cases")
    # Validate the whole manifest before writing any rows to avoid a mixed index
    # when a held-out row appears late in an otherwise valid input file.
    splits = [_reference_split(record) for record in records]
    base = Path(root).resolve() if root else Path(manifest_path).resolve().parent
    work = []
    for record, split in zip(records, splits):
        resolved = _resolve(record["path"], base)
        if resolved.is_dir():
            paths = sorted(path for path in resolved.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
        elif resolved.is_file():
            paths = [resolved]
        else:
            raise FileNotFoundError(f"reference image or study directory does not exist: {resolved}")
        if not paths:
            raise ValueError(f"reference study directory has no supported images: {resolved}")
        work.extend((record, split, path) for path in paths)
    encoder = MedSigLIPEncoder.shared(settings)
    store = VectorStore(settings)
    total = 0
    for start in range(0, len(work), batch):
        ids, embeddings, metadatas = [], [], []
        for record, split, resolved in work[start:start + batch]:
            check_cancelled()
            image, metadata = load_study_image(resolved)
            region = normalize_region(record.get("region") or metadata.get("region") or region_of(str(resolved)))
            patient_match = re.search(r"(?:^|[/\\])(patient[^/\\]+)(?:[/\\]|$)", str(resolved), re.IGNORECASE)
            patient_id = record.get("patient_id") or metadata.get("patient_id") or (patient_match.group(1) if patient_match else "")
            study_id = record.get("study_id") or metadata.get("study_id") or hashlib.sha256(str(resolved.parent).encode("utf-8")).hexdigest()[:24]
            reference = record.get("reference", "").strip()
            embeddings.append(encoder.encode(image))
            metadatas.append({
                "region": region,
                "label": record["label"],
                "reference": reference or f"{region} reference case — dataset label: {record['label']}",
                "reference_kind": "provided_reference_text" if reference else "dataset_label",
                "source": record.get("source", "manifest"),
                "dataset_split": split,
                "patient_id": str(patient_id),
                "study_id": str(study_id),
                "image_sha256": metadata["image_sha256"],
                "image_path": str(resolved),
                "encoder_model": settings.models.siglip_id,
            })
            hash_suffix = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
            clean_stem = re.sub(r"[^A-Za-z0-9]+", "_", resolved.stem)[:40]
            ids.append(f"case_{clean_stem}_{hash_suffix}")
        store.upsert(ids, embeddings, metadatas)
        total += len(ids)
        print(f"indexed {total}/{len(work)}")
    print(f"done: {total} reference images in collection '{settings.rag.collection}'")
    return total
