"""Inspect local weight completeness without importing Torch or loading tensors."""
from __future__ import annotations

import json
import struct
import zipfile
import pickletools
import math
from pathlib import Path


def _safetensors_complete(path: Path) -> bool:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if not 2 < length < min(size - 8, 100_000_000):
                return False
            header = json.loads(handle.read(length))
        if not isinstance(header, dict):
            return False
        tensors = [v for k, v in header.items() if k != "__metadata__"]
        if not tensors:
            return False
        widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1}
        offsets = []
        for tensor in tensors:
            shape = tensor["shape"]
            if not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape):
                return False
            a, b = tensor["data_offsets"]
            if type(a) is not int or type(b) is not int or not 0 <= a <= b <= size - 8 - length:
                return False
            if b - a != math.prod(shape) * widths[tensor["dtype"]]:
                return False
            offsets.append((a, b))
        position = 0
        for a, b in sorted(offsets):
            if a != position:
                return False
            position = b
        return position == size - 8 - length
    except (OSError, ValueError, TypeError, KeyError, AttributeError, struct.error):
        return False


def inspect_checkpoint(path) -> dict:
    path = Path(path)
    result = {"installed": False, "reason": "checkpoint missing", "files": 0, "total_bytes": 0}
    if not path.is_file():
        return result
    result.update(files=1, total_bytes=path.stat().st_size)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            valid = any(n.endswith("/data.pkl") for n in names) and any("/data/" in n for n in names)
        result.update(installed=valid, reason="checkpoint archive present; inference not verified" if valid else "invalid checkpoint archive")
    except zipfile.BadZipFile:
        # Older official TorchXRayVision checkpoints use Torch's legacy pickle
        # container. Parse its primitive magic-number prefix without unpickling
        # arbitrary model code. Only actual model loading can verify inference.
        try:
            with path.open("rb") as handle:
                prefix = handle.read(64)
            ops = list(pickletools.genops(prefix))
            valid = len(ops) == 3 and ops[0][0].name == "PROTO" and ops[1][1] == 0x1950A86A20F9469CFC6C and ops[2][0].name == "STOP" and path.stat().st_size > 1_000_000
            result.update(installed=valid, reason="legacy Torch header present; inference not verified" if valid else "invalid checkpoint header")
        except (OSError, ValueError):
            result["reason"] = "not a supported Torch checkpoint"
    except OSError:
        result["reason"] = "unreadable checkpoint"
    return result


def inspect_model(path) -> dict:
    path = Path(path)
    result = {"installed": False, "reason": "model config or weights missing", "files": 0, "total_bytes": 0}
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("model_type"):
            return result
        index = path / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(mapping, dict) or any(not isinstance(name, str) for name in mapping.values()):
                return result
            filenames = sorted(set(mapping.values()))
            if not filenames:
                return result
        else:
            filenames = ["model.safetensors"]
        files = []
        for name in filenames:
            candidate = (path / name).resolve()
            if candidate.parent != path.resolve() or not _safetensors_complete(candidate):
                result["reason"] = f"missing, incomplete or invalid weight shard: {name}"
                return result
            files.append(candidate)
        result.update(installed=True, reason="complete tensor files; inference not verified", files=len(files), total_bytes=sum(f.stat().st_size for f in files))
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return result
