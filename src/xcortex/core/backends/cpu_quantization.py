"""Explicit CPU dynamic INT8 execution, built from the official local weights.

Packed Linear weights are INT8; remaining parameters and activations are FP32.
Convert one layer at a time so a complete FP32 copy is never required.
"""
from __future__ import annotations

import warnings
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from ..execution import check_cancelled


def _cpu_engine() -> str:
    import torch

    supported = [str(engine) for engine in torch.backends.quantized.supported_engines if engine != "none"]
    if not supported:
        raise RuntimeError("This PyTorch build has no CPU dynamic INT8 engine; keep cpu_quantization=none")
    engine = next((candidate for candidate in ("x86", "fbgemm", "onednn") if candidate in supported), supported[0])
    torch.backends.quantized.engine = engine
    return engine


def _source_candidates(name: str) -> list[str]:
    """Gemma3's current and original official checkpoint namespace variants."""
    result = [name]
    for current, legacy in (
        ("model.language_model.", "language_model.model."),
        ("model.vision_tower.", "vision_tower.vision_model."),
        ("model.vision_tower.", "vision_tower."),
        ("model.multi_modal_projector.", "multi_modal_projector."),
    ):
        if name.startswith(current):
            result.append(legacy + name[len(current):])
        if name.startswith(legacy):
            result.append(current + name[len(legacy):])
    return result


def load_cpu_int8_model(resolved: str | Path, *, local_files_only: bool = True,
                        on_progress: Callable[[int, int], None] | None = None) -> tuple[Any, dict[str, Any]]:
    """Materialize Gemma3 directly into INT8 linears without a full weight copy.

    Only local safetensors are accepted. Parameters start on meta, while computed
    nonpersistent buffers remain on CPU. Each Linear is loaded and packed before
    moving to the next; remaining parameters are loaded only after packing all
    linears. The official BF16 checkpoint files are never changed.
    """
    import torch
    import safetensors
    from torch import nn
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    from torch.ao.quantization.qconfig import default_dynamic_qconfig
    from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig

    def open_checkpoint(path: Path) -> Any:
        # Torch's default safetensors mmap uses copy-on-write storage for the
        # entire shard. On Windows this reserves whole-file commit even before
        # any tensor is read, defeating a bounded streaming loader. Pread keeps
        # only an ordinary file handle and allocates individual tensor buffers.
        try:
            return safe_open(str(path), framework="pt", device="cpu", backend="pread")
        except TypeError as exc:
            raise RuntimeError(
                "CPU INT8 streaming requires safetensors>=0.8 with the pread backend; "
                "update safetensors in the xcortex environment without reinstalling PyTorch"
            ) from exc

    root = Path(resolved).resolve()
    if not root.is_dir():
        raise FileNotFoundError("CPU INT8 streaming requires a local installed model directory")
    engine = _cpu_engine()
    config = AutoConfig.from_pretrained(str(root), local_files_only=local_files_only)
    if config.model_type != "gemma3":
        raise ValueError("The CPU INT8 streaming loader currently supports Gemma3 checkpoints only")
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    model.eval()
    tied = dict(getattr(model, "all_tied_weights_keys", {}) or {})
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index has no valid weight map")
    else:
        single = root / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("CPU INT8 streaming requires local safetensors weights")
        with open_checkpoint(single) as reader:
            weight_map = {key: single.name for key in reader.keys()}
    shard_paths: dict[str, Path] = {}
    for filename in set(weight_map.values()):
        if not isinstance(filename, str):
            raise ValueError("Invalid checkpoint shard name")
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or external checkpoint shard: {filename}")
        shard_paths[filename] = path
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not names or "" in names:
        raise ValueError("Expected a composite Gemma3 model with Linear layers")
    consumed: set[str] = set()

    with ExitStack() as stack:
        readers = {name: stack.enter_context(open_checkpoint(path))
                   for name, path in shard_paths.items()}

        # Validate names and shapes from headers before allocating real weights.
        source_map: dict[str, str] = {}
        for name, parameter in model.state_dict().items():
            candidates = _source_candidates(name)
            if name in tied:
                candidates.extend(_source_candidates(tied[name]))
            source = next((candidate for candidate in candidates if candidate in weight_map), None)
            if source is None:
                raise ValueError(f"Checkpoint is missing required parameter: {name}")
            shape = readers[weight_map[source]].get_slice(source).get_shape()
            if tuple(shape) != tuple(parameter.shape):
                raise ValueError(f"Checkpoint shape differs for {name}: {tuple(shape)} vs {tuple(parameter.shape)}")
            source_map[name] = source
        unexpected = sorted(set(weight_map) - set(source_map.values()))
        if unexpected:
            raise ValueError(f"Checkpoint has unexpected parameters: {unexpected[:5]}")

        def read_parameter(name: str, shape) -> Any:
            check_cancelled()
            source = source_map.get(name)
            if source is None:
                raise ValueError(f"Checkpoint is missing required parameter: {name}")
            tensor = readers[weight_map[source]].get_tensor(source)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"Checkpoint shape differs for {name}: {tuple(tensor.shape)} vs {tuple(shape)}")
            consumed.add(source)
            # Own the storage, including for an already-FP32 source checkpoint.
            return tensor.to(dtype=torch.float32, device="cpu", copy=True)

        if on_progress is not None:
            on_progress(0, len(names))
        for number, name in enumerate(names, 1):
            module = model.get_submodule(name)
            module.weight = nn.Parameter(read_parameter(name + ".weight", module.weight.shape), requires_grad=False)
            if module.bias is not None:
                module.bias = nn.Parameter(read_parameter(name + ".bias", module.bias.shape), requires_grad=False)
            module.qconfig = default_dynamic_qconfig
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                packed = DynamicLinear.from_float(module)
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, packed)
            del module
            if on_progress is not None:
                on_progress(number, len(names))

        # The largest shared embedding is not retained while packing lm_head.
        for name, parameter in list(model.named_parameters(remove_duplicate=False)):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            parent._parameters[child_name] = nn.Parameter(
                read_parameter(name, parameter.shape), requires_grad=False,
            )
        # Persisted buffers, when present, are part of the checkpoint contract.
        for module_name, module in model.named_modules():
            for name, buffer in list(module._buffers.items()):
                if buffer is None:
                    continue
                full_name = f"{module_name}.{name}" if module_name else name
                if name not in module._non_persistent_buffers_set:
                    module._buffers[name] = read_parameter(full_name, buffer.shape)
                elif buffer.device.type == "meta":
                    raise ValueError(f"Computed buffer was unexpectedly left on meta: {full_name}")

    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("CPU INT8 streaming left an unmaterialized parameter")
    unexpected = sorted(set(weight_map) - consumed)
    if unexpected:
        raise ValueError(f"Checkpoint has unexpected parameters: {unexpected[:5]}")
    try:
        model.generation_config = GenerationConfig.from_pretrained(str(root), local_files_only=local_files_only)
    except OSError:
        model.generation_config = GenerationConfig.from_model_config(config)
    model.hf_device_map = {"": "cpu"}
    metadata = {
        "mode": "dynamic_int8", "engine": engine, "loading_strategy": "streamed_safetensors",
        "quantized_linear_count": len(names), "quantized_linear_names": names,
        "linear_weight_dtype": "qint8", "remaining_parameter_dtype": "float32", "activation_dtype": "float32",
        "source_weights_modified": False, "checkpoint_tensors_consumed": len(consumed),
        "checkpoint_io_backend": "pread", "safetensors_version": safetensors.__version__,
    }
    return model, metadata


def quantize_cpu_linears(model: Any) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    from torch.ao.quantization.qconfig import default_dynamic_qconfig

    engine = _cpu_engine()
    # Keep names rather than old modules: retaining a module list would retain
    # every upcast FP32 weight until conversion ends, defeating bounded packing.
    layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    names = []
    for name in layers:
        check_cancelled()
        module = model.get_submodule(name)
        if module.weight.device.type != "cpu":
            raise ValueError("CPU dynamic INT8 conversion requires every Linear weight on CPU")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        # Quantized dynamic kernels require FP32 inputs/weights when packing.
        module.float()
        module.qconfig = default_dynamic_qconfig
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            packed = DynamicLinear.from_float(module)
        setattr(parent, child_name, packed)
        names.append(name)
        del module
    if not names:
        raise ValueError("The model has no Linear layers to quantize")
    # Embeddings, norms, convolutions and attention arithmetic use FP32. Packed
    # INT8 weights are not Parameters, and are unaffected by this conversion.
    model.float()
    model.eval()
    return {
        "mode": "dynamic_int8", "engine": engine, "quantized_linear_count": len(names),
        "quantized_linear_names": names, "linear_weight_dtype": "qint8",
        "remaining_parameter_dtype": "float32", "activation_dtype": "float32",
        "source_weights_modified": False,
    }
