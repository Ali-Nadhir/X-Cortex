"""CPU INT8 execution checks with a tiny real multimodal Gemma architecture."""
from __future__ import annotations

from types import SimpleNamespace
import json
import shutil

import pytest

torch = pytest.importorskip("torch")

from xcortex.core.backends.cpu_quantization import load_cpu_int8_model, quantize_cpu_linears


def _tiny_gemma():
    transformers = pytest.importorskip("transformers")
    text = transformers.Gemma3TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=32,
        max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"],
        query_pre_attn_scalar=16,
    )
    vision = transformers.SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=7,
        vision_use_head=False,
    )
    config = transformers.Gemma3Config(
        text_config=text,
        vision_config=vision,
        mm_tokens_per_image=4,
        image_token_index=100,
        boi_token_index=101,
        eoi_token_index=102,
    )
    return transformers.Gemma3ForConditionalGeneration(config).eval().bfloat16()


def test_quantized_real_gemma_accepts_image_and_cached_decoding():
    """Exercise the real vision, projection, text, cache and LM-head forwards."""
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = _tiny_gemma()
        original_count = sum(isinstance(module, torch.nn.Linear) for module in model.modules())
        metadata = quantize_cpu_linears(model)

        packed = [module for module in model.modules() if isinstance(module, DynamicLinear)]
        assert len(packed) == original_count == metadata["quantized_linear_count"]
        assert not any(isinstance(module, torch.nn.Linear) for module in model.modules())
        assert all(module.weight().dtype == torch.qint8 for module in packed)
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        assert metadata["source_weights_modified"] is False
        assert metadata["activation_dtype"] == "float32"

        ids = torch.tensor([[2, 105, 101, 100, 100, 100, 100, 102, 11, 12]])
        types = torch.zeros_like(ids)
        types[:, 3:7] = 1
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "token_type_ids": types,
            "pixel_values": torch.randn(1, 3, 28, 28),
        }
        with torch.inference_mode():
            prefill = model(**inputs, use_cache=True)
            assert prefill.past_key_values is not None
            assert torch.isfinite(prefill.logits).all()
            decoded = model(
                input_ids=torch.tensor([[13]]),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            assert decoded.logits.shape == (1, 1, 128)
            assert decoded.logits.dtype == torch.float32
            assert torch.isfinite(decoded.logits).all()
            generated = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        assert torch.equal(generated[:, : ids.shape[1]], ids)
        assert ids.shape[1] < generated.shape[1] <= ids.shape[1] + 3


def test_quantization_rejects_model_without_linear_layers():
    with pytest.raises(ValueError, match="no Linear layers"):
        quantize_cpu_linears(torch.nn.Sequential(torch.nn.Embedding(8, 4)))


def test_quantization_requires_available_cpu_engine(monkeypatch):
    monkeypatch.setattr(torch.backends, "quantized", SimpleNamespace(supported_engines=["none"]))
    with pytest.raises(RuntimeError, match="no CPU dynamic INT8 engine"):
        quantize_cpu_linears(torch.nn.Sequential(torch.nn.Linear(4, 2)))


def test_quantization_rejects_weights_not_on_cpu():
    model = torch.nn.Sequential(torch.nn.Linear(4, 2, device="meta"))
    with pytest.raises(ValueError, match="every Linear weight on CPU"):
        quantize_cpu_linears(model)


def test_streaming_loader_materializes_sharded_gemma_and_tied_head(tmp_path):
    """Load actual safetensors shards, including an omitted tied LM-head tensor."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        source = _tiny_gemma()
        embedding = source.get_input_embeddings().weight.detach().float().clone()
        source.save_pretrained(tmp_path, max_shard_size="10KB")
        assert (tmp_path / "model.safetensors.index.json").is_file()
        loaded, metadata = load_cpu_int8_model(tmp_path)
        assert metadata["loading_strategy"] == "streamed_safetensors"
        assert metadata["quantized_linear_count"] > 0
        assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
        assert all(buffer.device.type == "cpu" for buffer in loaded.buffers())
        assert torch.equal(loaded.get_input_embeddings().weight, embedding)
        # The packed head must come from the same tied source, not random init.
        assert torch.allclose(loaded.lm_head.weight().dequantize(), embedding, atol=0.002, rtol=0.02)
        ids = torch.tensor([[2, 105, 101, 100, 100, 100, 100, 102, 11, 12]])
        types = torch.zeros_like(ids)
        types[:, 3:7] = 1
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "token_type_ids": types,
            "pixel_values": torch.randn(1, 3, 28, 28),
        }
        with torch.inference_mode():
            prefill = loaded(**inputs, use_cache=True)
            assert torch.isfinite(prefill.logits).all()
            decoded = loaded(input_ids=torch.tensor([[13]]), past_key_values=prefill.past_key_values)
            assert torch.isfinite(decoded.logits).all()
            generated = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
        assert torch.equal(generated[:, :ids.shape[1]], ids)
        assert generated.shape[1] > ids.shape[1]


def test_streaming_loader_accepts_original_gemma_checkpoint_names(tmp_path):
    from safetensors.torch import load_file, save_file

    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    source = _tiny_gemma()
    source.save_pretrained(current, max_shard_size="10KB")
    for filename in ("config.json", "generation_config.json"):
        shutil.copyfile(current / filename, legacy / filename)

    def original_name(name):
        for modern, old in (
            ("model.language_model.", "language_model.model."),
            ("model.vision_tower.", "vision_tower.vision_model."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
        ):
            if name.startswith(modern):
                return old + name[len(modern):]
        return name

    index = json.loads((current / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for shard in set(index["weight_map"].values()):
        tensors = {original_name(name): tensor for name, tensor in load_file(current / shard).items()}
        save_file(tensors, legacy / shard)
    index["weight_map"] = {original_name(name): shard for name, shard in index["weight_map"].items()}
    (legacy / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    loaded, metadata = load_cpu_int8_model(legacy)
    assert metadata["checkpoint_tensors_consumed"] == len(index["weight_map"])
    assert torch.equal(loaded.get_input_embeddings().weight, source.get_input_embeddings().weight.float())
    with torch.inference_mode():
        assert torch.isfinite(loaded(input_ids=torch.tensor([[2, 5, 6]])).logits).all()


def test_streaming_loader_fails_for_missing_required_tensor(tmp_path):
    _tiny_gemma().save_pretrained(tmp_path, max_shard_size="10KB")
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing = next(name for name in index["weight_map"] if name.endswith("q_proj.weight"))
    del index["weight_map"][missing]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required parameter"):
        load_cpu_int8_model(tmp_path)


@pytest.mark.parametrize("sharded", [False, True])
def test_streaming_loader_uses_pread_for_every_checkpoint_open(tmp_path, monkeypatch, sharded):
    import safetensors

    _tiny_gemma().save_pretrained(tmp_path, max_shard_size="10KB" if sharded else "1GB")
    original = safetensors.safe_open
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", observe)
    _, metadata = load_cpu_int8_model(tmp_path)
    assert calls
    assert all(call.get("backend") == "pread" for call in calls)
    assert metadata["checkpoint_io_backend"] == "pread"
    assert metadata["safetensors_version"] == safetensors.__version__


def test_streaming_loader_has_actionable_error_for_old_safetensors(tmp_path, monkeypatch):
    import safetensors

    _tiny_gemma().save_pretrained(tmp_path)

    def old_safe_open(*args, **kwargs):
        raise TypeError("unexpected keyword argument 'backend'")

    monkeypatch.setattr(safetensors, "safe_open", old_safe_open)
    with pytest.raises(RuntimeError, match="safetensors>=0.8.*pread"):
        load_cpu_int8_model(tmp_path)
