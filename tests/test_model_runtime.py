"""Weight integrity and reasoner lifecycle tests without downloading models."""
from __future__ import annotations

import base64
import io
import json
import queue
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from PIL import Image

from xcortex.config import Settings
from xcortex.core.backends.medgemma import MedGemmaBackend
from xcortex.core.backends.openai_compat import OpenAICompatBackend
from xcortex.core.execution import (
    ExecutionCancelled,
    cancellation_scope,
    check_cancelled,
    current_cancel_event,
    inference_lock,
)
from xcortex.model_files import inspect_checkpoint, inspect_model


def _tensor_file(path: Path, *, name="weight", header=None, data=b"\0" * 4):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(header or {name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return path


def _model_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "gemma3"}), encoding="utf-8")
    return path


def test_model_presence_does_not_accept_empty_or_dummy_weights(tmp_path):
    model = _model_dir(tmp_path / "model")
    assert not inspect_model(model)["installed"]
    (model / "model.safetensors").write_bytes(b"not model weights")
    assert not inspect_model(model)["installed"]
    _tensor_file(model / "model.safetensors")
    state = inspect_model(model)
    assert state["installed"] is True
    assert state["files"] == 1
    assert state["total_bytes"] == (model / "model.safetensors").stat().st_size
    assert "inference not verified" in state["reason"]


def test_sharded_weights_require_every_shard_and_reject_truncation(tmp_path):
    model = _model_dir(tmp_path / "model")
    mapping = {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
    _tensor_file(model / mapping["a"], name="a")
    assert not inspect_model(model)["installed"]
    second = _tensor_file(model / mapping["b"], name="b")
    assert inspect_model(model)["installed"]
    second.write_bytes(second.read_bytes()[:-1])
    assert not inspect_model(model)["installed"]


def test_index_cannot_reference_weights_outside_model_directory(tmp_path):
    model = _model_dir(tmp_path / "model")
    _tensor_file(tmp_path / "outside.safetensors")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}), encoding="utf-8")
    assert not inspect_model(model)["installed"]


@pytest.mark.parametrize("mapping", [[], ["a"], "invalid", None])
def test_corrupted_index_mapping_is_reported_without_crashing(tmp_path, mapping):
    model = _model_dir(tmp_path / "model")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
    assert not inspect_model(model)["installed"]


@pytest.mark.parametrize("header,data", [
    ({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"\0" * 4),
    ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 4),
])
def test_invalid_tensor_shape_or_overlapping_ranges_are_not_complete(tmp_path, header, data):
    model = _model_dir(tmp_path / "model")
    _tensor_file(model / "model.safetensors", header=header, data=data)
    assert not inspect_model(model)["installed"]


def test_torch_checkpoint_supports_official_legacy_and_zip_formats(tmp_path):
    for zipped in (True, False):
        checkpoint = tmp_path / f"weights-{zipped}.pt"
        # A plausible weight artifact also clears the legacy-header sanity floor.
        torch.save({"weight": torch.zeros(300_000)}, checkpoint, _use_new_zipfile_serialization=zipped)
        assert inspect_checkpoint(checkpoint)["installed"], inspect_checkpoint(checkpoint)
    dummy = tmp_path / "dummy.pt"
    dummy.write_text("not a checkpoint", encoding="utf-8")
    assert not inspect_checkpoint(dummy)["installed"]


def test_settings_prefer_complete_weights_folder_and_fall_back_to_legacy(tmp_path):
    settings = Settings()
    settings.paths.weights = str(tmp_path / "weights")
    settings.paths.data = str(tmp_path / "data")
    settings.models.local_files_only = True
    repo = "google/test-model"
    legacy = _model_dir(tmp_path / "data/models/hf/google__test-model")
    _tensor_file(legacy / "model.safetensors")
    assert Path(settings.resolve_model(repo)) == legacy
    current = _model_dir(tmp_path / "weights/hf/google__test-model")
    _tensor_file(current / "model.safetensors")
    assert Path(settings.resolve_model(repo)) == current
    with pytest.raises(FileNotFoundError, match="not completely installed"):
        settings.resolve_model("google/missing")


def test_cancellation_scopes_restore_outer_event_even_after_exception():
    outer, inner = threading.Event(), threading.Event()
    assert current_cancel_event() is None
    with cancellation_scope(outer):
        assert current_cancel_event() is outer
        with cancellation_scope():
            assert current_cancel_event() is outer
        inner.set()
        with pytest.raises(ExecutionCancelled):
            with cancellation_scope(inner):
                pytest.fail("already cancelled scope must not execute")
        assert current_cancel_event() is outer
    assert current_cancel_event() is None


def test_inference_lock_releases_on_cancelled_exit():
    lock, cancel = threading.Lock(), threading.Event()
    with cancellation_scope(cancel):
        with pytest.raises(ExecutionCancelled):
            with inference_lock(lock):
                assert lock.locked()
                cancel.set()
    assert not lock.locked()


def test_waiting_for_inference_lock_can_be_cancelled_without_stealing_it():
    lock, cancel = threading.Lock(), threading.Event()
    lock.acquire()
    timer = threading.Timer(0.01, cancel.set)
    timer.start()
    try:
        with cancellation_scope(cancel), pytest.raises(ExecutionCancelled):
            with inference_lock(lock):
                pytest.fail("cancelled waiter must not acquire another request's lock")
        assert lock.locked()
    finally:
        timer.cancel()
        lock.release()


class _Batch(dict):
    def to(self, device):
        self.device = device
        return self


class _QueueStreamer:
    """Use a tiny bounded wait so timeout regressions run quickly."""
    sentinel = object()

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer, self.options = tokenizer, kwargs
        self.queue = queue.Queue()

    def emit(self, token):
        self.queue.put(token)

    def end(self):
        self.queue.put(self.sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        token = self.queue.get(timeout=0.005)
        if token is self.sentinel:
            raise StopIteration
        return token


def _backend(monkeypatch, generate, *, timeout=2.0):
    import transformers
    settings = Settings()
    settings.models.device = "cpu"
    settings.models.generation_timeout_seconds = timeout
    backend = MedGemmaBackend(settings)
    backend._model = SimpleNamespace(device=torch.device("cpu"), generate=generate)
    backend._processor = SimpleNamespace(tokenizer=object())
    backend._build_inputs = lambda messages: _Batch()
    monkeypatch.setattr(transformers, "TextIteratorStreamer", _QueueStreamer)
    monkeypatch.setattr("xcortex.report.constraints.build_report_constraint", lambda tokenizer: None)
    return backend


@pytest.mark.parametrize("requested,expected", [("bfloat16", torch.bfloat16), ("float16", torch.float32), ("float32", torch.float32)])
def test_medgemma_cpu_retains_bfloat16_without_silent_memory_doubling(requested, expected):
    settings = Settings()
    settings.models.dtype = requested
    assert MedGemmaBackend(settings)._resolve_dtype(torch.device("cpu")) == expected


def test_medgemma_streams_tokens_and_records_only_successful_inference(monkeypatch):
    def generate(**kwargs):
        streamer = kwargs["streamer"]
        assert streamer.options["timeout"] > 0
        assert kwargs["do_sample"] is False
        streamer.emit("hello ")
        streamer.emit("world")
        streamer.end()
    backend = _backend(monkeypatch, generate)
    tokens = []
    assert backend.generate([], on_token=tokens.append) == "hello world"
    assert tokens == ["hello ", "world"]
    assert backend._tested
    assert not backend._inference_lock.locked()


def test_complete_structured_object_stops_after_all_fields_not_early_choice(monkeypatch):
    """The worker stops on whole-object acceptance, not a streamed option ID."""
    class Constraint:
        def is_complete(self, ids):
            return ids[-1].item() == 9

    def generate(**kwargs):
        stopper = kwargs["stopping_criteria"][0]
        kwargs["streamer"].emit('{"answer_choice":"A"')
        assert not stopper(torch.tensor([[0, 1]]), None).item()
        kwargs["streamer"].emit(',"study":{},"findings":[],"impression":"Test","recommendations":[]}')
        assert stopper(torch.tensor([[0, 1, 9]]), None).item()
        kwargs["streamer"].end()
        return torch.tensor([[0, 1, 9]])

    backend = _backend(monkeypatch, generate)
    backend._build_inputs = lambda messages: _Batch(input_ids=torch.tensor([[0]]))
    monkeypatch.setattr("xcortex.report.constraints.build_report_constraint", lambda tokenizer: Constraint())
    text = backend.generate([])
    assert json.loads(text)["impression"] == "Test"
    assert backend.last_run["generation_count"] == 1
    assert backend.last_run["finish_reason"] == "structured_json_complete"
    assert backend.last_run["output_token_count"] == 2
    assert backend.last_run["timings"]["generation_seconds"] >= 0


def test_medgemma_worker_failure_surfaces_without_hanging_streamer(monkeypatch):
    def generate(**kwargs):
        raise ValueError("native failure")
    backend = _backend(monkeypatch, generate)
    with pytest.raises(RuntimeError, match="MedGemma generation failed.*native failure"):
        backend.generate([])
    assert not backend._tested
    assert not backend._inference_lock.locked()


def test_medgemma_empty_generation_is_not_a_success(monkeypatch):
    backend = _backend(monkeypatch, lambda **kwargs: kwargs["streamer"].end())
    with pytest.raises(RuntimeError, match="empty response"):
        backend.generate([])
    assert not backend._tested


def test_medgemma_timeout_keeps_lock_until_native_worker_really_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []
    def generate(**kwargs):
        calls.append(1)
        started.set()
        release.wait(5)
        kwargs["streamer"].end()
    backend = _backend(monkeypatch, generate, timeout=0.02)
    try:
        with pytest.raises(TimeoutError, match="configured timeout"):
            backend.generate([])
        assert started.is_set()
        assert backend._inference_lock.locked()
        assert not backend._tested
        with pytest.raises(TimeoutError, match="previous inference"):
            backend.generate([])
        assert calls == [1]
    finally:
        release.set()
        assert backend._inference_lock.acquire(timeout=2), "native worker did not release its inference lock"
        backend._inference_lock.release()


def test_medgemma_cancellation_keeps_lock_until_native_worker_really_finishes(monkeypatch):
    started, release, cancel = threading.Event(), threading.Event(), threading.Event()
    def generate(**kwargs):
        started.set()
        release.wait(5)
        kwargs["streamer"].end()
    backend = _backend(monkeypatch, generate)
    timer = threading.Timer(0.02, cancel.set)
    timer.start()
    try:
        with cancellation_scope(cancel), pytest.raises(ExecutionCancelled):
            backend.generate([])
        assert started.is_set()
        assert backend._inference_lock.locked()
        assert not backend._tested
    finally:
        timer.cancel()
        release.set()
        assert backend._inference_lock.acquire(timeout=2)
        backend._inference_lock.release()


def test_medgemma_thread_start_failure_does_not_leak_inference_lock(monkeypatch):
    import xcortex.core.backends.medgemma as module
    backend = _backend(monkeypatch, lambda **kwargs: None)
    class BrokenThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise RuntimeError("cannot start worker")
    monkeypatch.setattr(module.threading, "Thread", BrokenThread)
    with pytest.raises(RuntimeError, match="cannot start worker"):
        backend.generate([])
    assert not backend._inference_lock.locked()


def test_medgemma_error_after_stream_end_is_not_reported_as_success(monkeypatch):
    ended, release, done = threading.Event(), threading.Event(), threading.Event()
    def generate(**kwargs):
        kwargs["streamer"].emit("partial answer")
        kwargs["streamer"].end()
        ended.set()
        assert release.wait(3)
        raise RuntimeError("late native failure")
    backend = _backend(monkeypatch, generate)
    results, errors = [], []
    def call():
        try:
            results.append(backend.generate([]))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()
    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    try:
        assert ended.wait(1)
        assert not done.wait(0.05), "EOS alone must not publish an unfinished model result"
    finally:
        release.set()
        caller.join(timeout=3)
    assert done.is_set()
    assert not results
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "late native failure" in str(errors[0])
    assert not backend._tested


def _provider(monkeypatch, content, *, status=200):
    import xcortex.core.backends.openai_compat as module
    original_client = httpx.Client
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(status, content=content, headers={"content-type": "text/event-stream"})
    def client_factory(*args, **kwargs):
        return original_client(*args, transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(module.httpx, "Client", client_factory)
    settings = Settings()
    settings.api.api_key_env = "XCORTEX_RUNTIME_TEST_KEY"
    monkeypatch.setenv(settings.api.api_key_env, "synthetic-test-value")
    return OpenAICompatBackend(settings), seen


def test_remote_vision_requires_explicit_capability_configuration(study_png):
    settings = Settings()
    backend = OpenAICompatBackend(settings)
    messages = [{"role": "user", "content": [{"type": "image", "path": str(study_png)}]}]
    with pytest.raises(RuntimeError, match="vision capability is disabled"):
        backend._flatten(messages)
    settings.api.vision_enabled = True
    flattened = OpenAICompatBackend(settings)._flatten(messages)
    uri = flattened[0]["content"][0]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (320, 320)


def test_provider_sse_parses_final_line_without_newline_and_streams_tokens(monkeypatch):
    content = b': heartbeat\nevent: message\ndata: {"choices": []}\ndata: {"choices":[{"delta":{"content":"hello "}}]}\ndata: {"choices":[{"delta":{"content":"world"}}]}'
    backend, requests = _provider(monkeypatch, content)
    tokens = []
    assert backend.generate([{"role": "user", "content": "hello"}], on_token=tokens.append) == "hello world"
    assert tokens == ["hello ", "world"]
    assert requests[0]["stream"] is True
    assert backend._tested


@pytest.mark.parametrize("content,message", [
    (b'data: {"error":{"message":"model unavailable"}}\n', "Provider stream error"),
    (b'data: not-json\n', "Malformed JSON"),
    (b'data: [DONE]\n', "empty response"),
])
def test_provider_sse_errors_are_not_successful_empty_or_partial_reports(monkeypatch, content, message):
    backend, _ = _provider(monkeypatch, content)
    with pytest.raises(RuntimeError, match=message):
        backend.generate([{"role": "user", "content": "test"}])
    assert not backend._tested


def test_provider_http_error_includes_status_and_actionable_detail(monkeypatch):
    backend, _ = _provider(monkeypatch, b'{"error":"vision unavailable"}', status=400)
    with pytest.raises(RuntimeError, match="HTTP 400.*vision unavailable"):
        backend.generate([{"role": "user", "content": "test"}])
    assert not backend._tested


def test_provider_cancellation_after_first_delta_stops_remaining_output(monkeypatch):
    content = b'data: {"choices":[{"delta":{"content":"first"}}]}\ndata: {"choices":[{"delta":{"content":"second"}}]}\n'
    backend, _ = _provider(monkeypatch, content)
    cancel, tokens = threading.Event(), []
    def on_token(token):
        tokens.append(token)
        cancel.set()
    with cancellation_scope(cancel), pytest.raises(ExecutionCancelled):
        backend.generate([{"role": "user", "content": "test"}], on_token=on_token)
    assert tokens == ["first"]
    assert not backend._tested
