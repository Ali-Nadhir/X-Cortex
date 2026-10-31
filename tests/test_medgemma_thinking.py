"""Thought/final-answer separation without loading or downloading a model."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from xcortex.config import Settings
from xcortex.core.backends.medgemma import MedGemmaBackend, _FinalAnswerFilter
from xcortex.core.execution import ExecutionCancelled, cancellation_scope


def _filtered(chunks):
    answer_filter = _FinalAnswerFilter()
    visible = "".join(answer_filter.feed(chunk) for chunk in chunks)
    answer_filter.finish()
    return visible


def test_every_chunk_boundary_preserves_answer_and_hides_reasoning_json():
    raw = '<unused94>thought\n{"private": "reasoning"}<unused95>{"impression": "Final answer"}'
    expected = '{"impression": "Final answer"}'
    for boundary in range(len(raw) + 1):
        assert _filtered([raw[:boundary], raw[boundary:]]) == expected
    assert _filtered(list(raw)) == expected


def test_repeated_and_nested_thought_blocks_never_escape():
    raw = '<unused94>first<unused95>Answer <unused94>second<unused94>nested<unused95>hidden<unused95>done'
    assert _filtered(list(raw)) == "Answer done"


@pytest.mark.parametrize("raw", [
    'Plain answer.', 'The value is < 3 and > 1.', '<unused940> is ordinary text',
    '{"finding":"Indeterminate", "score":0.28}', '',
])
def test_plain_responses_are_unchanged(raw):
    assert _filtered(list(raw)) == raw


@pytest.mark.parametrize("ending", ["<unused94>thought\nprivate", "<unused94>private<unused9", "<unused94>private<unused95"])
def test_unclosed_thought_or_closing_delimiter_fails_without_leaking(ending):
    answer_filter = _FinalAnswerFilter()
    assert "".join(answer_filter.feed(char) for char in ending) == ""
    with pytest.raises(RuntimeError, match="unfinished thinking.*max_new_tokens"):
        answer_filter.finish()


@pytest.mark.parametrize("ending", ["<", "<u", "<unused9", "<unused94", "<unused95"])
def test_incomplete_reserved_delimiter_is_not_streamed_or_reported_successful(ending):
    answer_filter = _FinalAnswerFilter()
    assert answer_filter.feed(ending) == ""
    with pytest.raises(RuntimeError, match="unfinished thinking"):
        answer_filter.finish()


def test_unmatched_close_is_a_protocol_error():
    with pytest.raises(RuntimeError, match="unmatched thinking delimiter"):
        _filtered(["<unused95>answer"])


def test_long_thought_is_discarded_with_bounded_pending_storage():
    answer_filter = _FinalAnswerFilter()
    assert answer_filter.feed("<unused94>") == ""
    for _ in range(100):
        assert answer_filter.feed("private " * 1000 + "<unused") == ""
        assert len(answer_filter._pending) < len(answer_filter._OPEN)
    assert answer_filter.feed("95>Final") == "Final"
    answer_filter.finish()


class _Batch(dict):
    def to(self, device):
        return self


class _Streamer:
    _end = object()

    def __init__(self, tokenizer, **kwargs):
        self.queue = queue.Queue()

    def emit(self, text):
        self.queue.put(text)

    def end(self):
        self.queue.put(self._end)

    def __next__(self):
        item = self.queue.get(timeout=0.01)
        if item is self._end:
            raise StopIteration
        return item


def _backend(monkeypatch, chunks, *, observed=None):
    import transformers
    settings = Settings()
    settings.models.generation_timeout_seconds = 2
    backend = MedGemmaBackend(settings)

    def generate(**kwargs):
        if observed is not None:
            observed.append(kwargs)
        for chunk in chunks:
            kwargs["streamer"].emit(chunk)
        kwargs["streamer"].end()

    backend._model = SimpleNamespace(device=torch.device("cpu"), generate=generate)
    backend._processor = SimpleNamespace(tokenizer=object())
    backend._build_inputs = lambda messages: _Batch()
    monkeypatch.setattr(transformers, "TextIteratorStreamer", _Streamer)
    monkeypatch.setattr("xcortex.report.constraints.build_report_constraint", lambda tokenizer: None)
    return backend


def test_backend_streams_and_returns_only_final_json(monkeypatch):
    raw = '<unused94>thought\n{"triage": "wrong private draft"}<unused95>{"triage": "low"}'
    backend = _backend(monkeypatch, list(raw))
    delivered = []
    result = backend.generate([], on_token=delivered.append)
    assert result == '{"triage": "low"}' == "".join(delivered)
    assert all("private" not in token for token in delivered)
    assert backend._tested
    assert not backend._inference_lock.locked()


@pytest.mark.parametrize("raw,error", [
    ('<unused94>thought\n{"triage": "private"}', "unfinished thinking"),
    ('<unused94>thought\nprivate<unused95>', "without a final answer"),
    ('<unused94>private<unused95><unused94>truncated', "unfinished thinking"),
])
def test_backend_thought_only_or_truncated_output_fails_without_success_or_tokens(monkeypatch, raw, error):
    backend = _backend(monkeypatch, list(raw))
    delivered = []
    with pytest.raises(RuntimeError, match=error):
        backend.generate([], on_token=delivered.append)
    assert delivered == []
    assert not backend._tested
    assert not backend._inference_lock.locked()


def test_report_generation_passes_built_constraint_to_native_generate(monkeypatch):
    calls, tokenizers = [], []
    backend = _backend(monkeypatch, ['{"triage":"low"}'], observed=calls)
    constraint = lambda batch_id, input_ids: [1]

    def build(tokenizer):
        tokenizers.append(tokenizer)
        return constraint

    monkeypatch.setattr("xcortex.report.constraints.build_report_constraint", build)
    assert backend.generate([], task="report") == '{"triage":"low"}'
    assert tokenizers == [backend._processor.tokenizer]
    assert calls[0]["prefix_allowed_tokens_fn"] is constraint


@pytest.mark.parametrize("task,enabled", [("chat", True), ("report", False)])
def test_chat_and_explicitly_unstructured_reports_do_not_build_constraint(monkeypatch, task, enabled):
    calls = []
    backend = _backend(monkeypatch, ["Final answer"], observed=calls)
    backend.settings.models.structured_reports = enabled

    def unexpected_build(tokenizer):
        pytest.fail("grammar must not be built for this generation mode")

    monkeypatch.setattr("xcortex.report.constraints.build_report_constraint", unexpected_build)
    assert backend.generate([], task=task) == "Final answer"
    assert "prefix_allowed_tokens_fn" not in calls[0]


def test_absent_report_constraint_does_not_pass_null_native_keyword(monkeypatch):
    calls = []
    backend = _backend(monkeypatch, ["Final answer"], observed=calls)
    backend.generate([])
    assert "prefix_allowed_tokens_fn" not in calls[0]


def test_report_constraint_setup_failure_releases_lock_without_native_work(monkeypatch):
    calls = []
    backend = _backend(monkeypatch, ["must not generate"], observed=calls)

    def broken_build(tokenizer):
        raise RuntimeError("report grammar setup failed")

    monkeypatch.setattr("xcortex.report.constraints.build_report_constraint", broken_build)
    with pytest.raises(RuntimeError, match="report grammar setup failed"):
        backend.generate([])
    assert calls == []
    assert not backend._inference_lock.locked()
    assert not backend._tested


def test_cancellation_during_constraint_setup_does_not_start_native_generation(monkeypatch):
    calls, cancel = [], threading.Event()
    backend = _backend(monkeypatch, ["must not generate"], observed=calls)

    def cancelled_build(tokenizer):
        cancel.set()
        return lambda batch_id, input_ids: [1]

    monkeypatch.setattr("xcortex.report.constraints.build_report_constraint", cancelled_build)
    with cancellation_scope(cancel), pytest.raises(ExecutionCancelled):
        backend.generate([])
    assert calls == []
    assert not backend._inference_lock.locked()
