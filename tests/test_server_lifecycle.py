from __future__ import annotations

import asyncio
import io
import json
import struct
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from xcortex.server.app import AnalyzeRequest, SessionBusy, SessionStore, create_app


def upload(api: TestClient) -> str:
    buffer = io.BytesIO()
    Image.new("L", (32, 32), 128).save(buffer, "PNG")
    response = api.post("/api/upload", files={"file": ("study.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    return response.json()["session_id"]


def result():
    return {"report": {"study": {"region": "wrist"}}, "verification": {"ok": True}, "evidence": []}


def eventually(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "worker did not complete within the test deadline"


@pytest.mark.parametrize("content", [b"{broken", b'{}', b'{"sessions": []}', b'{"sessions": {"s1": []}}'])
def test_corrupt_state_is_quarantined_before_new_writes(tmp_path, content):
    target = tmp_path / "sessions.json"
    target.write_bytes(content)
    store = SessionStore(target)
    assert store.recovery["status"] == "quarantined"
    preserved = Path(store.recovery["quarantine_path"])
    assert preserved.read_bytes() == content
    store.create("new", {"id": "new", "status": "pending"})
    assert preserved.read_bytes() == content
    assert set(json.loads(target.read_text())["sessions"]) == {"new"}


def test_restart_recovers_analysis_and_followup_without_discarding_report(tmp_path):
    target = tmp_path / "sessions.json"
    target.write_text(json.dumps({"sessions": {
        "analysis": {"id": "analysis", "status": "running", "history": "retained"},
        "chat": {
            "id": "chat", "status": "ready", "report": {"impression": "retained"},
            "active_operation": {"kind": "ask", "id": "old"},
            "turns": [{"role": "user", "content": "unanswered"}],
        },
    }}))
    store = SessionStore(target)
    assert store.get("analysis")["status"] == "interrupted"
    assert store.get("analysis")["history"] == "retained"
    chat = store.get("chat")
    assert chat["status"] == "ready"
    assert chat["report"]["impression"] == "retained"
    assert chat["last_operation_status"] == "interrupted"
    assert "active_operation" not in chat
    assert set(store.recovery["interrupted_sessions"]) == {"analysis", "chat"}
    assert SessionStore(target).recovery["status"] == "clean"


def test_claim_and_snapshot_are_atomic_and_failed_write_rolls_back(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.json")
    store.create("s", {"id": "s", "status": "pending", "turns": []})
    snapshot = store.get("s")
    snapshot["turns"].append({"role": "user", "content": "not committed"})
    assert store.get("s")["turns"] == []
    flush = store._flush

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_flush", fail)
    with pytest.raises(OSError, match="disk full"):
        store.claim("s", "analyze", status="running")
    assert not store.busy("s")
    assert store.get("s")["status"] == "pending"
    monkeypatch.setattr(store, "_flush", flush)
    token, _ = store.claim("s", "analyze", status="running")
    with pytest.raises(SessionBusy):
        store.claim("s", "ask")
    assert not store.finish("s", "stale-token", status="ready")
    assert store.finish("s", token, status="ready")


def test_simultaneous_analysis_has_one_atomic_winner(settings):
    app = create_app(settings)
    api = TestClient(app)
    sid = upload(api)
    start = threading.Barrier(2)
    release = threading.Event()
    calls = []

    def analyze(study, run_dir, on_event, *, cancel_event, on_activity=None):
        calls.append(study.id)
        assert release.wait(3)
        return result()

    app.state.agent.analyze = analyze

    def request():
        start.wait(3)
        return api.post("/api/analyze", json={"session_id": sid, "history": "retained"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request) for _ in range(2)]
        try:
            done, _ = wait(futures, timeout=3, return_when=FIRST_COMPLETED)
            assert len(done) == 1
            assert next(iter(done)).result().status_code == 409
            assert calls == [sid]
        finally:
            release.set()
        assert sorted(f.result(timeout=3).status_code for f in futures) == [200, 409]
    assert app.state.sessions.get(sid)["status"] == "ready"
    assert app.state.sessions.get(sid)["history"] == "retained"


def test_followups_exclude_analysis_and_preserve_ordered_prior_turns(settings):
    app = create_app(settings)
    api = TestClient(app)
    sid = upload(api)
    app.state.sessions.update(sid, status="ready", report=result()["report"])
    entered, release = threading.Event(), threading.Event()
    observed_prior = []

    def ask(evidence, question, on_event, *, history, report, prior_turns, cancel_event):
        observed_prior.append(prior_turns)
        entered.set()
        assert release.wait(3)
        on_event("token", {"text": f"answer {question}"})
        return f"answer {question}"

    app.state.agent.ask = ask
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/api/ask", json={"session_id": sid, "question": "first"})
        try:
            assert entered.wait(3)
            assert api.post("/api/ask", json={"session_id": sid, "question": "second"}).status_code == 409
            assert api.post("/api/analyze", json={"session_id": sid}).status_code == 409
        finally:
            release.set()
        assert pending.result(timeout=3).status_code == 200
    assert api.post("/api/ask", json={"session_id": sid, "question": "second"}).status_code == 200
    turns = SessionStore(Path(settings.paths.state)).get(sid)["turns"]
    assert [t["content"] for t in turns] == ["first", "answer first", "second", "answer second"]
    assert observed_prior[0] == []
    assert observed_prior[1] == turns[:2]


def test_worker_admission_is_bounded_across_sessions(settings):
    app = create_app(settings)
    api = TestClient(app)
    first, second = upload(api), upload(api)
    entered, release = threading.Event(), threading.Event()

    def analyze(study, run_dir, on_event, *, cancel_event, on_activity=None):
        entered.set()
        assert release.wait(3)
        return result()

    app.state.agent.analyze = analyze
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(api.post, "/api/analyze", json={"session_id": first})
        try:
            assert entered.wait(3)
            assert api.post("/api/analyze", json={"session_id": second}).status_code == 503
            assert app.state.sessions.get(second)["status"] == "pending"
        finally:
            release.set()
        assert active.result(timeout=3).status_code == 200


def test_timeout_cannot_be_overwritten_by_late_success_or_admit_overlapping_work(settings):
    settings.agent.request_timeout_seconds = 0.08
    settings.agent.stream_idle_timeout_seconds = 2
    app = create_app(settings)
    api = TestClient(app)
    sid = upload(api)
    release = threading.Event()

    def ignores_cancel(study, run_dir, on_event, *, cancel_event, on_activity=None):
        assert release.wait(3)
        return result()

    app.state.agent.analyze = ignores_cancel
    try:
        response = api.post("/api/analyze", json={"session_id": sid})
        assert "event: error" in response.text
        assert "timed_out" in response.text
        assert app.state.sessions.get(sid)["status"] == "timed_out"
        assert app.state.sessions.busy(sid)
        assert api.post("/api/analyze", json={"session_id": sid}).status_code == 409
    finally:
        release.set()
    eventually(lambda: not app.state.active_runs)
    record = app.state.sessions.get(sid)
    assert record["status"] == "timed_out"
    assert not record.get("report")
    assert "active_operation" not in record
    assert not app.state.sessions.busy(sid)
    assert api.post("/api/analyze", json={"session_id": sid}).status_code == 200
    assert app.state.sessions.get(sid)["status"] == "ready"


def test_disconnect_cancels_worker_and_queue_does_not_grow_without_consumer(settings):
    app = create_app(settings)
    api = TestClient(app)
    sid = upload(api)

    def emit_forever(study, run_dir, on_event, *, cancel_event, on_activity=None):
        for _ in range(10000):
            on_event("token", {"text": "bounded"})
        return result()

    app.state.agent.analyze = emit_forever
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/analyze")
    response = endpoint(AnalyzeRequest(session_id=sid))
    run = next(iter(app.state.active_runs.values()))
    eventually(lambda: run.events.qsize() == run.events.maxsize)
    assert run.events.maxsize > 0

    async def disconnect():
        await anext(response.body_iterator)
        await response.body_iterator.aclose()

    asyncio.run(disconnect())
    assert run.cancel_event.is_set()
    assert run.finished.wait(3)
    record = app.state.sessions.get(sid)
    assert record["status"] == "cancelled"
    assert record["last_operation_status"] == "cancelled"
    assert not record.get("report")


def test_inference_failure_is_streamed_and_persisted(settings):
    app = create_app(settings)
    api = TestClient(app)
    sid = upload(api)

    def broken(*args, **kwargs):
        raise RuntimeError("model could not load")

    app.state.agent.analyze = broken
    response = api.post("/api/analyze", json={"session_id": sid})
    assert "event: error" in response.text
    assert "model could not load" in response.text
    record = app.state.sessions.get(sid)
    assert record["status"] == "failed"
    assert record["last_error"] == "model could not load"
    assert not record.get("active_operation")
    assert not app.state.sessions.busy(sid)


def test_health_does_not_equate_dummy_files_or_structural_weights_with_inference(settings):
    settings.app.mode = "models"
    settings.paths.weights = str(Path(settings.paths.data) / "weights")
    model_path = settings.model_path(settings.models.medgemma_id)
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type": "gemma3"}')
    settings.models.sam2_checkpoint = str(Path(settings.paths.data) / "dummy.pt")
    Path(settings.models.sam2_checkpoint).write_bytes(b"not weights")
    api = TestClient(create_app(settings))
    first = api.get("/api/health").json()
    assert first["ready"] is False
    assert first["components"]["sam2"]["installed"] is False
    assert first["components"]["medgemma"]["installed"] is False
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (model_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    second = api.get("/api/health").json()
    assert second["components"]["medgemma"]["installed"] is True
    assert second["components"]["medgemma"]["loaded"] is False
    assert second["components"]["medgemma"]["tested"] is False
    assert second["medgemma_ready"] is False
    assert second["ready"] is False


def test_explicit_and_configured_anatomy_are_persisted_without_overriding_dicom(settings):
    settings.models.default_region = "chest"
    app = create_app(settings)
    api = TestClient(app)
    seen_meta = []

    def analyze(study, run_dir, on_event, *, cancel_event, on_activity=None):
        seen_meta.append(study.metadata)
        return result()

    app.state.agent.analyze = analyze
    configured = upload(api)
    assert api.post("/api/analyze", json={"session_id": configured}).status_code == 200
    assert seen_meta[-1]["region"] == "chest"
    assert seen_meta[-1]["region_source"] == "configured"
    explicit = upload(api)
    assert api.post("/api/analyze", json={"session_id": explicit, "region": "THORAX"}).status_code == 200
    assert seen_meta[-1]["region"] == "chest"
    assert seen_meta[-1]["region_source"] == "user"
    dicom = upload(api)
    app.state.sessions.update(dicom, meta={"region": "WRIST", "source_format": ".dcm"})
    assert api.post("/api/analyze", json={"session_id": dicom, "region": "chest"}).status_code == 400
    assert app.state.sessions.get(dicom)["status"] == "pending"
    assert api.post("/api/analyze", json={"session_id": dicom}).status_code == 200
    assert seen_meta[-1]["region"] == "WRIST"
    invalid = upload(api)
    assert api.post("/api/analyze", json={"session_id": invalid, "region": "arbitrary text"}).status_code == 400
    assert not app.state.sessions.busy(invalid)


def test_shutdown_requests_cooperative_cancellation(settings):
    app = create_app(settings)
    entered = threading.Event()

    def analyze(study, run_dir, on_event, *, cancel_event, on_activity=None):
        entered.set()
        assert cancel_event.wait(3)
        raise RuntimeError("cancelled during shutdown")

    app.state.agent.analyze = analyze
    with TestClient(app) as api:
        sid = upload(api)
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/analyze")
        endpoint(AnalyzeRequest(session_id=sid))
        assert entered.wait(3)
    eventually(lambda: not app.state.active_runs)
    assert app.state.sessions.get(sid)["last_operation_status"] == "cancelled"


def test_native_background_inference_keeps_session_claim_and_capacity(settings):
    app = create_app(settings)
    api = TestClient(app)
    sid, other = upload(api), upload(api)
    native_release, native_waiting = threading.Event(), threading.Event()

    def wait_until_idle():
        native_waiting.set()
        assert native_release.wait(3)

    def stopped_waiting(*args, **kwargs):
        raise TimeoutError("native prefill is still stopping")

    app.state.agent.backend.wait_until_idle = wait_until_idle
    app.state.agent.analyze = stopped_waiting
    try:
        response = api.post("/api/analyze", json={"session_id": sid})
        assert "native prefill is still stopping" in response.text
        assert native_waiting.wait(1)
        assert app.state.sessions.get(sid)["last_operation_status"] == "failed"
        assert app.state.sessions.busy(sid)
        assert len(app.state.active_runs) == 1
        assert api.post("/api/analyze", json={"session_id": sid}).status_code == 409
        assert api.post("/api/analyze", json={"session_id": other}).status_code == 503
    finally:
        native_release.set()
    eventually(lambda: not app.state.active_runs)
    assert not app.state.sessions.busy(sid)
    assert app.state.sessions.get(sid)["status"] == "failed"
