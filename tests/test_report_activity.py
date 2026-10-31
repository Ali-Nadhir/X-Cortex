"""Private report generation updates liveness without exposing draft text."""
from __future__ import annotations

import io
import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from xcortex.core.execution import current_cancel_event
from xcortex.core.messages import EvidenceBundle, EvidenceItem
from xcortex.server.app import create_app


def _report():
    return {
        "study": {"region": "chest"},
        "triage_summary": {
            "abnormality_probability": 0.281119,
            "threshold": 0.33,
            "band": "low",
            "calibrated": False,
            "score_type": "max_pathology_score",
            "method": "test classifier",
        },
        "findings": [],
        "impression": "Tool evidence reviewed.",
        "recommendations": [],
    }


def _client(settings, generate):
    settings.agent.request_timeout_seconds = 3
    settings.agent.stream_idle_timeout_seconds = 0.2
    app = create_app(settings)
    bundle = EvidenceBundle(
        run_id="private-activity",
        items=[EvidenceItem("TRI-1", "triage", "Uncalibrated scores", _report()["triage_summary"])],
        normalized_image_path="unused-normalized.png",
        study_metadata={"region": "chest"},
    )
    app.state.agent.pipeline = SimpleNamespace(run=lambda *_: bundle)
    app.state.agent.backend = SimpleNamespace(generate=generate)
    api = TestClient(app)
    buffer = io.BytesIO()
    Image.new("L", (16, 16), 128).save(buffer, "PNG")
    upload = api.post("/api/upload", files={"file": ("study.png", buffer.getvalue(), "image/png")})
    assert upload.status_code == 200
    return app, api, upload.json()["session_id"]


def test_private_draft_activity_prevents_idle_timeout_without_streaming_drafts(settings):
    draft = "PRIVATE DRAFT: Cardiomegaly is present."
    calls = []

    def generate(messages, on_token, task):
        assert task == "report"
        for _ in range(20):
            on_token(draft)
            calls.append(time.monotonic())
            time.sleep(0.025)
        return json.dumps(_report())

    app, api, session_id = _client(settings, generate)
    response = api.post("/api/analyze", json={"session_id": session_id})
    assert calls[-1] - calls[0] > settings.agent.stream_idle_timeout_seconds
    assert "event: error" not in response.text
    assert '"status": "completed"' in response.text
    assert "event: token" in response.text
    assert "event: report" in response.text
    assert draft not in response.text
    assert "Cardiomegaly" not in response.text
    assert "event: activity" not in response.text
    assert app.state.sessions.get(session_id)["status"] == "ready"


def test_truly_idle_report_generation_still_times_out_without_clinical_publication(settings):
    def generate(messages, on_token, task):
        assert current_cancel_event().wait(2), "idle generation was not cancelled"
        return json.dumps(_report())

    app, api, session_id = _client(settings, generate)
    response = api.post("/api/analyze", json={"session_id": session_id})
    assert "event: error" in response.text
    assert "timed_out" in response.text
    assert "event: token" not in response.text
    assert "event: report" not in response.text
    assert app.state.sessions.get(session_id)["status"] == "timed_out"


def test_private_draft_activity_does_not_extend_absolute_request_deadline(settings):
    def generate(messages, on_token, task):
        while True:
            on_token("PRIVATE DRAFT")
            time.sleep(0.01)

    app, api, session_id = _client(settings, generate)
    settings.agent.request_timeout_seconds = 0.25
    response = api.post("/api/analyze", json={"session_id": session_id})
    assert "event: error" in response.text
    assert "timed_out" in response.text
    assert "event: token" not in response.text
    assert "PRIVATE DRAFT" not in response.text
    assert app.state.sessions.get(session_id)["status"] == "timed_out"
