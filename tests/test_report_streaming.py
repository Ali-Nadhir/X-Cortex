"""Single-pass report publication preserves schema checks and private drafts."""
from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from xcortex.core.agent import Agent, AgentError
from xcortex.core.execution import ExecutionCancelled
from xcortex.core.messages import EvidenceBundle, EvidenceItem, Study


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
        "limitations": [],
    }


def _agent(settings, responses):
    agent = Agent(settings)
    triage = copy.deepcopy(_report()["triage_summary"])
    triage.update(pathology_scores={"Cardiomegaly": 0.281119}, detections=[])
    bundle = EvidenceBundle(
        run_id="report-stream",
        items=[EvidenceItem("TRI-1", "triage", "Uncalibrated scores", triage)],
        normalized_image_path="unused-normalized.png",
        study_metadata={"region": "chest"},
    )
    agent.pipeline = SimpleNamespace(run=lambda *_: bundle)
    attempts = []

    def generate(messages, on_token, task):
        assert task == "report"
        response = responses[len(attempts)]
        attempts.append(copy.deepcopy(messages))
        text = response if isinstance(response, str) else json.dumps(response)
        for index in range(0, len(text), 17):
            on_token(text[index:index + 17])
        return text

    agent.backend = SimpleNamespace(generate=generate)
    return agent, attempts


def _image_observation_report():
    report = _report()
    report["findings"] = [{
        "structure": "heart",
        "observation": "Cardiomegaly is present.",
        "evidence_ids": ["Q0"],
    }]
    return report


def test_single_report_is_published_with_nonblocking_metadata_audit(settings, tmp_path):
    agent, attempts = _agent(settings, [_image_observation_report(), _report()])
    events = []
    result = agent.analyze(
        Study("report-stream", "unused.png"), tmp_path,
        lambda kind, data: events.append((kind, data)),
    )

    assert len(attempts) == 1
    assert [kind for kind, _ in events] == ["token", "report"]
    text = "".join(data["text"] for kind, data in events if kind == "token")
    assert text == json.dumps(result["report"], ensure_ascii=False)
    assert json.loads(text) == result["report"]
    assert "Cardiomegaly" in text
    assert result["verification"]["ok"] is True
    assert result["verification"]["scope"] == "deterministic_metadata"
    assert result["verification"]["clinical_validation"] is False


@pytest.mark.parametrize("responses", [
    ["not valid JSON", "not valid JSON"],
    [{"findings": []}, {"findings": []}],
])
def test_invalid_single_report_publishes_no_clinical_text(settings, tmp_path, responses):
    agent, attempts = _agent(settings, responses)
    events = []
    with pytest.raises(AgentError, match="single permitted generation"):
        agent.analyze(
            Study("report-stream", "unused.png"), tmp_path,
            lambda kind, data: events.append((kind, data)),
        )
    assert len(attempts) == 1
    assert events == []


def test_cancellation_after_verification_prevents_publication(settings, tmp_path):
    agent, _ = _agent(settings, [_report()])
    cancel = threading.Event()
    real_verify = agent.verifier.verify

    def verify_then_cancel(*args):
        verified = real_verify(*args)
        assert verified.ok
        cancel.set()
        return verified

    agent.verifier.verify = verify_then_cancel
    events = []
    with pytest.raises(ExecutionCancelled):
        agent.analyze(
            Study("report-stream", "unused.png"), tmp_path,
            lambda kind, data: events.append((kind, data)), cancel_event=cancel,
        )
    assert events == []


def test_chat_still_forwards_tokens_immediately(settings):
    agent = Agent(settings)
    events = []

    def generate(messages, on_token, task):
        assert task == "chat"
        on_token("First ")
        assert events == [("token", {"text": "First "})]
        on_token("answer.")
        return "First answer."

    agent.backend = SimpleNamespace(generate=generate)
    assert agent.ask([], "Question", lambda kind, data: events.append((kind, data))) == "First answer."
    assert events == [("token", {"text": "First "}), ("token", {"text": "answer."})]
