"""The fixed LangGraph executes all collection then exactly one interpretation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from xcortex.core.agent import Agent, AgentError
from xcortex.core.execution import ExecutionCancelled, check_cancelled, current_cancel_event
from xcortex.core.messages import Artifact, EvidenceBundle, EvidenceItem, Study
from xcortex.core.workflow import assemble_packet
from xcortex.report.prompts import SYSTEM_PROMPT
from xcortex.report.schema import ClinicalReport
from xcortex.report.verifier import GroundingVerifier, ReportAuditor


def _report():
    return {
        "study": {"region": "chest"},
        "triage_summary": None,
        "findings": [{
            "structure": "cardiac silhouette", "observation": "Cardiomegaly is present.",
            "evidence_ids": ["Q0"], "measurements": [],
        }],
        "impression": "Cardiomegaly.", "recommendations": [], "limitations": [],
    }


def _bundle(reference_count=5):
    return EvidenceBundle(
        run_id="single", normalized_image_path="normalized.png", study_metadata={"region": "chest"},
        items=[
            EvidenceItem("TRI-1", "triage", "No proposals", {"detections": []}, [
                Artifact("overlay-anomaly.png", "/a/single/overlay-anomaly.png", "overlay"),
            ]),
            EvidenceItem("RAG-1", "rag", "References", {
                "matches": [{"case_id": f"reference-{i}"} for i in range(reference_count)],
            }, [Artifact(f"ref_{i}.png", f"/a/single/ref_{i}.png", "reference") for i in range(reference_count)]),
        ],
    )


def _agent(settings, generate, bundle=None):
    agent = Agent(settings)
    agent.pipeline = SimpleNamespace(run=lambda *_: bundle or _bundle())
    agent.backend = SimpleNamespace(generate=generate)
    return agent


def test_graph_is_fixed_and_only_one_generation_with_independent_visual_findings(settings, tmp_path):
    settings.agent.max_attempts = 99  # Historical configuration cannot add calls.
    settings.rag.attach_reference_images = False
    settings.rag.max_reference_images = 0
    messages_seen = []

    def generate(messages, on_token, task):
        assert task == "report"
        messages_seen.append(copy.deepcopy(messages))
        on_token("PRIVATE WORKING DRAFT")
        return json.dumps(_report())

    agent = _agent(settings, generate)
    events = []
    result = agent.analyze(Study("single", "unused.png", metadata={"question": "Assess the heart size."}), tmp_path,
                           lambda kind, data: events.append((kind, data)))
    assert len(messages_seen) == 1
    assert result["workflow"] == {
        "engine": "langgraph", "nodes": ["collect", "assemble", "reason", "format"], "generation_count": 1,
    }
    graph = agent.workflow.graph.get_graph()
    assert {(edge.source, edge.target) for edge in graph.edges} == {
        ("__start__", "collect"), ("collect", "assemble"), ("assemble", "reason"),
        ("reason", "format"), ("format", "__end__"),
    }
    assert result["report"]["findings"][0]["evidence_ids"] == ["Q0"]
    assert result["report"]["triage_summary"] is None
    assert result["verification"]["ok"] is True
    assert result["verification"]["clinical_validation"] is False
    assert result["verification"]["scope"] == "deterministic_metadata"
    assert [kind for kind, _ in events] == ["token", "report"]
    assert "PRIVATE WORKING DRAFT" not in json.dumps(events)
    manifest = result["context_manifest"]
    assert [image["id"] for image in manifest] == ["Q0", "Q1", "R1", "R2", "R3", "R4", "R5"]
    assert all("path" not in image for image in manifest)
    content = messages_seen[0][1]["content"]
    assert len([part for part in content if part["type"] == "image"]) == 7
    assert any("Assess the heart size." in part.get("text", "") for part in content)
    assert "independently" in SYSTEM_PROMPT.lower()
    assert "spatial context only" not in SYSTEM_PROMPT


@pytest.mark.parametrize("text", ["not JSON", json.dumps({"findings": []})])
def test_malformed_single_response_fails_without_retry_or_draft_publication(settings, tmp_path, text):
    calls = []

    def generate(messages, on_token, task):
        calls.append(task)
        on_token("PRIVATE INVALID DRAFT")
        return text

    agent = _agent(settings, generate)
    events = []
    with pytest.raises(AgentError, match="single permitted generation"):
        agent.analyze(Study("single", "unused.png"), tmp_path, lambda *event: events.append(event))
    assert calls == ["report"]
    assert events == []


def test_metadata_warnings_remain_visible_without_retry_or_clinical_veto(settings, tmp_path):
    data = _report()
    data["findings"][0]["evidence_ids"] = ["MISSING-7"]
    calls = []

    def generate(messages, on_token, task):
        calls.append(task)
        return json.dumps(data)

    result = _agent(settings, generate).analyze(Study("single", "unused.png"), tmp_path)
    assert calls == ["report"]
    assert result["verification"]["ok"] is False
    assert result["verification"]["nonblocking"] is True
    assert result["verification"]["warnings"][0]["claim"] == "MISSING-7"
    assert result["report"]["impression"] == "Cardiomegaly."


def test_every_graph_node_has_the_request_cancellation_scope(settings, tmp_path):
    cancel = threading.Event()
    seen = []

    def generate(messages, on_token, task):
        assert current_cancel_event() is cancel
        cancel.set()
        on_token("never publish")
        return json.dumps(_report())

    agent = _agent(settings, generate)

    def collect(*_):
        assert current_cancel_event() is cancel
        seen.append("collect")
        return _bundle()

    agent.pipeline = SimpleNamespace(run=collect)
    events = []
    with pytest.raises(ExecutionCancelled):
        agent.analyze(Study("single", "unused.png"), tmp_path, lambda *event: events.append(event), cancel_event=cancel)
    assert seen == ["collect"]
    assert events == []


def test_cancel_after_audit_prevents_publication(settings, tmp_path):
    cancel = threading.Event()
    agent = _agent(settings, lambda *args, **kwargs: json.dumps(_report()))
    original = agent.verifier.verify

    def audit(*args):
        assert current_cancel_event() is cancel
        result = original(*args)
        cancel.set()
        return result

    agent.verifier.verify = audit
    events = []
    with pytest.raises(ExecutionCancelled):
        agent.analyze(Study("single", "unused.png"), tmp_path, lambda *event: events.append(event), cancel_event=cancel)
    assert events == []


def test_incomplete_reference_image_packet_fails_before_reasoning(settings, tmp_path):
    bundle = _bundle()
    bundle.items[1].artifacts.pop()
    calls = []
    agent = _agent(settings, lambda *args, **kwargs: calls.append(True), bundle)
    with pytest.raises(AgentError, match="retrieval image packet is incomplete"):
        agent.analyze(Study("single", "unused.png"), tmp_path)
    assert calls == []


def test_reference_only_finding_and_fabricated_measurement_are_audit_warnings():
    bundle = _bundle(1)
    packet = assemble_packet(bundle, Path("unused"))
    data = _report()
    data["findings"][0]["evidence_ids"] = ["R1"]
    data["findings"][0]["measurements"] = [{"value": 9.0, "unit": "mm", "source": "R1"}]
    result = ReportAuditor().verify(ClinicalReport.model_validate(data), bundle, packet)
    assert result.ok is False
    assert result.clinical_validation is False
    assert any("alone" in warning.detail for warning in result.warnings)
    assert any("reference-patient" in warning.detail for warning in result.warnings)


def test_reference_free_baseline_has_query_and_no_invented_triage():
    bundle = EvidenceBundle(run_id="base", normalized_image_path="query.png", study_metadata={"region": "chest"})
    packet = assemble_packet(bundle, Path("unused"))
    assert [image.id for image in packet.images] == ["Q0"]
    assert packet.reference_count == 0
    assert ReportAuditor().verify(ClinicalReport.model_validate(_report()), bundle, packet).ok


def test_mcq_option_is_explicit_in_the_same_generation_and_schema():
    data = _report()
    data["answer_choice"] = "B"
    assert ClinicalReport.model_validate(data).answer_choice == "B"
    data["answer_choice"] = "G"
    with pytest.raises(ValueError):
        ClinicalReport.model_validate(data)


def test_legacy_strict_verifier_handles_nullable_summary_as_a_result():
    result = GroundingVerifier().verify(ClinicalReport.model_validate(_report()), _bundle())
    assert not result.ok
    assert any(violation.claim == "triage_summary=null" for violation in result.violations)


@pytest.mark.parametrize("choice", [None, "C", "b", " B "])
def test_mcq_invalid_or_omitted_choice_fails_once_and_preserves_raw_response(settings, tmp_path, choice):
    data = _report()
    if choice is not None:
        data["answer_choice"] = choice
    raw = json.dumps(data)
    calls, events = [], []

    def generate(messages, on_token, task, response_schema):
        calls.append(response_schema)
        return raw

    agent = _agent(settings, generate)
    study = Study("mcq", "unused.png", metadata={"question": "Which option?", "options": {"A": "Absent", "B": "Present"}})
    with pytest.raises(AgentError, match="single permitted generation"):
        agent.analyze(study, tmp_path, lambda *event: events.append(event))
    assert len(calls) == 1
    assert "answer_choice" in calls[0]["required"]
    assert calls[0]["properties"]["answer_choice"]["enum"] == ["A", "B"]
    assert events == []
    assert (tmp_path / "reasoner-response.txt").read_text(encoding="utf-8") == raw


def test_mcq_explicit_null_fails_once(settings, tmp_path):
    data = {**_report(), "answer_choice": None}
    calls = []
    agent = _agent(settings, lambda *args, **kwargs: calls.append(kwargs) or json.dumps(data))
    with pytest.raises(AgentError, match="single permitted generation"):
        agent.analyze(Study("mcq", "unused.png", metadata={"options": {"A": "Absent", "B": "Present"}}), tmp_path)
    assert len(calls) == 1


def test_mcq_prompt_grammar_and_validation_use_the_same_supplied_choices(settings, tmp_path):
    data = {**_report(), "answer_choice": "B"}
    for finding in data["findings"]:
        finding.pop("measurements", None)  # Unrequested optional field is not generated in the compact MCQ contract.
    seen = []

    def generate(messages, on_token, task, response_schema):
        user_text = messages[1]["content"][-1]["text"]
        advertised = json.loads(user_text.split("REQUIRED ClinicalReport JSON SCHEMA:\n", 1)[1].split("\n\n", 1)[0])
        assert advertised["required"] == response_schema["required"]
        assert advertised["properties"]["answer_choice"]["enum"] == response_schema["properties"]["answer_choice"]["enum"]
        assert "answer_choice" in response_schema["required"]
        assert response_schema["properties"]["answer_choice"]["enum"] == ["A", "B"]
        assert "PRIVATE_ANSWER_KEY_X" not in user_text
        seen.append(messages)
        return json.dumps(data)

    agent = _agent(settings, generate)
    agent.backend.last_run = {"input_tokens": 100, "finish_reason": "eos"}
    result = agent.analyze(Study("mcq", "unused.png", metadata={
        "question": "Assess the heart size. A) Absent B) Present",
        "options": {"B": "Present", "A": "Absent"}, "explanation": "PRIVATE_ANSWER_KEY_X",
    }), tmp_path)
    assert len(seen) == 1
    assert result["report"]["answer_choice"] == "B"
    assert result["raw_response"] == json.dumps(data)
    assert result["generation_metadata"] == {"input_tokens": 100, "finish_reason": "eos"}
    agent.backend.last_run["input_tokens"] = 200
    assert result["generation_metadata"]["input_tokens"] == 100


@pytest.mark.parametrize("options", [{}, [], {"G": "Unknown"}, {"a": "Lowercase"}, {"A": ""}, {"A": 1}])
def test_malformed_explicit_mcq_options_fail_before_evidence_collection(settings, tmp_path, options):
    agent = _agent(settings, lambda *args, **kwargs: pytest.fail("must not reason"))
    agent.pipeline = SimpleNamespace(run=lambda *_: pytest.fail("must validate before loading tools"))
    with pytest.raises(AgentError, match="Invalid MCQ task options"):
        agent.analyze(Study("mcq", "unused.png", metadata={"options": options}), tmp_path)


def test_ordinary_report_does_not_infer_mcq_from_clinical_history(settings, tmp_path):
    seen = []

    def generate(messages, on_token, task):
        seen.append(messages)
        return json.dumps(_report())

    result = _agent(settings, generate).analyze(Study("plain", "unused.png", history="Plan A) Review. B) Follow up."), tmp_path)
    assert len(seen) == 1
    assert result["report"]["answer_choice"] is None
    assert "answer_choice" not in result["response_schema"]["required"]
