"""Benchmark integrity: eligible failures count and every attempt survives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from xcortex.config import Settings
from xcortex.evaluation.chestagentbench import load_audit
from xcortex.evaluation.runner import run_benchmark
import xcortex.evaluation.runner as runner


def _report(choice="B"):
    return {"study": {"region": "chest"}, "triage_summary": None, "findings": [],
            "impression": "Fixture only", "recommendations": [], "limitations": [],
            "answer_choice": choice}


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    settings = Settings()
    settings.app.mode = "mock"
    settings.paths.artifacts = str(tmp_path / "artifacts")
    settings.paths.db = str(tmp_path / "db")
    settings.paths.weights = str(tmp_path / "weights")
    root = tmp_path / "raw"
    image = root / "figures/7575/figure_1.jpg"
    image.parent.mkdir(parents=True)
    Image.new("L", (16, 16), 128).save(image)
    audit = {"image_path": "figures/7575/figure_1.jpg", "modality": "chest_xray",
             "benchmark_ready": True, "reviewer": "fixture-auditor",
             "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    question = {"question_id": "q1", "case_id": "7575", "question": "What? A) one B) two",
                "options": {"A": "one", "B": "two"}, "images": [audit["image_path"]],
                "image_source_urls": ["https://example.com/fixture"], "categories": [], "metadata_issues": []}
    paths = {"questions_path": tmp_path / "questions.jsonl", "answers_path": tmp_path / "answers.jsonl",
             "audit_path": tmp_path / "audit.jsonl", "figures_root": root,
             "results_path": tmp_path / "results.jsonl"}
    calls = []
    outcomes = []

    class FakeAgent:
        def __init__(self, _):
            self.backend = SimpleNamespace(last_run={"generation_count": 1, "fixture": True})

        def analyze(self, study, run_dir):
            calls.append((study, run_dir))
            outcome = outcomes.pop(0) if outcomes else "B"
            if isinstance(outcome, BaseException):
                raise outcome
            return {"report": _report(outcome), "evidence": [], "verification": {"ok": True},
                    "workflow": {"engine": "langgraph", "generation_count": 1},
                    "raw_response": "fixture raw response", "generation_metadata": self.backend.last_run}

    monkeypatch.setattr(runner, "Agent", FakeAgent)

    def write(questions=None, review=None):
        tasks = questions if questions is not None else [question]
        paths["questions_path"].write_text("".join(json.dumps(q) + "\n" for q in tasks), encoding="utf-8")
        paths["answers_path"].write_text("".join(json.dumps({"question_id": q["question_id"], "answer": "B", "explanation": "SCORING ONLY"}) + "\n" for q in tasks), encoding="utf-8")
        paths["audit_path"].write_text(json.dumps(audit if review is None else review) + "\n", encoding="utf-8")

    write()
    return SimpleNamespace(settings=settings, paths=paths, question=question, audit=audit,
                           calls=calls, outcomes=outcomes, write=write)


def _run(case, **kwargs):
    return run_benchmark(case.settings, **case.paths, export_pdf=False, **kwargs)


def _records(case):
    return [json.loads(line) for line in case.paths["results_path"].read_text(encoding="utf-8").splitlines()]


def test_load_audit(setup_case):
    case = setup_case
    assert load_audit(case.paths["audit_path"])[case.audit["image_path"]]["reviewer"] == "fixture-auditor"


def test_runner_scores_success_and_skips_multiview_without_truncation(setup_case):
    case = setup_case
    multi = {**case.question, "question_id": "multi", "images": [*case.question["images"], "figures/7575/other.jpg"]}
    case.write([case.question, multi])
    summary = _run(case)
    assert (summary["tasks_selected"], summary["tasks_attempted"], summary["tasks_skipped"]) == (2, 1, 1)
    assert summary["accuracy"] == 1.0
    assert len(case.calls) == 1
    records = _records(case)
    assert records[0]["raw_response"] == "fixture raw response"
    assert records[0]["workflow"]["generation_count"] == 1
    assert records[1]["reason"] == "multi_image_not_supported_in_current_scope"
    assert "SCORING ONLY" not in str(case.calls[0][0].metadata)


def test_failed_eligible_attempt_remains_in_accuracy_denominator(setup_case):
    case = setup_case
    case.write([case.question, {**case.question, "question_id": "q2"}])
    case.outcomes[:] = ["B", RuntimeError("native failure")]
    summary = _run(case)
    assert summary["tasks_attempted"] == 2
    assert summary["tasks_completed"] == summary["tasks_failed"] == 1
    assert summary["accuracy"] == summary["valid_rate"] == summary["completion_rate"] == 0.5
    assert summary["accuracy_completed_only"] == 1.0
    assert _records(case)[1]["generation_metadata"]["fixture"] is True


@pytest.mark.parametrize("change", [{"sha256": "WRONG-HASH"}, {"reviewer": ""}, {"benchmark_ready": False}])
def test_unapproved_or_changed_image_never_reaches_model(setup_case, change):
    case = setup_case
    case.write(review={**case.audit, **change})
    summary = _run(case)
    assert summary["tasks_attempted"] == 0
    assert summary["tasks_skipped"] == 1
    assert case.calls == []


def test_malformed_supplied_choices_never_reach_model(setup_case):
    case = setup_case
    case.write([{**case.question, "metadata_issues": ["unparseable_options"]}])
    assert _run(case)["tasks_skipped"] == 1
    assert case.calls == []


def test_null_choice_is_invalid_and_incorrect_without_repair(setup_case):
    case = setup_case
    case.outcomes[:] = [None]
    summary = _run(case)
    assert summary["tasks_attempted"] == 1
    assert summary["accuracy"] == summary["valid_rate"] == 0
    assert len(case.calls) == 1


def test_results_and_artifacts_survive_interruption(setup_case):
    case = setup_case
    case.write([case.question, {**case.question, "question_id": "q2"}])
    case.outcomes[:] = ["B", KeyboardInterrupt("fixture interruption")]
    with pytest.raises(KeyboardInterrupt):
        _run(case)
    records = _records(case)
    assert [r["status"] for r in records] == ["completed", "interrupted"]
    assert all((Path(r["artifact_directory"]) / "result.json").is_file() for r in records)
    summary = json.loads(case.paths["results_path"].with_suffix(".summary.json").read_text())
    assert summary["state"] == "interrupted"
    assert summary["accuracy"] == 0.5


def test_repeated_conditions_have_distinct_artifacts_and_no_overwrites(setup_case):
    case = setup_case
    first = _run(case, condition="first")
    with pytest.raises(FileExistsError, match="already exists"):
        _run(case)
    case.paths["results_path"] = case.paths["results_path"].with_name("second.jsonl")
    second = _run(case, condition="second")
    assert first["artifact_directory"] != second["artifact_directory"]
    assert case.calls[0][1] != case.calls[1][1]
    provenance = json.loads(Path(second["provenance_file"]).read_text())
    assert provenance["condition"] == "second"
    assert provenance["effective_settings_sha256"]
    assert provenance["source_sha256"]["report/prompts.py"]


def test_pdf_failure_preserves_completed_answer(setup_case, monkeypatch):
    case = setup_case

    def fail_pdf(**kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(runner, "render_clinical_pdf", fail_pdf)
    summary = run_benchmark(case.settings, **case.paths, export_pdf=True)
    assert summary["accuracy"] == 1.0
    record = _records(case)[0]
    assert record["status"] == "completed"
    assert record["pdf_error"]["message"] == "renderer unavailable"
