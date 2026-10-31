"""ChestAgentBench development evaluation through the fixed single-pass pipeline.

Eligibility exclusions are distinct from failed eligible inference attempts.
Private answer keys are used only for scoring, never supplied to the agent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
import uuid

import psutil

from ..config import Settings
from ..core.agent import Agent
from ..core.messages import Study
from ..report.pdf import render_clinical_pdf
from .chestagentbench import AnswerKey, ChestAgentQuestion, build_question_content, load_audit, score_report_choice
from .provenance import append_record, experiment_provenance, utc_now, write_json


def _eligibility(question: ChestAgentQuestion, audit: dict[str, dict], root: Path) -> str | None:
    if len(question.images) != 1:
        return "multi_image_not_supported_in_current_scope"
    if (len(question.options) < 2
            or any(key not in "ABCDEF" or len(key) != 1 or not value.strip()
                   for key, value in question.options.items())):
        return "malformed_supplied_options"
    review = audit.get(question.images[0], {})
    if (review.get("benchmark_ready") is not True
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        return "image_not_audited_or_not_cxr"
    try:
        build_question_content(question, root, audit)
    except (ValueError, FileNotFoundError) as exc:
        return f"input_validation_failed: {exc}"
    return None


def _generation_diagnostics(agent: Agent) -> dict[str, Any]:
    value = getattr(getattr(agent, "backend", None), "last_run", None)
    return dict(value) if isinstance(value, dict) else {}


def run_benchmark(
    settings: Settings,
    questions_path: str | Path,
    answers_path: str | Path,
    audit_path: str | Path,
    figures_root: str | Path,
    results_path: str | Path,
    *,
    limit: int | None = None,
    case_id: str | None = None,
    question_id: str | None = None,
    export_pdf: bool = True,
    experiment_id: str | None = None,
    condition: str = "full_context",
) -> dict[str, Any]:
    """Run audited tasks once, recording immutable per-attempt evidence.

    Accuracy and valid-rate include every eligible attempted question, including
    inference/schema errors. Ineligible images and multi-image tasks are skipped
    before inference. Existing results are never overwritten; choose a fresh
    results path for another run or condition.
    """
    questions_file, answers_file, audit_file = map(Path, (questions_path, answers_path, audit_path))
    root, results_file = Path(figures_root), Path(results_path)
    summary_file = results_file.with_suffix(".summary.json")
    provenance_file = results_file.with_suffix(".provenance.json")
    for path in (results_file, summary_file, provenance_file):
        if path.exists():
            raise FileExistsError(f"Experiment output already exists; choose a new results path: {path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer when supplied")

    audit = load_audit(audit_file)
    questions = [ChestAgentQuestion.model_validate(json.loads(line))
                 for line in questions_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    answers = [AnswerKey.model_validate(json.loads(line))
               for line in answers_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in selected question file")
    if len({key.question_id for key in answers}) != len(answers):
        raise ValueError("Duplicate question IDs in private answer file")
    answer_keys = {key.question_id: key for key in answers}
    tasks = [q for q in questions if (not case_id or q.case_id == case_id)
             and (not question_id or q.question_id == question_id)]
    if limit is not None:
        tasks = tasks[:limit]
    for q in tasks:
        if q.question_id not in answer_keys:
            raise ValueError(f"Missing private answer key for task: {q.question_id}")

    experiment_id = experiment_id or f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:10]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", experiment_id):
        raise ValueError("experiment_id must be a safe 1-96 character identifier")
    experiment_dir = Path(settings.paths.artifacts) / "eval" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    provenance = experiment_provenance(
        settings, {"questions": questions_file, "private_answers": answers_file, "audit": audit_file},
        experiment_id=experiment_id, condition=condition, artifact_directory=experiment_dir,
    )

    attempted = completed = failed = skipped = valid = correct = 0
    state = "running"

    def summary() -> dict[str, Any]:
        return {
            "benchmark": "ChestAgentBench", "experiment_id": experiment_id, "condition": condition,
            "state": state, "tasks_selected": len(tasks), "tasks_attempted": attempted,
            "tasks_evaluated": completed, "tasks_completed": completed, "tasks_failed": failed,
            "tasks_skipped": skipped, "valid_choices": valid, "correct_choices": correct,
            "accuracy_denominator": "all_eligible_attempted_questions",
            "accuracy": round(correct / attempted, 4) if attempted else 0.0,
            "valid_rate": round(valid / attempted, 4) if attempted else 0.0,
            "completion_rate": round(completed / attempted, 4) if attempted else 0.0,
            "accuracy_completed_only": round(correct / completed, 4) if completed else None,
            "artifact_directory": str(experiment_dir.resolve()),
            "provenance_file": str(provenance_file.resolve()), "timestamp": utc_now(),
        }

    with results_file.open("x", encoding="utf-8") as log:
        write_json(provenance_file, provenance, exclusive=True)
        write_json(summary_file, summary())
        try:
            agent = Agent(settings)
            for idx, question in enumerate(tasks, 1):
                q_id = question.question_id
                key = answer_keys[q_id]
                reason = _eligibility(question, audit, root)
                if reason is None and key.answer not in question.options:
                    reason = "answer_key_not_in_supplied_options"
                base: dict[str, Any] = {
                    "experiment_id": experiment_id, "condition": condition,
                    "question_id": q_id, "case_id": question.case_id,
                }
                if reason is not None:
                    skipped += 1
                    append_record(log, {**base, "status": "skipped", "reason": reason,
                                        "image_count": len(question.images), "images": question.images})
                    write_json(summary_file, summary())
                    continue

                content = build_question_content(question, root, audit)
                image_path = next(part["path"] for part in content if part.get("type") == "image")
                task_hash = hashlib.sha256(q_id.encode()).hexdigest()[:12]
                run_id = f"eval_{experiment_id}_{idx:04d}_{task_hash}"
                run_dir = experiment_dir / run_id
                run_dir.mkdir(exist_ok=False)
                study = Study(id=run_id, image_path=image_path, history="", metadata={
                    "question": question.question, "region": "chest", "question_id": q_id,
                    "case_id": question.case_id, "options": question.options,
                })
                attempted += 1
                write_json(run_dir / "result.json", {
                    **base, "status": "running", "artifact_directory": str(run_dir.resolve()),
                })
                write_json(summary_file, summary())
                proc = psutil.Process()
                start_rss = proc.memory_info().rss / (1024 * 1024)
                start = time.perf_counter()
                print(f"[{idx}/{len(tasks)}] Evaluating {q_id} ({condition})...", flush=True)
                interrupted: BaseException | None = None
                try:
                    result = agent.analyze(study, run_dir)
                    report = result["report"]
                    score = score_report_choice(report, key, question.options)
                except BaseException as exc:
                    failed += 1
                    if not isinstance(exc, Exception):
                        interrupted = exc
                    entry = {**base, "status": "interrupted" if interrupted else "failed",
                             "error": str(exc), "error_type": type(exc).__name__,
                             "predicted_choice": None, "valid_choice": False, "correct": False,
                             "generation_metadata": _generation_diagnostics(agent)}
                else:
                    completed += 1
                    valid += int(score["valid_choice"])
                    correct += int(score["correct"])
                    entry = {**base, "status": "completed", "question": question.question,
                             "options": question.options, "ground_truth_answer": key.answer,
                             "predicted_choice": score["predicted_choice"], "valid_choice": score["valid_choice"],
                             "correct": score["correct"], "report": report,
                             **{name: result.get(name) for name in (
                                 "verification", "context_manifest", "workflow", "evidence",
                                 "raw_response", "generation_metadata",
                             )}}
                entry.update({
                    "latency_seconds": round(time.perf_counter() - start, 4),
                    "rss_start_mb": round(start_rss, 3),
                    "rss_end_mb": round(proc.memory_info().rss / (1024 * 1024), 3),
                    "memory_measurement": "RSS endpoints only; not a peak measurement",
                    "artifact_directory": str(run_dir.resolve()), "pdf_artifact": None,
                })
                # Preserve inference before optional rendering, even if killed there.
                write_json(run_dir / "result.json", entry)
                if entry["status"] == "completed" and export_pdf:
                    try:
                        pdf_path = run_dir / "clinical_report.pdf"
                        render_clinical_pdf(
                            report=entry["report"],
                            metadata={"study_id": run_id, "question": question.question, "region": "chest"},
                            evidence=entry.get("evidence"), artifacts_dir=run_dir, output_path=pdf_path,
                        )
                        entry["pdf_artifact"] = str(pdf_path.resolve())
                    except BaseException as exc:
                        entry["pdf_error"] = {"error_type": type(exc).__name__, "message": str(exc)}
                        if not isinstance(exc, Exception):
                            interrupted = exc
                write_json(run_dir / "result.json", entry)
                append_record(log, entry)
                write_json(summary_file, summary())
                if interrupted is not None:
                    raise interrupted
        except BaseException:
            state = "interrupted"
            write_json(summary_file, summary())
            raise
        else:
            state = "complete"
            write_json(summary_file, summary())
    result_summary = summary()
    print(f"Evaluation complete. Summary: {result_summary}", flush=True)
    return result_summary
