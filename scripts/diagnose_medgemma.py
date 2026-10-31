"""One native development attempt per invocation, with immutable local evidence.

Run each condition in a separate process so auxiliary/model residency cannot
contaminate the next condition. This is a diagnostic, not a benchmark estimate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xcortex.config import Settings
from xcortex.core.agent import Agent
from xcortex.core.messages import Study
from xcortex.core.pipeline import AnalysisPipeline
from xcortex.evaluation.chestagentbench import (
    AnswerKey, ChestAgentQuestion, build_question_content, load_audit, score_report_choice,
)
from xcortex.evaluation.provenance import experiment_provenance, write_json
from xcortex.tools.registry import ToolRegistry
from xcortex.core.backends.telemetry import ResourceMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cpu-int8.yaml")
    parser.add_argument("--condition", required=True, choices=["direct_mcq", "original_report", "full_context"])
    parser.add_argument("--question-id", default="7575_7575_8353800862306")
    parser.add_argument("--preparation", default="data/benchmarks/chestagentbench/preparations/pilot-v2-seed20261027")
    parser.add_argument("--figures-root", default="data/benchmarks/chestagentbench/raw")
    parser.add_argument("--output-root", default="data/diagnostics/medgemma-20261029")
    parser.add_argument("--generation-timeout", type=float, default=600)
    args = parser.parse_args()
    preparation = Path(args.preparation)
    questions_file = preparation / "development.questions.jsonl"
    questions = [ChestAgentQuestion.model_validate(json.loads(line))
                 for line in questions_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    question = next(q for q in questions if q.question_id == args.question_id)
    if len(question.images) != 1:
        raise ValueError("This bounded diagnostic requires one audited original image")
    audit_file = preparation / "image_audit.audited.jsonl"
    content = build_question_content(question, args.figures_root, load_audit(audit_file))
    settings = Settings.load(args.config)
    settings.models.generation_timeout_seconds = args.generation_timeout
    run_id = f"{args.condition}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    destination = Path(args.output_root) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "provenance.json", experiment_provenance(
        settings, {"questions": questions_file, "audit": audit_file, "config": Path(args.config)},
        experiment_id=run_id, condition=args.condition, artifact_directory=destination,
    ))
    write_json(destination / "task.json", question.model_dump())
    print(json.dumps({"status": "started", "condition": args.condition, "directory": str(destination.resolve())}), flush=True)

    agent = Agent(settings)
    if args.condition != "full_context":
        # Explicit original-only ablation. Uses the same production normalizer
        # and graph; the production/default registry remains mandatory and intact.
        agent.pipeline = AnalysisPipeline(ToolRegistry(), settings)
    image_path = next(part["path"] for part in content if part["type"] == "image")
    study = Study(run_id, image_path, metadata={
        "region": "chest", "question": question.question, "options": question.options,
        "case_id": question.case_id, "question_id": question.question_id,
    })
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(10):
            try:
                # last_run contains plain JSON values; snapshot may race a new
                # telemetry key, so skip that heartbeat rather than inference.
                snapshot = json.loads(json.dumps(getattr(agent.backend, "last_run", {})))
                snapshot["whole_run_sampled_peak_rss_bytes"] = resources.peak_rss
                snapshot["whole_run_sampled_peak_private_bytes"] = resources.peak_private
                temp = destination / "progress.tmp"
                temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                temp.replace(destination / "progress.json")
                timings = snapshot.get("timings", {})
                print(json.dumps({"status": snapshot.get("status", "collecting"),
                                  "phase": snapshot.get("phase"),
                                  "elapsed_seconds": round(time.perf_counter() - started, 1),
                                  "load_progress": snapshot.get("load_progress"),
                                  "first_token_seconds": timings.get("first_generated_token_seconds")}), flush=True)
            except RuntimeError:
                pass

    def event(kind, payload):
        if kind == "stage":
            with (destination / "stages.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"elapsed_seconds": time.perf_counter() - started, **payload}) + "\n")
            print(json.dumps({"event": kind, **payload}), flush=True)

    started = time.perf_counter()
    resources = ResourceMonitor().start()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result = {}
    try:
        if args.condition == "direct_mcq":
            bundle = agent.pipeline.run(study, destination, event)
            schema = {"type": "object", "properties": {"answer_choice": {
                "type": "string", "enum": list(question.options),
            }}, "required": ["answer_choice"], "additionalProperties": False}
            messages = [
                {"role": "system", "content": "Interpret the supplied chest radiograph and answer the user's multiple-choice question. Return only the requested JSON with one supplied answer-choice letter."},
                {"role": "user", "content": [
                    {"type": "image", "path": bundle.normalized_image_path},
                    {"type": "text", "text": question.question + "\nReturn JSON: {\"answer_choice\": \"one supplied letter\"}."},
                ]},
            ]
            raw = agent.backend.generate(messages, task="report", response_schema=schema)
            report = json.loads(raw)
            result = {"report": report, "raw_response": raw,
                      "generation_metadata": getattr(agent.backend, "last_run", {}),
                      "workflow": {"engine": "direct_model_control", "generation_count": 1}}
        else:
            result = agent.analyze(study, destination, on_event=event)
        # The private answer is opened only AFTER generation. It is never part
        # of Study, a prompt, index, or the backend's decoding schema.
        answers_file = preparation / "private_answer_keys/development.answers.jsonl"
        answer = next(AnswerKey.model_validate(json.loads(line))
                      for line in answers_file.read_text(encoding="utf-8").splitlines()
                      if json.loads(line)["question_id"] == question.question_id)
        result["score"] = score_report_choice(result["report"], answer, question.options)
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      generation_metadata=getattr(agent.backend, "last_run", {}))
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        result.update(condition=args.condition, question_id=question.question_id,
                      wall_seconds=time.perf_counter() - started, whole_run_resources=resources.stop())
        write_json(destination / "result.json", result)
        print(json.dumps({"status": result.get("status"), "condition": args.condition,
                          "score": result.get("score"), "wall_seconds": result["wall_seconds"],
                          "result": str((destination / "result.json").resolve())}), flush=True)


if __name__ == "__main__":
    main()
