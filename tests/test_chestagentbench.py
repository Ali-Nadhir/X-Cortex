"""Benchmark labels stay private; preparation preserves cases and image inputs."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from xcortex.evaluation.chestagentbench import (
    AnswerKey, build_question_content, extract_figures, load_metadata,
    prepare_chestagentbench, safe_image_path, score_choice, score_report_choice,
)


def row(case="1", identifier="q1", images=None, *, five_options=False):
    images = images if images is not None else [f"figures/{case}/figure_1.jpg"]
    letters = "ABCDE" if five_options else "ABCDEF"
    return {
        "case_id": case, "full_question_id": identifier, "question_id": "source-" + identifier,
        "question": "What does the supplied image show?\n" + "\n".join(f"{letter}) Choice {letter}" for letter in letters),
        "answer": "B", "explanation": "SECRET GROUND TRUTH EXPLANATION",
        "images": images, "image_source_urls": [f"https://example.org/{image}?itok=test" for image in images],
        "categories": "classification,reasoning", "sections": "diagnosis,differential_diagnosis",
        "type": "multiple choice (A/B/C/D/E/F)",
    }


def metadata(tmp_path, rows):
    path = tmp_path / "metadata.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in rows), encoding="utf-8")
    return path


def test_parse_options_and_private_labels_without_silent_sixth_choice(tmp_path):
    source = row(five_options=True)
    source["question"] = source["question"].replace(") Choice", ". Choice")
    tasks, key = load_metadata(metadata(tmp_path, [source]))
    assert list(tasks[0].options) == list("ABCDE")
    assert "supplied_options_differ_from_declared_six_choices" in tasks[0].metadata_issues
    dumped = tasks[0].model_dump()
    assert "answer" not in dumped and "explanation" not in dumped and "sections" not in dumped
    assert "SECRET" not in json.dumps(dumped)
    assert key["q1"].answer == "B"


def test_missing_images_and_unavailable_answer_are_explicit(tmp_path):
    source = row(images=[], five_options=True)
    source["answer"] = "F"
    tasks, _ = load_metadata(metadata(tmp_path, [source]))
    assert "no_query_images" in tasks[0].metadata_issues
    assert "answer_not_in_supplied_options" in tasks[0].metadata_issues
    with pytest.raises(ValueError, match="not ready"):
        build_question_content(tasks[0], tmp_path, {})


def test_duplicate_question_identity_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        load_metadata(metadata(tmp_path, [row(), row()]))


@pytest.mark.parametrize("unsafe", ["../image.jpg", "/absolute.jpg", "C:/image.jpg", "figures\\..\\escape.jpg"])
def test_untrusted_paths_cannot_escape_root(tmp_path, unsafe):
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        safe_image_path(tmp_path, unsafe)


def test_case_splits_keep_shared_images_together_and_labels_separate(tmp_path):
    rows = [row(str(i), f"q{i}") for i in range(12)]
    rows[1]["images"] = rows[0]["images"]
    rows[1]["image_source_urls"] = rows[0]["image_source_urls"]
    # Another question on an existing case must remain with that case.
    rows.append(row("0", "q0-second"))
    path = metadata(tmp_path, rows)
    summary = prepare_chestagentbench(path, tmp_path / "a", development_cases=3, reference_cases=2, seed=17)
    repeated = prepare_chestagentbench(path, tmp_path / "b", development_cases=3, reference_cases=2, seed=17)
    split_names = ["development_case_ids", "reference_case_ids", "heldout_reserved_case_ids"]
    splits = [set(summary[name]) for name in split_names]
    assert all(not splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3))
    assert set.union(*splits) == {str(i) for i in range(12)}
    assert next(s for s in splits if "0" in s) == next(s for s in splits if "1" in s)
    assert all(summary[name] == repeated[name] for name in split_names)
    assert summary["patient_ids_available"] is False
    assert summary["model_generations"] == 0
    for name in ["development.questions.jsonl", "heldout_reserved.questions.jsonl", "reserved_reference_images.jsonl"]:
        content = (tmp_path / "a" / name).read_text()
        assert "SECRET" not in content
        assert '"explanation"' not in content and '"answer"' not in content
    assert "SECRET" in (tmp_path / "a/private_answer_keys/development.answers.jsonl").read_text()
    with pytest.raises(ValueError, match="already exists"):
        prepare_chestagentbench(path, tmp_path / "a", development_cases=3)


def test_shared_available_file_hashes_keep_cases_together(tmp_path):
    rows = [row(str(i), f"q{i}") for i in range(8)]
    for i in (0, 1):
        path = safe_image_path(tmp_path, rows[i]["images"][0])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (32, 24), 100).save(path)
    summary = prepare_chestagentbench(metadata(tmp_path, rows), tmp_path / "prepared", figures_root=tmp_path, development_cases=2, reference_cases=2, seed=1)
    partitions = [set(summary[name]) for name in ["development_case_ids", "reference_case_ids", "heldout_reserved_case_ids"]]
    assert next(s for s in partitions if "0" in s) == next(s for s in partitions if "1" in s)
    inventory = [json.loads(line) for line in (tmp_path / "prepared/image_audit.pending.jsonl").read_text().splitlines()]
    available = [image for image in inventory if image["local_available"]]
    assert len(available) == 2 and available[0]["width"] == 32
    assert available[0]["modality"] == "unreviewed"


def test_original_images_are_all_attached_and_audit_is_required(tmp_path):
    source = row(images=["figures/1/a.png", "figures/1/b.png"])
    question = load_metadata(metadata(tmp_path, [source]))[0][0]
    audit = {}
    for image in question.images:
        local = safe_image_path(tmp_path, image)
        local.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (32, 24), 100).save(local)
        audit[image] = {"modality": "chest_xray", "reviewer": "test-auditor", "sha256": hashlib.sha256(local.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="content audit"):
        build_question_content(question, tmp_path, {})
    content = build_question_content(question, tmp_path, audit)
    assert len([item for item in content if item["type"] == "image"]) == 2
    text = json.dumps(content)
    assert "SECRET" not in text and "classification" not in text and "diagnosis" not in text
    assert "Choice F" in content[-1]["text"]
    audit[question.images[0]]["modality"] = "ct"
    with pytest.raises(ValueError, match="content audit"):
        build_question_content(question, tmp_path, audit)
    audit[question.images[0]]["modality"] = "chest_xray"
    audit[question.images[0]]["sha256"] = "wrong-hash"
    with pytest.raises(ValueError, match="hash"):
        build_question_content(question, tmp_path, audit)


def test_only_requested_zip_images_are_extracted_and_never_overwritten(tmp_path):
    archive = tmp_path / "figures.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("figures/1/a.png", b"requested")
        bundle.writestr("figures/2/b.png", b"not requested")
    destination = tmp_path / "raw"
    assert extract_figures(archive, destination, ["figures/1/a.png"]) == ["figures/1/a.png"]
    assert not (destination / "figures/2/b.png").exists()
    (destination / "figures/1/a.png").write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        extract_figures(archive, destination, ["figures/1/a.png"])
    with pytest.raises(ValueError, match="Unsafe"):
        extract_figures(archive, destination, ["../escape.png"])


@pytest.mark.parametrize("response,expected,valid", [("B", True, True), ("Answer: B.", True, True), ("C", False, True), ("B or C", False, False), ("The answer may be B because...", False, False), ("F", False, False), ("", False, False)])
def test_scoring_never_repairs_or_retries_outputs(response, expected, valid):
    key = AnswerKey(question_id="q1", answer="B", explanation="not model input")
    result = score_choice(response, key, {"A": "one", "B": "two", "C": "three"})
    assert result["correct"] is expected
    assert result["valid_choice"] is valid
    assert result["answer_attempts"] == 1


def test_report_scoring_requires_explicit_supplied_choice_not_impression():
    key = AnswerKey(question_id="q1", answer="B", explanation="private")
    options = {"A": "one", "B": "two"}
    assert score_report_choice({"answer_choice": "B", "impression": "Independent report"}, key, options)["correct"] is True
    assert score_report_choice({"impression": "Answer: B"}, key, options)["valid_choice"] is False
    assert score_report_choice({"answer_choice": "F"}, key, options)["valid_choice"] is False
