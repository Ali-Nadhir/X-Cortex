"""Pinned ChestAgentBench preparation and deterministic MCQ scoring.

No model is loaded here. Official answers/explanations are kept in a separate
private answer key, never in question inputs or reference-image manifests.
Preparation is not clinical evaluation or a claim of patient independence.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import stat
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

CHESTAGENTBENCH_REPO = "wanglab/chest-agent-bench"
CHESTAGENTBENCH_REVISION = "921e60440927f9893228d843c0206b755744252a"
_BASE = f"https://huggingface.co/datasets/{CHESTAGENTBENCH_REPO}/resolve/{CHESTAGENTBENCH_REVISION}"
_FILES = {
    "metadata.jsonl": (4_899_709, "git-sha1", "33c91ac07e288542a10149dd5ab7779c929e8fb6"),
    "README.md": (3_440, "git-sha1", "2e08b4911952b8285a4ad15d2c22d57a2be03e03"),
    "figures.zip": (822_949_819, "sha256", "8d5ac156c267046aadf54f20d961896c6e73d31ec59b8f635d1b208fa1444226"),
}
_OPTION = re.compile(r"(?m)^\s*([A-F])[.)]\s+")


class ChestAgentQuestion(BaseModel):
    """Permitted task data; deliberately has no answer/explanation field."""

    model_config = ConfigDict(extra="forbid")
    question_id: str
    case_id: str
    question: str
    options: dict[str, str]
    images: list[str]
    image_source_urls: list[str]
    categories: list[str]
    metadata_issues: list[str] = Field(default_factory=list)


class AnswerKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    answer: Literal["A", "B", "C", "D", "E", "F"]
    explanation: str


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected: tuple[int, str, str]) -> bool:
    size, algorithm, checksum = expected
    if not path.is_file() or path.stat().st_size != size:
        return False
    if algorithm == "git-sha1":
        digest = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == checksum
    return _sha256(path) == checksum


def fetch_chestagentbench(destination: str | Path, *, include_figures: bool = False) -> dict:
    """Download only official pinned assets and verify published blob hashes.

    The public dataset has no license declared in its card. Preserve its card
    and attribution; MedRAX's source-code Apache license is not an image license.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = ["metadata.jsonl", "README.md"] + (["figures.zip"] if include_figures else [])
    files = []
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name in names:
            path = destination / ("archive" if name == "figures.zip" else "") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not _verify_file(path, _FILES[name]):
                if path.exists():
                    raise ValueError(f"Existing {path} fails pinned integrity checks; preserve and inspect it")
                temporary = path.with_suffix(path.suffix + ".part")
                with client.stream("GET", f"{_BASE}/{name}") as response:
                    response.raise_for_status()
                    written = 0
                    with temporary.open("wb") as handle:
                        for block in response.iter_bytes(4 * 1024 * 1024):
                            written += len(block)
                            if written > _FILES[name][0]:
                                raise ValueError(f"Unexpected size downloading {name}")
                            handle.write(block)
                if not _verify_file(temporary, _FILES[name]):
                    raise ValueError(f"Pinned integrity check failed for {name}; partial file preserved")
                temporary.replace(path)
            files.append({"name": name, "path": str(path.resolve()), "bytes": path.stat().st_size,
                          "sha256": _sha256(path), "source_url": f"{_BASE}/{name}",
                          "official_integrity": {"algorithm": _FILES[name][1], "value": _FILES[name][2]}})
    provenance = {
        "repo_id": CHESTAGENTBENCH_REPO, "revision": CHESTAGENTBENCH_REVISION,
        "files": files, "dataset_license_declared": None,
        "attribution": "ChestAgentBench / MedRAX, Fallahpour et al., ICML 2025; source cases: Eurorad / ESR",
        "license_note": "Dataset card has no explicit license. Eurorad states cases submitted since 2015-07-06 use CC BY-NC-SA 4.0; per-case/older image terms require review before redistribution.",
        "license_source": "https://www.eurorad.org/author-guidelines",
        "redistribution_reviewed": False,
    }
    _json(destination / "download_provenance.json", provenance)
    return provenance


def safe_image_path(root: str | Path, relative: str) -> Path:
    if "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe image path: {relative}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in {".", ".."} for p in parts.parts):
        raise ValueError(f"Unsafe image path: {relative}")
    root = Path(root).resolve()
    path = root.joinpath(*parts.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes root: {relative}")
    return path


def load_metadata(path: str | Path) -> tuple[list[ChestAgentQuestion], dict[str, AnswerKey]]:
    questions, answers, seen = [], {}, set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row["full_question_id"])
        if question_id in seen:
            raise ValueError(f"Duplicate full_question_id at line {number}: {question_id}")
        seen.add(question_id)
        question = str(row["question"])
        matches = list(_OPTION.finditer(question))
        options = {}
        for i, match in enumerate(matches):
            letter = match[1]
            if letter in options:
                raise ValueError(f"Repeated choice {letter} in {question_id}")
            end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
            options[letter] = question[match.end():end].strip()
        issues = []
        if list(options) != list("ABCDEF"):
            issues.append("supplied_options_differ_from_declared_six_choices")
        if len(options) < 2 or any(not value for value in options.values()):
            issues.append("unparseable_options")
        answer = str(row["answer"]).strip().upper()
        if answer not in options:
            issues.append("answer_not_in_supplied_options")
        images = row["images"]
        urls = row["image_source_urls"]
        if not isinstance(images, list) or not all(isinstance(p, str) for p in images):
            raise ValueError(f"Unsupported image field in {question_id}; inspect instead of silently flattening")
        if not isinstance(urls, list) or not all(isinstance(p, str) for p in urls):
            raise ValueError(f"Unsupported source URL field in {question_id}")
        for image in images:
            safe_image_path(Path.cwd(), image)
        if not images:
            issues.append("no_query_images")
        if len(images) != len(urls):
            issues.append("image_url_count_mismatch")
        questions.append(ChestAgentQuestion(
            question_id=question_id, case_id=str(row["case_id"]), question=question,
            options=options, images=images, image_source_urls=urls,
            categories=[c.strip() for c in str(row.get("categories", "")).split(",") if c.strip()],
            metadata_issues=issues,
        ))
        answers[question_id] = AnswerKey.model_validate({"question_id": question_id, "answer": answer, "explanation": str(row.get("explanation", ""))})
    return questions, answers


def extract_figures(archive: str | Path, root: str | Path, image_paths: list[str]) -> list[str]:
    """Extract explicitly requested image members only, rejecting unsafe archives."""
    extracted = []
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        counts = Counter(info.filename for info in entries)
        for name in sorted(set(image_paths)):
            target = safe_image_path(root, name)
            if counts[name] != 1:
                raise ValueError(f"Expected one archive member for {name}; found {counts[name]}")
            info = bundle.getinfo(name)
            if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir() or info.file_size > 50_000_000:
                raise ValueError(f"Unsafe image archive member: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with bundle.open(info) as source:
                    digest = hashlib.sha256(source.read()).hexdigest()
                if _sha256(target) != digest:
                    raise ValueError(f"Existing extracted image differs: {target}")
            else:
                temporary = target.with_suffix(target.suffix + ".part")
                with bundle.open(info) as source, temporary.open("wb") as destination:
                    destination.write(source.read())
                temporary.replace(target)
            extracted.append(name)
    return extracted


def _case_groups(questions: list[ChestAgentQuestion], figures_root: Path | None) -> list[list[str]]:
    """Group known same cases/images; dataset supplies no patient identifiers."""
    parents = {q.case_id: q.case_id for q in questions}
    def find(case):
        while parents[case] != case:
            parents[case] = parents[parents[case]]
            case = parents[case]
        return case
    owners, hashes = {}, {}
    for q in questions:
        tokens = ["path:" + image for image in q.images]
        tokens += ["url:" + urlsplit(url)._replace(query="", fragment="").geturl() for url in q.image_source_urls]
        if figures_root:
            for image in q.images:
                local = safe_image_path(figures_root, image)
                if local.is_file():
                    if image not in hashes:
                        hashes[image] = _sha256(local)
                    tokens.append("sha256:" + hashes[image])
        for token in tokens:
            if token in owners:
                a, b = find(q.case_id), find(owners[token])
                parents[max(a, b)] = min(a, b)
            else:
                owners[token] = q.case_id
    groups = defaultdict(list)
    for case in sorted(parents):
        groups[find(case)].append(case)
    return sorted(groups.values(), key=lambda group: group[0])


def prepare_chestagentbench(
    metadata_path: str | Path, output_dir: str | Path, *, figures_root: str | Path | None = None,
    development_cases: int = 5, reference_cases: int = 0, seed: int = 20261027,
) -> dict:
    """Prepare case-disjoint development, reserved references and remaining holdout.

    Counts select whole leakage groups and may exceed the requested minimum.
    All remaining cases are reserved, not a frozen final evaluation cohort.
    No image is marked CXR-ready without an explicit content audit.
    """
    if development_cases < 1 or reference_cases < 0:
        raise ValueError("Select at least one development case and a nonnegative reference count")
    questions, answers = load_metadata(metadata_path)
    root = Path(figures_root) if figures_root is not None else None
    groups = _case_groups(questions, root)
    random.Random(seed).shuffle(groups)
    def take(count):
        selected = []
        while groups and len(selected) < count:
            selected.extend(groups.pop(0))
        if len(selected) < count:
            raise ValueError("Insufficient independent case groups for requested preparation")
        return set(selected)
    development, reference = take(development_cases), take(reference_cases)
    heldout = {case for group in groups for case in group}
    output = Path(output_dir)
    if (output / "preparation.json").exists():
        raise ValueError("Preparation already exists; use a new versioned output directory")
    splits = {"development": development, "heldout_reserved": heldout}
    for split, cases in splits.items():
        tasks = [q for q in questions if q.case_id in cases]
        _jsonl(output / f"{split}.questions.jsonl", [q.model_dump() for q in tasks])
        _jsonl(output / "private_answer_keys" / f"{split}.answers.jsonl", [answers[q.question_id].model_dump() for q in tasks])
    reference_images = []
    for q in questions:
        if q.case_id in reference:
            for i, image in enumerate(q.images):
                item = {"case_id": q.case_id, "image_path": image, "image_source_url": q.image_source_urls[i] if i < len(q.image_source_urls) else None,
                        "role": "reserved_reference_image_only", "clinical_text": None,
                        "note": "No benchmark question, answer or explanation is included; independent reference metadata has not been approved."}
                if item not in reference_images:
                    reference_images.append(item)
    _jsonl(output / "reserved_reference_images.jsonl", reference_images)
    status = []
    for image in sorted({image for q in questions for image in q.images}):
        local = safe_image_path(root, image) if root else None
        inventory = {"image_path": image, "local_available": bool(local and local.is_file()),
                     "modality": "unreviewed", "benchmark_ready": False,
                     "note": "Manual image-content and question-answerability audit required; no CXR scope inferred from dataset name."}
        if local and local.is_file():
            from PIL import Image
            with Image.open(local) as picture:
                inventory.update(width=picture.width, height=picture.height, image_mode=picture.mode,
                                 file_format=picture.format, sha256=_sha256(local))
                picture.verify()
        status.append(inventory)
    _jsonl(output / "image_audit.pending.jsonl", status)
    summary = {
        "benchmark": "ChestAgentBench", "repo_id": CHESTAGENTBENCH_REPO,
        "revision": CHESTAGENTBENCH_REVISION, "source_metadata_sha256": _sha256(Path(metadata_path)),
        "seed": seed, "purpose": "development preparation, not frozen final protocol or evaluation results",
        "dataset_question_count": len(questions), "dataset_case_count": len({q.case_id for q in questions}),
        "dataset_unique_image_count": len(status), "patient_ids_available": False,
        "patient_disjointness": "unknown; grouping uses case IDs, source image paths/URLs and available file hashes",
        "development_case_ids": sorted(development), "reference_case_ids": sorted(reference),
        "heldout_reserved_case_ids": sorted(heldout),
        "case_counts": {"development": len(development), "reference": len(reference), "heldout_reserved": len(heldout)},
        "question_counts": {split: sum(q.case_id in cases for q in questions) for split, cases in splits.items()},
        "metadata_issue_counts": dict(Counter(issue for q in questions for issue in q.metadata_issues)),
        "all_input_images_require_content_audit": True, "model_generations": 0,
        "heldout_questions_not_for_reference_index": True,
    }
    _json(output / "preparation.json", summary)
    return summary


def load_audit(path: str | Path) -> dict[str, dict]:
    """Load an image audit JSONL into a mapping keyed by relative image_path."""
    audit_map = {}
    audit_file = Path(path)
    if not audit_file.is_file():
        raise FileNotFoundError(f"Audit file not found: {path}")
    for line in audit_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        image_path = record.get("image_path")
        if image_path:
            audit_map[image_path] = record
    return audit_map


def build_question_content(question: ChestAgentQuestion, figures_root: str | Path, audit: dict[str, dict]) -> list[dict]:
    """Actual model inputs, fail-closed until all supplied query images are audited.

    This preserves every official input image, including multi-image cases.
    Audit approval is not clinical ground truth. No labels/explanations/categories
    or diagnostic source sections are added to the model's context.
    """
    if not question.images or any(issue in question.metadata_issues for issue in ("unparseable_options", "answer_not_in_supplied_options")):
        raise ValueError(f"Task {question.question_id} is not ready: missing images or malformed choices")
    content = []
    for i, image in enumerate(question.images, 1):
        local = safe_image_path(figures_root, image)
        review = audit.get(image, {})
        if not local.is_file() or review.get("modality") != "chest_xray" or not review.get("reviewer"):
            raise ValueError(f"Image requires explicit chest-X-ray content audit: {image}")
        if review.get("sha256") != _sha256(local):
            raise ValueError(f"Image audit hash differs or is missing: {image}")
        content += [{"type": "text", "text": f"Current query image {i}; not a retrieved reference."},
                    {"type": "image", "path": str(local.resolve())}]
    content.append({"type": "text", "text": question.question + "\n\nReturn exactly one supplied answer-choice letter. Make one answer attempt; do not request tools or an automatic retry."})
    return content


def score_choice(response: str, answer: AnswerKey, options: dict[str, str]) -> dict:
    """Deterministic scoring; malformed/ambiguous outputs remain failed attempts."""
    text = response.strip().upper()
    match = re.fullmatch(r"(?:ANSWER\s*:\s*)?([A-F])[.)]?", text)
    predicted = match[1] if match and match[1] in options else None
    return {"question_id": answer.question_id, "predicted_choice": predicted,
            "valid_choice": predicted is not None, "correct": predicted == answer.answer,
            "answer_attempts": 1, "raw_response": response}


def score_report_choice(report: dict, answer: AnswerKey, options: dict[str, str]) -> dict:
    """Score only an explicit structured answer field, never infer it from prose."""
    supplied = report.get("answer_choice")
    result = score_choice(supplied if isinstance(supplied, str) else "", answer, options)
    result["answer_source"] = "report.answer_choice"
    return result
