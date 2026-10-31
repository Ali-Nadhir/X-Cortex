from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from .schema import ClinicalReport

# Integers matter too. Exclude digits inside anatomical names, units and IDs.
CLAIM_PATTERN = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"(?=(?:mm|cm|px)(?:²|2)?\b|pixels?\b|percent\b|$|[^\w.]|\.(?!\d))", re.I
)
_CITATION = re.compile(r"\b[A-Z][A-Z0-9]*-\d+\b")
_UNIT = re.compile(r"\s*(mm(?:\s*(?:²|2|\^2))?|cm(?:\s*(?:²|2|\^2))?|px(?:\s*(?:²|2|\^2))?|pixels?|percent|%)(?![A-Za-z])", re.I)
_PHYSICAL_UNITS = {"mm", "mm2", "cm", "cm2"}
_QUANTITY = re.compile(r"probability|threshold|similarity|iou|confidence|score|area|diameter|length|width|height|thickness|distance", re.I)

# This lexical guard is intentionally conservative, not clinical entailment.
_DISEASES = {
    "fracture": r"\b(?:fractures?|broken bone|cortical discontinuity)\b",
    "dislocation": r"\b(?:dislocations?|subluxation)\b",
    "lesion": r"\blesions?\b",
    "tumor": r"\b(?:tumou?rs?|neoplasms?)\b",
    "malignancy": r"\b(?:malignan\w*|cancer)\b",
    "pneumonia": r"\bpneumonia\b",
    "effusion": r"\beffusions?\b",
    "pneumothorax": r"\bpneumothora(?:x|ces)\b",
    "edema": r"\b(?:edema|oedema)\b",
    "cardiomegaly": r"\b(?:cardiomegaly|enlarged heart)\b",
    "atelectasis": r"\batelectasis\b",
    "consolidation": r"\bconsolidations?\b",
    "arthritis": r"\b(?:arthritis|arthritic|degenerative changes?)\b",
    "osteomyelitis": r"\bosteomyelitis\b",
    "opacity": r"\bopacit(?:y|ies)\b",
    "mass": r"\bmass(?:es)?\b",
}
_DISEASE_PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in _DISEASES.items()}


class Violation(BaseModel):
    claim: str
    detail: str


class VerificationResult(BaseModel):
    ok: bool
    violations: list[Violation] = Field(default_factory=list)


class ReportAuditResult(VerificationResult):
    """Nonblocking metadata checks, explicitly not validation of image findings."""

    scope: Literal["deterministic_metadata"] = "deterministic_metadata"
    clinical_validation: Literal[False] = False
    nonblocking: Literal[True] = True
    warnings: list[Violation] = Field(default_factory=list)


@dataclass(frozen=True)
class _Quantity:
    value: float
    unit: str
    kind: str


def _close(claim: float, reference: float) -> bool:
    # No arbitrary rounding or multiplication by 100: prompts require copying.
    return math.isfinite(claim) and math.isfinite(reference) and math.isclose(claim, reference, rel_tol=1e-9, abs_tol=1e-9)


def _unit(value: str) -> str:
    value = value.strip().lower().replace("²", "2").replace("^", "").replace(" ", "")
    return {"pixel": "px", "pixels": "px", "percent": "%", "px2": "px"}.get(value, value)


def _quantities(payload: dict) -> list[_Quantity]:
    values: list[_Quantity] = []

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("value"), (int, float)) and not isinstance(node["value"], bool) and isinstance(node.get("unit"), str):
                values.append(_Quantity(float(node["value"]), _unit(node["unit"]), str(node.get("quantity", "measurement"))))
            for key, value in node.items():
                if key in {"contour", "bbox", "index", "reference", "case_id"}:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    suffix = re.search(r"_(mm2|cm2|mm|cm|px2|px)$", key)
                    if suffix:
                        values.append(_Quantity(float(value), _unit(suffix[1]), key[:suffix.start()]))
                    elif key.endswith("_percent"):
                        values.append(_Quantity(float(value), "%", key[:-8]))
                    elif key in {"abnormality_probability", "threshold", "score", "confidence", "similarity_normal", "similarity_abnormal"}:
                        values.append(_Quantity(float(value), "", key))
                elif isinstance(value, (dict, list, tuple)):
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(payload)
    return values


def _physical_allowed(payload: dict) -> bool:
    calibration = str(payload.get("calibration", "")).lower()
    return "uncalibrated" not in calibration and payload.get("calibrated") is not False


def _kind_matches(kind: str, hint: str | None) -> bool:
    if hint is None:
        return True
    if hint == "probability":
        return kind == "abnormality_probability"
    if hint == "similarity":
        return kind.startswith("similarity")
    if hint == "iou":
        return kind == "score"
    if hint == "score":
        return kind in {"score", "abnormality_probability", "confidence"}
    return kind == hint


def _clinical_texts(payload: dict) -> list[str]:
    """Only query-study assertions; never summaries, prompts or retrieved cases."""
    result: list[str] = []

    def add(node, uncertain=False):
        if isinstance(node, str):
            result.append(f"possible {node}" if uncertain else node)
        elif isinstance(node, list):
            for child in node:
                add(child, uncertain)
        elif isinstance(node, dict):
            label = node.get("observation") or node.get("finding") or node.get("diagnosis") or node.get("label") or node.get("name")
            if isinstance(label, str):
                status = str(node.get("status", "")).lower()
                if node.get("present") is False or node.get("negated") is True or status in {"absent", "negative"}:
                    label = f"no {label}"
                elif uncertain or status in {"uncertain", "possible", "indeterminate"}:
                    label = f"possible {label}"
                result.append(label)
    for key in ("prediction", "findings", "finding", "observations", "observation", "diagnoses", "diagnosis", "detections"):
        if key in payload:
            add(payload[key], uncertain=key == "detections" and payload.get("calibrated") is False)
    return result


def _disease_claims(text: str):
    for clause in re.split(r"[.;\n]|\bbut\b|\bhowever\b", text):
        for name, pattern in _DISEASE_PATTERNS.items():
            for match in pattern.finditer(clause):
                before = clause[:match.start()].lower()
                after = clause[match.end():].lower()
                if re.search(r"(?:not assessed|not evaluated|cannot assess|cannot evaluate)\s*(?:for\s*)?$", before) or re.match(r"\s*(?:is |was |were |are )?(?:not assessed|not evaluated)", after):
                    continue
                if re.search(r"\b(?:possible|suspected|uncertain|cannot exclude|may represent|could represent)\b[^,;]*$", before):
                    polarity = "uncertain"
                elif re.search(r"\b(?:no|without|negative for|absence of|no evidence of)\b[^,;]*$", before) or re.match(r"\s*(?:is |was |were |are )?(?:absent|not present|not seen|not identified)", after):
                    polarity = "absent"
                else:
                    polarity = "present"
                yield name, polarity, match.group()


class GroundingVerifier:
    """Historical strict tool-only policy, retained for audit reproduction.

    The current single-pass workflow uses ReportAuditor below. This class must
    not veto independent visual observations in that workflow.
    """

    def verify(self, report: ClinicalReport, bundle) -> VerificationResult:
        violations: list[Violation] = []
        from ..rag.metadata import normalize_region
        metadata = getattr(bundle, "study_metadata", {}) or {}
        expected_region = normalize_region(metadata.get("region"))
        actual_region = normalize_region(report.study.region)
        if expected_region != "unknown" and actual_region not in {expected_region, "unknown"}:
            violations.append(Violation(claim=f"study.region={report.study.region}", detail="reported anatomy conflicts with the query study metadata"))
        known_views = {"AP": "AP", "ANTEROPOSTERIOR": "AP", "PA": "PA", "POSTEROANTERIOR": "PA", "LAT": "LAT", "LATERAL": "LAT"}
        expected_view = known_views.get(str(metadata.get("view", "")).upper())
        supplied_view = str(report.study.view or "").strip().upper()
        actual_view = known_views.get(supplied_view, supplied_view)
        if expected_view and actual_view and actual_view != expected_view:
            violations.append(Violation(claim=f"study.view={report.study.view}", detail="reported projection conflicts with the query study metadata"))
        items = {item.id: item for item in bundle.items}
        triage_item = bundle.by_tool("triage")
        triage_payload = (triage_item.payload or {}) if triage_item else {}
        for key in ("abnormality_probability", "threshold", "band", "calibrated", "score_type", "method"):
            if report.triage_summary is None:
                violations.append(Violation(claim="triage_summary=null", detail="historical strict policy requires a triage summary"))
                break
            actual = getattr(report.triage_summary, key)
            expected = triage_payload.get(key)
            equal = _close(float(actual), float(expected)) if key in {"abnormality_probability", "threshold"} and isinstance(expected, (int, float)) and not isinstance(expected, bool) else actual == expected
            if not equal:
                violations.append(Violation(claim=f"{key}={actual}", detail=f"triage {key} does not match the corresponding triage evidence field"))

        for finding in report.findings:
            for evidence_id in finding.evidence_ids:
                if evidence_id not in items:
                    violations.append(Violation(claim=evidence_id, detail="unknown evidence reference"))
            cited = [items[eid] for eid in finding.evidence_ids if eid in items]
            if not cited and finding.observation.strip().lower() != "not assessed":
                violations.append(Violation(claim=finding.observation, detail="finding requires supporting evidence_ids"))
            for measurement in finding.measurements:
                source = items.get(measurement.source)
                if source is None:
                    violations.append(Violation(claim=measurement.source, detail="unknown measurement source"))
                    continue
                if measurement.source not in finding.evidence_ids:
                    violations.append(Violation(claim=measurement.source, detail="measurement source must be cited in the finding evidence_ids"))
                unit = _unit(measurement.unit)
                payload = source.payload or {}
                refs = _quantities(payload)
                if not any(ref.unit == unit and _close(measurement.value, ref.value) for ref in refs):
                    violations.append(Violation(claim=f"{measurement.value} {measurement.unit}", detail=f"measurement quantity/unit/value not present in source {measurement.source}"))
                if unit in _PHYSICAL_UNITS and (not _physical_allowed(payload) or not any(ref.unit == unit for ref in refs)):
                    violations.append(Violation(claim=f"{measurement.value} {measurement.unit}", detail="physical measurement unit used without a calibrated quantity in its source evidence"))
            self._verify_text(finding.structure, cited, items, violations)
            self._verify_text(finding.observation, cited, items, violations)

        for text in [report.impression, *report.recommendations, *report.limitations]:
            self._verify_text(text, list(items.values()), items, violations)
        return VerificationResult(ok=not violations, violations=violations)

    def _verify_text(self, text, cited, all_items, violations):
        for citation in _CITATION.findall(text):
            if citation not in all_items:
                violations.append(Violation(claim=citation, detail="unknown prose evidence reference"))
        clean = _CITATION.sub(lambda m: " " * len(m.group()), text)
        for match in CLAIM_PATTERN.finditer(clean):
            claim = float(match.group())
            unit_match = _UNIT.match(clean, match.end())
            unit = _unit(unit_match[1]) if unit_match else ""
            prefix = re.split(r"[.;,]|\bwith\b|\band\b", clean[:match.start()])[-1]
            hints = list(_QUANTITY.finditer(prefix[-50:]))
            hint = hints[-1].group().lower() if hints else None
            if unit == "%" and hint != "similarity":
                hint = None
            matched = False
            for item in cited:
                payload = item.payload or {}
                if unit in _PHYSICAL_UNITS and not _physical_allowed(payload):
                    continue
                refs = _quantities(payload)
                if any(ref.unit == unit and _close(claim, ref.value) and _kind_matches(ref.kind, hint) for ref in refs):
                    matched = True
                    break

                tail = clean[match.end():].lower()
                if not unit and (isinstance(payload.get("structures"), list) and re.match(r"\s+(?:structures?|regions?|masks?)\b", tail) and claim == len(payload["structures"]) or isinstance(payload.get("matches"), list) and re.match(r"\s+(?:matches?|references?)\b", tail) and claim == len(payload["matches"])):
                    matched = True
                    break
            if not matched:
                violations.append(Violation(claim=match.group() + (f" {unit_match[1]}" if unit_match else ""), detail="numeric quantity/unit/value not traceable to the cited evidence"))

        supports = set()
        for item in cited:
            if item.tool in {"rag", "retrieval"}:
                continue
            for statement in _clinical_texts(item.payload or {}):
                supports.update((name, polarity) for name, polarity, _ in _disease_claims(statement))
        for name, polarity, claim in _disease_claims(text):
            if (name, polarity) not in supports:
                violations.append(Violation(claim=claim, detail=f"{polarity} {name} claim is not supported by a query-study finding in the cited evidence"))
        if any((item.payload or {}).get("calibrated") is False for item in cited if item.tool == "triage"):
            for match in re.finditer(r"\b(?:probability|probabilities)\b|p\s*\(\s*abnormal\s*\)", text, re.I):
                if not re.search(r"\bnot\b[^.;]*$", text[max(0, match.start() - 35):match.start()], re.I):
                    violations.append(Violation(claim=text, detail="uncalibrated triage score must not be described as a probability"))
                    break


class ReportAuditor:
    """Check declared provenance and copied values without judging visual truth.

    Findings can cite Q0 and disagree with every auxiliary model. Audit warnings
    are published alongside the original report; they never cause regeneration.
    Clinical correctness must be assessed against independent reference labels.
    """

    def verify(self, report: ClinicalReport, bundle, packet=None) -> ReportAuditResult:
        from ..rag.metadata import normalize_region

        warnings: list[Violation] = []

        def warn(claim: str, detail: str):
            warnings.append(Violation(claim=claim, detail=detail))

        items = {item.id: item for item in bundle.items}
        image_sources = {}
        reference_ids = {item.id for item in bundle.items if item.tool in {"rag", "retrieval"}}
        if bundle.normalized_image_path:
            image_sources["Q0"] = None
        if packet is not None:
            for image in packet.images:
                image_sources[image.id] = items.get(image.source_evidence_id)
                if image.role == "reference":
                    reference_ids.add(image.id)
        known_ids = set(items) | set(image_sources)
        metadata = bundle.study_metadata or {}
        expected_region = normalize_region(metadata.get("region"))
        actual_region = normalize_region(report.study.region)
        if expected_region != "unknown" and actual_region not in {expected_region, "unknown"}:
            warn(f"study.region={report.study.region}", "reported anatomy conflicts with query metadata")
        view_aliases = {"AP": "AP", "ANTEROPOSTERIOR": "AP", "PA": "PA", "POSTEROANTERIOR": "PA", "LAT": "LAT", "LATERAL": "LAT"}
        expected_view = view_aliases.get(str(metadata.get("view", "")).upper())
        actual_view = str(report.study.view or "").strip().upper()
        actual_view = view_aliases.get(actual_view, actual_view)
        if expected_view and actual_view and actual_view != expected_view:
            warn(f"study.view={report.study.view}", "reported projection conflicts with query metadata")

        if report.triage_summary is not None:
            triage = bundle.by_tool("triage")
            payload = triage.payload if triage is not None else {}
            for key in ("abnormality_probability", "threshold", "band", "calibrated", "score_type", "method"):
                actual = getattr(report.triage_summary, key)
                expected = payload.get(key)
                if key in {"abnormality_probability", "threshold"} and isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    equal = _close(float(actual), float(expected))
                else:
                    equal = actual == expected
                if not equal:
                    warn(f"{key}={actual}", f"auxiliary summary {key} does not match the supplied triage metadata")

        for finding in report.findings:
            for source_id in finding.evidence_ids:
                if source_id not in known_ids:
                    warn(source_id, "unknown finding source ID")
            if finding.evidence_ids and set(finding.evidence_ids) <= reference_ids:
                warn(finding.observation, "reference-case information alone does not establish a query-study finding")
            for measurement in finding.measurements:
                if measurement.source not in finding.evidence_ids:
                    warn(measurement.source, "measurement source is not cited in the finding")
                source = items.get(measurement.source) or image_sources.get(measurement.source)
                if source is None:
                    warn(measurement.source, "no machine-traceable measurement quantity was supplied for this source")
                    continue
                if source.id in reference_ids:
                    warn(measurement.source, "reference-patient measurements must not become query-patient measurements")
                    continue
                payload = source.payload or {}
                unit = _unit(measurement.unit)
                quantities = _quantities(payload)
                if not any(ref.unit == unit and _close(measurement.value, ref.value) for ref in quantities):
                    warn(f"{measurement.value} {measurement.unit}", "measurement quantity/unit/value is not supplied by its declared source")
                if unit in _PHYSICAL_UNITS and not _physical_allowed(payload):
                    warn(f"{measurement.value} {measurement.unit}", "physical quantity source is explicitly uncalibrated")

        # Recognize only declared citation syntax. Do not run lexical disease
        # entailment against detector labels: the reasoner sees Q0 independently.
        prose = [report.impression, *report.recommendations, *report.limitations]
        prose.extend(finding.observation for finding in report.findings)
        for text in prose:
            for citation in _CITATION.findall(text):
                if citation not in known_ids:
                    warn(citation, "unknown prose source ID")
        return ReportAuditResult(ok=not warnings, violations=warnings, warnings=warnings)
