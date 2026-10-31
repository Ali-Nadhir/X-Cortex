import json

import pytest
from pydantic import ValidationError

from xcortex.core.messages import EvidenceBundle, Study
from xcortex.report.prompts import build_user_prompt
from xcortex.report.schema import REPORT_SCHEMA, ClinicalReport, Measurement, TriageSummary


def test_schema_exports_json():
    dumped = json.dumps(REPORT_SCHEMA)
    assert "triage_summary" in dumped
    assert "findings" in dumped


def test_report_roundtrip():
    data = {
        "study": {"region": "wrist", "history": "fall"},
        "triage_summary": {"abnormality_probability": 0.71, "threshold": 0.33, "band": "high"},
        "findings": [{"structure": "radius", "observation": "abnormal", "evidence_ids": ["TRI-1"]}],
        "impression": "abnormal",
        "recommendations": ["review"],
    }
    report = ClinicalReport.model_validate(data)
    assert report.model_dump()["findings"][0]["evidence_ids"] == ["TRI-1"]


def test_extra_fields_rejected():
    data = {
        "study": {},
        "triage_summary": {"abnormality_probability": 0.1, "threshold": 0.33, "band": "low"},
        "findings": [],
        "impression": "",
        "recommendations": [],
        "hallucinated_field": True,
    }
    try:
        ClinicalReport.model_validate(data)
        raised = False
    except Exception:
        raised = True
    assert raised


def test_report_prompt_contains_actual_exported_schema():
    prompt = build_user_prompt(Study(id="s", image_path="x.png"), EvidenceBundle(run_id="r"))
    assert json.dumps(REPORT_SCHEMA, ensure_ascii=False) in prompt
    assert '"$defs"' in prompt
    assert '"required"' in prompt


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurements_rejected(value):
    with pytest.raises(ValidationError):
        Measurement(value=value, unit="mm²", source="SEG-1")


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_triage_score_range_is_explicit(value):
    with pytest.raises(ValidationError):
        TriageSummary(abnormality_probability=value, threshold=0.33, band="low")


def test_triage_score_provenance_is_optional_and_serialized():
    legacy = TriageSummary(abnormality_probability=0.5, threshold=0.33, band="indeterminate")
    assert legacy.calibrated is None
    assert legacy.score_type is None
    current = legacy.model_copy(update={"calibrated": False, "score_type": "uncalibrated_similarity_score", "method": "MedSigLIP"})
    assert current.model_dump()["calibrated"] is False
