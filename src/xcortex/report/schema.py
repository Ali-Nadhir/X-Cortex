from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1, description="Exact supporting evidence ID, also cited by the finding")


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: str
    observation: str
    evidence_ids: list[str] = Field(default_factory=list, description="Source IDs: Q0 for direct observations of the query image; auxiliary evidence IDs for model proposals. Reference cases are not query-study diagnoses.")
    confidence_band: str | None = None
    measurements: list[Measurement] = Field(default_factory=list)


class StudyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str | None = None
    history: str | None = None
    view: str | None = None


class TriageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    abnormality_probability: float = Field(ge=0, le=1, allow_inf_nan=False, description="Legacy score field; interpret using score_type and calibrated")
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    band: str
    calibrated: bool | None = Field(default=None, description="Copy from triage evidence; null means unspecified, not calibrated")
    score_type: str | None = None
    method: str | None = None


class ClinicalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study: StudyContext
    triage_summary: TriageSummary | None = Field(default=None, description="Copy an available auxiliary summary verbatim; null when no compatible score was supplied. This is not MedGemma's diagnostic confidence.")
    findings: list[Finding]
    impression: str
    recommendations: list[str]
    limitations: list[str] = Field(default_factory=list)
    answer_choice: Literal["A", "B", "C", "D", "E", "F"] | None = Field(default=None, description="For an explicit multiple-choice user question, select one of its supplied option IDs; otherwise null. No second generation is used to repair this field.")


REPORT_SCHEMA = ClinicalReport.model_json_schema()


class _MCQStudyContext(StudyContext):
    region: str | None = Field(default=None, max_length=48)
    history: str | None = Field(default=None, max_length=240)
    view: str | None = Field(default=None, max_length=64)


class _MCQTriageSummary(TriageSummary):
    band: str = Field(max_length=48)
    score_type: str | None = Field(default=None, max_length=80)
    method: str | None = Field(default=None, max_length=96)


class _MCQFinding(BaseModel):
    """The same finding fields, with unrequested measurements omitted on wire.

    LM Format Enforcer 0.11.3 permits a first array member even with maxItems=0.
    Excluding the optional measurements property is enforceable; ClinicalReport
    supplies its ordinary empty default after this request contract is checked.
    """
    model_config = ConfigDict(extra="forbid")

    structure: str = Field(min_length=1, max_length=64)
    observation: str = Field(min_length=1, max_length=160)
    evidence_ids: list[Annotated[str, Field(min_length=1, max_length=24)]] = Field(default_factory=list, max_length=3)
    confidence_band: str | None = Field(default=None, max_length=32)


class _MCQReport(BaseModel):
    """Bounded answer/report from the same single medical-model generation."""
    model_config = ConfigDict(extra="forbid")

    answer_choice: Literal["A", "B", "C", "D", "E", "F"]
    study: _MCQStudyContext
    findings: list[_MCQFinding] = Field(max_length=2)
    impression: str = Field(min_length=1, max_length=200)
    recommendations: list[Annotated[str, Field(min_length=1, max_length=160)]] = Field(max_length=2)
    limitations: list[Annotated[str, Field(min_length=1, max_length=160)]] = Field(default_factory=list, max_length=3)
    triage_summary: _MCQTriageSummary | None = None


MCQ_REPORT_SCHEMA = _MCQReport.model_json_schema()


def validate_mcq_options(raw: object) -> dict[str, str] | None:
    """Validate explicit task options; never infer them from clinical prose."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise ValueError("MCQ options must be a nonempty mapping of supplied A-F IDs to text")
    options: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key not in ("A", "B", "C", "D", "E", "F"):
            raise ValueError("MCQ option IDs must be supplied uppercase letters A-F")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"MCQ option {key} must have nonempty text")
        options[key] = value
    return {key: options[key] for key in sorted(options)}


def report_schema_for_options(options: dict[str, str] | None = None) -> dict[str, Any]:
    """One request's contract, shared by its prompt and decoding constraint.

    Ordinary reports retain the optional/null choice. Explicit MCQ requests must
    select an actual supplied option in the same initial generation.
    """
    validated = validate_mcq_options(options)
    schema = deepcopy(REPORT_SCHEMA if validated is None else MCQ_REPORT_SCHEMA)
    if validated is not None:
        schema["properties"]["answer_choice"] = {
            "type": "string", "enum": list(validated),
            "description": "Required answer to the user's MCQ: exactly one supplied option ID, never null.",
        }
        # LMFE's forced field-order configuration uses this required ordering.
        schema["properties"] = {
            "answer_choice": schema["properties"]["answer_choice"],
            **{key: value for key, value in schema["properties"].items() if key != "answer_choice"},
        }
        schema["required"] = ["answer_choice", *[key for key in schema.get("required", []) if key != "answer_choice"]]
    return schema


def validate_report_for_options(data: dict[str, Any], options: dict[str, str] | None = None) -> ClinicalReport:
    """Check the task answer before Pydantic can fill an omitted choice with null."""
    validated = validate_mcq_options(options)
    if validated is not None:
        supplied = data.get("answer_choice")
        if not isinstance(supplied, str) or supplied not in validated:
            raise ValueError("MCQ response must explicitly select one of its supplied option IDs")
        # Validate the same bounded typed contract used by the decoder. Ordering
        # is a decoding constraint, not a JSON semantic rule: an externally
        # supplied complete report is still valid when its keys are rearranged.
        _MCQReport.model_validate(data)
    return ClinicalReport.model_validate(data)
