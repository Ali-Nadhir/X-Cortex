import pytest

from xcortex.core.messages import EvidenceBundle, EvidenceItem
from xcortex.report.schema import ClinicalReport
from xcortex.report.verifier import GroundingVerifier


def bundle():
    items = [
        EvidenceItem(
            id="TRI-1",
            tool="triage",
            summary="p = 0.71",
            payload={"abnormality_probability": 0.71, "threshold": 0.33, "band": "high"},
        ),
        EvidenceItem(
            id="SEG-1",
            tool="segmentation",
            summary="two structures",
            payload={"structures": [{"index": 0, "area_mm2": 512.0, "score": 0.94}]},
        ),
        EvidenceItem(
            id="RAG-1",
            tool="rag",
            summary="3 matches",
            payload={"matches": [{"case_id": "a", "similarity_percent": 88}]},
        ),
    ]
    return EvidenceBundle(run_id="t", items=items)


def report(**overrides):
    data = {
        "study": {"region": "wrist"},
        "triage_summary": {"abnormality_probability": 0.71, "threshold": 0.33, "band": "high"},
        "findings": [
            {
                "structure": "distal radius",
                "observation": "Region area 512.0 mm² with 88% similar reference.",
                "evidence_ids": ["SEG-1", "TRI-1", "RAG-1"],
                "confidence_band": "high",
                "measurements": [{"value": 512.0, "unit": "mm²", "source": "SEG-1"}],
            }
        ],
        "impression": "Abnormal, probability 0.71.",
        "recommendations": ["review"],
        "limitations": [],
    }
    data.update(overrides)
    return ClinicalReport.model_validate(data)


def test_grounded_report_passes():
    result = GroundingVerifier().verify(report(), bundle())
    assert result.ok
    assert result.violations == []


def test_fabricated_measurement_fails():
    findings = [
        {
            "structure": "distal radius",
            "observation": "Region area 999.9 mm².",
            "evidence_ids": ["SEG-1"],
            "measurements": [{"value": 999.9, "unit": "mm²", "source": "SEG-1"}],
        }
    ]
    result = GroundingVerifier().verify(report(findings=findings), bundle())
    assert not result.ok
    assert any("999.9" in v.claim for v in result.violations)


def test_fabricated_probability_fails():
    triage = {"abnormality_probability": 0.42, "threshold": 0.33, "band": "high"}
    result = GroundingVerifier().verify(report(triage_summary=triage), bundle())
    assert not result.ok


def test_unknown_evidence_reference_fails():
    findings = [
        {
            "structure": "distal radius",
            "observation": "not assessed",
            "evidence_ids": ["XRAY-9"],
        }
    ]
    result = GroundingVerifier().verify(report(findings=findings), bundle())
    assert not result.ok
    assert any(v.claim == "XRAY-9" for v in result.violations)


def test_audit_reproduction_rejects_fabricated_source_and_integer_prose():
    data = report().model_dump()
    data["findings"][0].update(
        observation="A 999 mm lesion.",
        measurements=[{"value": 0.71, "unit": "mm²", "source": "NONEXISTENT"}],
    )
    result = GroundingVerifier().verify(ClinicalReport.model_validate(data), bundle())
    assert not result.ok
    assert any(v.claim == "NONEXISTENT" for v in result.violations)
    assert any("999" in v.claim for v in result.violations)


@pytest.mark.parametrize("value,unit,source", [
    (0.71, "mm²", "SEG-1"),  # triage scalar cannot become physical area
    (512, "mm", "SEG-1"),   # area cannot become a length
    (88, "mm²", "RAG-1"),   # reference similarity cannot become physical area
    (512, "mm²", "TRI-1"),  # correct number and unit, wrong source
    (5.12, "cm²", "SEG-1"), # conversion was never emitted by the source
])
def test_measurements_require_source_quantity_value_and_unit(value, unit, source):
    finding = {
        "structure": "region",
        "observation": "Segmented region.",
        "evidence_ids": [source],
        "measurements": [{"value": value, "unit": unit, "source": source}],
    }
    result = GroundingVerifier().verify(report(findings=[finding]), bundle())
    assert not result.ok
    assert any("measurement" in v.detail for v in result.violations)


def test_measurement_source_must_be_cited_by_its_finding():
    finding = report().findings[0].model_dump()
    finding.update(observation="Segmented region.", evidence_ids=["TRI-1"])
    result = GroundingVerifier().verify(report(findings=[finding]), bundle())
    assert not result.ok
    assert any("must be cited" in v.detail for v in result.violations)


@pytest.mark.parametrize("text", ["Region measures 999 mm.", "Region measures 999mm.", "Area 999.", "Score 0.999.", "Region area 512 mm."])
def test_unsupported_integer_decimal_and_unit_claims_in_prose(text):
    result = GroundingVerifier().verify(report(impression=text), bundle())
    assert not result.ok
    assert any("numeric" in v.detail for v in result.violations)


def test_finding_cannot_borrow_number_from_uncited_retrieval():
    finding = report().findings[0].model_dump()
    finding["evidence_ids"] = ["SEG-1", "TRI-1"]
    result = GroundingVerifier().verify(report(findings=[finding]), bundle())
    assert not result.ok
    assert any("88" in v.claim for v in result.violations)


def test_probability_cannot_borrow_segmentation_score():
    result = GroundingVerifier().verify(report(triage_summary={"abnormality_probability": 0.94, "threshold": 0.33, "band": "high"}), bundle())
    assert not result.ok
    assert any("triage abnormality_probability" in v.detail for v in result.violations)


def test_prose_probability_cannot_borrow_segmentation_iou():
    result = GroundingVerifier().verify(report(impression="Abnormality probability 0.94."), bundle())
    assert not result.ok
    assert any("numeric" in v.detail for v in result.violations)


def test_threshold_and_band_are_matched_to_corresponding_triage_fields():
    result = GroundingVerifier().verify(report(triage_summary={"abnormality_probability": 0.71, "threshold": 0.71, "band": "low"}), bundle())
    assert not result.ok
    assert {v.claim.split("=")[0] for v in result.violations} >= {"threshold", "band"}


def test_physical_calibration_is_source_specific_and_uncalibrated_overrides_value():
    evidence = bundle()
    evidence.items[1].payload["calibration"] = "uncalibrated (pixel units)"
    evidence.items.append(EvidenceItem(id="SEG-2", tool="segmentation", summary="calibrated", payload={"structures": [{"area_mm2": 512}], "calibration": "dicom"}))
    result = GroundingVerifier().verify(report(), evidence)
    assert not result.ok
    assert any("physical measurement unit" in v.detail for v in result.violations)


@pytest.mark.parametrize("observation", ["Fracture is present.", "No fracture.", "Pneumonia.", "Cortical discontinuity."])
def test_generic_triage_or_masks_do_not_establish_disease_or_absence(observation):
    finding = {"structure": "region", "observation": observation, "evidence_ids": ["TRI-1", "SEG-1"]}
    result = GroundingVerifier().verify(report(findings=[finding]), bundle())
    assert not result.ok
    assert any("query-study finding" in v.detail for v in result.violations)


def test_retrieved_case_diagnosis_does_not_become_query_finding():
    evidence = bundle()
    evidence.items[2].payload["findings"] = ["fracture"]
    finding = {"structure": "region", "observation": "Fracture.", "evidence_ids": ["RAG-1"]}
    result = GroundingVerifier().verify(report(findings=[finding]), evidence)
    assert not result.ok
    assert any("query-study finding" in v.detail for v in result.violations)


def test_supported_disease_passes_but_reversed_negation_fails():
    evidence = bundle()
    evidence.items[0].payload["findings"] = ["No fracture."]
    finding = {"structure": "region", "observation": "No fracture.", "evidence_ids": ["TRI-1"]}
    assert GroundingVerifier().verify(report(findings=[finding]), evidence).ok
    finding["observation"] = "Fracture is present."
    result = GroundingVerifier().verify(report(findings=[finding]), evidence)
    assert not result.ok
    assert any("present fracture" in v.detail for v in result.violations)


def test_missing_finding_citations_rejected_but_not_assessed_allowed():
    finding = {"structure": "region", "observation": "Segmented region."}
    assert not GroundingVerifier().verify(report(findings=[finding]), bundle()).ok
    finding["observation"] = "not assessed"
    assert GroundingVerifier().verify(report(findings=[finding]), bundle()).ok


def test_uncalibrated_provenance_must_be_preserved_and_score_not_called_probability():
    evidence = bundle()
    provenance = {"calibrated": False, "score_type": "uncalibrated_similarity_score", "method": "zero-shot"}
    evidence.items[0].payload.update(provenance)
    result = GroundingVerifier().verify(report(), evidence)
    assert not result.ok
    assert any("calibrated" in v.detail for v in result.violations)
    data = report().model_dump()
    data["triage_summary"].update(provenance)
    data["impression"] = "Abnormality score 0.71. This is not a clinical probability."
    assert GroundingVerifier().verify(ClinicalReport.model_validate(data), evidence).ok
    data["impression"] = "Abnormality probability 0.71."
    assert not GroundingVerifier().verify(ClinicalReport.model_validate(data), evidence).ok


def test_uncalibrated_detection_must_remain_uncertain():
    evidence = bundle()
    evidence.items[0].payload.update(calibrated=False, detections=[{"label": "Pneumonia", "score": 0.8}])
    data = report().model_dump()
    data["triage_summary"]["calibrated"] = False
    data["impression"] = "Abnormality score 0.71."
    data["findings"] = [{"structure": "lung", "observation": "Possible pneumonia.", "evidence_ids": ["TRI-1"]}]
    assert GroundingVerifier().verify(ClinicalReport.model_validate(data), evidence).ok
    data["findings"][0]["observation"] = "Pneumonia is present."
    assert not GroundingVerifier().verify(ClinicalReport.model_validate(data), evidence).ok


def test_bbox_index_and_contours_are_not_numeric_evidence_for_prose():
    evidence = bundle()
    evidence.items[1].payload["structures"][0].update(bbox=[999, 0, 20, 20], contour=[[999, 999]], index=999)
    result = GroundingVerifier().verify(report(impression="Area 999."), evidence)
    assert not result.ok


def test_citations_and_unit_suffix_digits_are_not_numeric_claims():
    finding = report().findings[0].model_dump()
    finding["observation"] = "Region area 512 mm2 [SEG-1]."
    assert GroundingVerifier().verify(report(findings=[finding]), bundle()).ok


def test_does_not_arbitrarily_round_or_percent_convert_evidence():
    evidence = bundle()
    evidence.items[0].payload["abnormality_probability"] = 0.7139
    data = report().model_dump()
    data["triage_summary"]["abnormality_probability"] = 0.71
    assert not GroundingVerifier().verify(ClinicalReport.model_validate(data), evidence).ok
