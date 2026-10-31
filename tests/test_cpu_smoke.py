"""Integration checks distinguish absent real weights from mock success."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from xcortex.cli import main
from xcortex.config import Settings
from xcortex.core.messages import EvidenceBundle, EvidenceItem
from xcortex.report.schema import ClinicalReport
from xcortex.report.verifier import GroundingVerifier
from xcortex.smoke import run_smoke


def isolated_settings(settings, tmp_path):
    settings.paths.weights = str(tmp_path / "weights")
    settings.models.detector_cache = str(tmp_path / "weights" / "detector")
    settings.models.local_files_only = True
    return settings


def test_missing_real_models_are_persisted_as_blocked(settings, study_png, tmp_path):
    settings = isolated_settings(settings, tmp_path)
    output = tmp_path / "smoke" / "results.json"
    result = run_smoke(settings, study_png, ["siglip", "reasoner"], output)
    assert result["status"] == "incomplete"
    assert result["clinical_validation"] is False
    assert {entry["status"] for entry in result["components"].values()} == {"blocked"}
    assert "MedGemma" in result["components"]["reasoner"]["error"]
    assert "MedSigLIP" in result["components"]["siglip"]["error"]
    assert json.loads(output.read_text()) == result


def test_cli_smoke_returns_failure_exit_code_for_missing_real_reasoner(study_png, tmp_path, capsys):
    config = tmp_path / "cpu.yaml"
    config.write_text(yaml.safe_dump({
        "app": {"mode": "models"},
        "models": {"local_files_only": True},
        "paths": {"weights": str(tmp_path / "weights"), "data": str(tmp_path / "data")},
    }))
    output = tmp_path / "smoke.json"
    with pytest.raises(SystemExit) as stopped:
        main(["--config", str(config), "smoke", "--image", str(study_png), "--components", "reasoner", "--out", str(output)])
    assert stopped.value.code == 2
    assert json.loads(output.read_text())["components"]["reasoner"]["status"] == "blocked"
    assert "CPU smoke incomplete" in capsys.readouterr().out


def test_explicit_missing_config_cannot_silently_use_defaults(tmp_path):
    with pytest.raises(FileNotFoundError):
        Settings.load(tmp_path / "misspelled-cpu-profile.yaml")


def test_smoke_rejects_conflicting_source_anatomy_before_inference(settings, tmp_path):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset

    path = tmp_path / "wrist.dcm"
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Rows = dataset.Columns = 2
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit, dataset.PixelRepresentation, dataset.SamplesPerPixel = 15, 0, 1
    dataset.PhotometricInterpretation, dataset.BodyPartExamined = "MONOCHROME2", "WRIST"
    dataset.PixelData = np.arange(4, dtype=np.uint16).reshape(2, 2).tobytes()
    dataset.save_as(path)
    with pytest.raises(ValueError, match="conflicts"):
        run_smoke(settings, path, ["detector"], tmp_path / "conflict.json", region="chest")


def test_smoke_drains_reasoner_after_generation_failure(settings, study_png, tmp_path, monkeypatch):
    """No clinical output is faked: the simulated inference times out and must fail."""
    settings = isolated_settings(settings, tmp_path)
    drained = []

    class TimedOutBackend:
        def __init__(self, settings):
            pass

        def generate(self, messages, task="chat"):
            raise TimeoutError("native prefill exceeded timeout")

        def wait_until_idle(self):
            drained.append(True)

    monkeypatch.setattr("xcortex.smoke.inspect_model", lambda _: {"installed": True})
    monkeypatch.setattr("xcortex.core.backends.medgemma.MedGemmaBackend", TimedOutBackend)
    result = run_smoke(settings, study_png, ["reasoner"], tmp_path / "timeout.json")
    assert drained == [True]
    assert result["components"]["reasoner"]["status"] == "failed"
    assert result["status"] == "incomplete"


def test_full_report_smoke_does_not_substitute_mock_when_weights_absent(settings, study_png, tmp_path):
    settings = isolated_settings(settings, tmp_path)
    assert settings.app.mode == "mock"
    result = run_smoke(settings, study_png, ["reasoner"], tmp_path / "full.json", full_report=True)
    assert result["full_report"]["status"] == "blocked"
    assert result["status"] == "incomplete"


def test_failed_full_report_drains_native_reasoner(settings, study_png, tmp_path, monkeypatch):
    settings = isolated_settings(settings, tmp_path)
    drained = []

    class FailedAgent:
        def __init__(self, settings):
            self.backend = self

        def analyze(self, *args):
            raise TimeoutError("native generation still unwinding")

        def wait_until_idle(self):
            drained.append(True)

    monkeypatch.setattr("xcortex.core.agent.Agent", FailedAgent)
    monkeypatch.setattr("xcortex.smoke.inspect_model", lambda _: {"installed": True})
    result = run_smoke(settings, study_png, ["reasoner"], tmp_path / "failed-full.json", full_report=True)
    assert result["full_report"]["status"] == "failed"
    assert drained == [True]


def test_report_cannot_change_known_study_anatomy_or_projection():
    bundle = EvidenceBundle("test", [EvidenceItem("TRI-1", "triage", "", {
        "abnormality_probability": 0.5, "threshold": 0.33, "band": "indeterminate",
    })], study_metadata={"region": "chest", "view": "AP"})
    report = ClinicalReport.model_validate({
        "study": {"region": "wrist", "view": "lateral"},
        "triage_summary": {"abnormality_probability": 0.5, "threshold": 0.33, "band": "indeterminate"},
        "findings": [], "impression": "Evidence reviewed.", "recommendations": [],
    })
    result = GroundingVerifier().verify(report, bundle)
    assert result.ok is False
    assert any("anatomy" in item.detail for item in result.violations)
    assert any("projection" in item.detail for item in result.violations)
    report.study.region, report.study.view = "thorax", "anteroposterior"
    assert GroundingVerifier().verify(report, bundle).ok is True
    report.study.view = "oblique"
    assert GroundingVerifier().verify(report, bundle).ok is False
