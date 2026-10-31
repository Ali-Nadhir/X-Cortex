"""Health must include the tokenizer/decoder dependencies actually used."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from xcortex.server.app import create_app


@pytest.mark.parametrize("structured", [True, False])
def test_health_distinguishes_missing_report_grammar_dependency(settings, tmp_path, monkeypatch, structured):
    settings.app.mode = "models"
    settings.models.reasoner = "medgemma"
    settings.models.triage_backend = "chestxray"
    settings.models.segmentation_enabled = False
    settings.models.structured_reports = structured
    settings.rag.enabled = False
    settings.paths.weights = str(tmp_path / "absent-weights")
    seen = []

    def installed(*names):
        seen.append(names)
        return "lmformatenforcer" not in names

    monkeypatch.setattr("xcortex.server.app._dependencies_present", installed)
    with TestClient(create_app(settings)) as client:
        components = client.get("/api/health").json()["components"]
    assert components["medgemma"]["dependencies_installed"] is (not structured)
    assert components["medsiglip"]["dependencies_installed"] is True
    assert all("sentencepiece" in names for names in seen if "transformers" in names)
    assert (any("lmformatenforcer" in names for names in seen)) is structured
