"""Tests for deterministic PDF clinical report generation and API endpoint."""
from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from xcortex.config import Settings
from xcortex.report.pdf import SimplePDFCanvas, render_clinical_pdf
from xcortex.server.app import SessionStore, create_app


def _sample_report():
    return {
        "study": {"region": "chest", "history": "Routine clinical evaluation."},
        "triage_summary": {
            "abnormality_probability": 0.884,
            "threshold": 0.33,
            "band": "unvalidated",
            "calibrated": False,
            "method": "AnomalyCLIP ViT-L/14@336px",
        },
        "findings": [
            {
                "structure": "cardiac silhouette",
                "observation": "Cardiomegaly is present with normal pulmonary vascularity.",
                "evidence_ids": ["Q0"],
                "measurements": [{"value": 15.2, "unit": "cm", "source": "Q0"}],
            }
        ],
        "impression": "Cardiomegaly without acute pulmonary edema.",
        "answer_choice": "A",
        "recommendations": ["Correlate with echocardiography."],
        "limitations": ["Single frontal projection."],
    }


def test_pdf_canvas_compiles_valid_pdf():
    canvas = SimplePDFCanvas(page_width=595.28, page_height=841.89)
    canvas.fill_rect(40, 750, 500, 40, 0.1, 0.2, 0.3)
    canvas.text(50, 765, "Test Header", font="/F2", size=14.0, r=1.0, g=1.0, b=1.0)
    canvas.line(40, 740, 540, 740, 0.5, 0.5, 0.5)

    img = Image.new("RGB", (64, 64), color=(120, 180, 240))
    canvas.draw_image("Im1", img, 50, 600, 100, 100)

    pdf_bytes = canvas.compile()
    assert pdf_bytes.startswith(b"%PDF-1.4\n")
    assert pdf_bytes.endswith(b"%%EOF\n")
    assert b"/Type /Catalog" in pdf_bytes
    assert b"/Type /Pages" in pdf_bytes
    assert b"/Type /Page" in pdf_bytes
    assert b"/DCTDecode" in pdf_bytes


def test_render_clinical_pdf_is_byte_reproducible(tmp_path):
    # Create test artifacts
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (128, 128), 120).save(art_dir / "normalized.png")
    Image.new("RGB", (128, 128), (255, 100, 50)).save(art_dir / "overlay_anomalyclip.png")
    for i in range(5):
        Image.new("RGB", (64, 64), (i * 40, 100, 150)).save(art_dir / f"ref_{i}.png")

    report = _sample_report()
    meta = {"study_id": "ST-001", "date": "2026-10-28", "question": "Evaluate cardiac size."}

    pdf1 = render_clinical_pdf(report, metadata=meta, artifacts_dir=art_dir)
    pdf2 = render_clinical_pdf(report, metadata=meta, artifacts_dir=art_dir)

    assert pdf1 == pdf2, "PDF generation must be byte-deterministic given identical input"
    assert pdf1.startswith(b"%PDF-1.4\n")
    assert pdf1.endswith(b"%%EOF\n")

    # Write to file
    out_file = tmp_path / "report.pdf"
    render_clinical_pdf(report, metadata=meta, artifacts_dir=art_dir, output_path=out_file)
    assert out_file.is_file()
    assert out_file.read_bytes() == pdf1


def test_session_pdf_endpoint(tmp_path):
    settings = Settings.load()
    settings.paths.artifacts = str(tmp_path / "artifacts")
    settings.paths.state = str(tmp_path / "sessions.json")
    Path(settings.paths.artifacts).mkdir(parents=True, exist_ok=True)

    session_id = "test-session-pdf"
    session_art = Path(settings.paths.artifacts) / session_id
    session_art.mkdir(parents=True, exist_ok=True)
    Image.new("L", (64, 64), 100).save(session_art / "normalized.png")

    store = SessionStore(Path(settings.paths.state))
    store.create(session_id, {
        "id": session_id,
        "status": "ready",
        "image_path": str(session_art / "normalized.png"),
        "meta": {"patient_id": "P1", "region": "chest"},
        "report": _sample_report(),
    })

    app = create_app(settings)
    app.state.sessions = store
    client = TestClient(app)

    # 1. Fetch valid PDF
    res = client.get(f"/api/sessions/{session_id}/pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF-1.4\n")

    # 2. Unknown session returns 404
    assert client.get("/api/sessions/nonexistent/pdf").status_code == 404

    # 3. Session without report returns 400
    no_report_id = "no-report-session"
    store.create(no_report_id, {"id": no_report_id, "status": "ready", "image_path": str(session_art / "normalized.png")})
    assert client.get(f"/api/sessions/{no_report_id}/pdf").status_code == 400
