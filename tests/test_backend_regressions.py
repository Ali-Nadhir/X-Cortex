from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from xcortex.config import Settings
from xcortex.core.agent import Agent, AgentError
from xcortex.core.backends.medgemma import MedGemmaBackend
from xcortex.core.backends.openai_compat import OpenAICompatBackend, _detect_mime
from xcortex.core.messages import EvidenceBundle, EvidenceItem, Study
from xcortex.rag.indexer import read_manifest
from xcortex.report.schema import ClinicalReport
from xcortex.report.verifier import GroundingVerifier
from xcortex.server.app import SessionStore, create_app
from xcortex.tools.segmentation import SAM2_CONFIG_MAP, SegmentationTool
from xcortex.utils.dicom import load_study_image


def test_session_store_atomic_and_thread_safe(tmp_path: Path):
    store_file = tmp_path / "sessions.json"
    store = SessionStore(store_file)

    # Test create
    store.create("s1", {"id": "s1", "title": "Case 1", "ts": 100, "status": "pending"})
    assert store.get("s1")["status"] == "pending"
    assert store_file.is_file()

    # Concurrent updates
    errors = []
    def update_worker(worker_id: int):
        try:
            for i in range(20):
                store.update("s1", last_worker=worker_id, iteration=i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=update_worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Verify file is valid JSON and not corrupt
    data = json.loads(store_file.read_text(encoding="utf-8"))
    assert "sessions" in data
    assert "s1" in data["sessions"]


def test_duplicate_analyze_rejected(tmp_path: Path, study_png: Path):
    settings = Settings()
    settings.app.mode = "mock"
    settings.paths.data = str(tmp_path / "data")
    settings.paths.artifacts = str(tmp_path / "artifacts")
    settings.paths.state = str(tmp_path / "state/sessions.json")

    app = create_app(settings)
    client = TestClient(app)

    # Upload image
    with open(study_png, "rb") as handle:
        res = client.post("/api/upload", files={"file": ("study.png", handle, "image/png")})
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # Mark session as currently running
    app.state.sessions.update(session_id, status="running")

    # Second analyze request should fail with 409 Conflict
    res_duplicate = client.post("/api/analyze", json={"session_id": session_id, "history": "retry"})
    assert res_duplicate.status_code == 409
    assert "in progress" in res_duplicate.json()["detail"].lower()


def test_followup_turns_persistence(tmp_path: Path, study_png: Path):
    settings = Settings()
    settings.app.mode = "mock"
    settings.paths.data = str(tmp_path / "data")
    settings.paths.artifacts = str(tmp_path / "artifacts")
    settings.paths.state = str(tmp_path / "state/sessions.json")

    client = TestClient(create_app(settings))

    # Upload & analyze
    with open(study_png, "rb") as handle:
        up = client.post("/api/upload", files={"file": ("study.png", handle, "image/png")}).json()
    sid = up["session_id"]
    client.post("/api/analyze", json={"session_id": sid, "history": "Wrist pain"})

    # Send first follow-up
    res1 = client.post("/api/ask", json={"session_id": sid, "question": "What structure was segmented?"})
    assert res1.status_code == 200

    # Verify turns stored in session
    detail1 = client.get(f"/api/sessions/{sid}").json()
    assert "turns" in detail1
    assert len(detail1["turns"]) == 2  # 1 user + 1 assistant
    assert detail1["turns"][0]["role"] == "user"
    assert detail1["turns"][0]["content"] == "What structure was segmented?"
    assert detail1["turns"][1]["role"] == "assistant"

    # Send second follow-up
    res2 = client.post("/api/ask", json={"session_id": sid, "question": "What is the triage score?"})
    assert res2.status_code == 200

    detail2 = client.get(f"/api/sessions/{sid}").json()
    assert len(detail2["turns"]) == 4  # 2 user + 2 assistant


def test_upload_size_limit_and_path_traversal(tmp_path: Path):
    settings = Settings()
    settings.app.mode = "mock"
    settings.paths.data = str(tmp_path / "data")
    settings.paths.artifacts = str(tmp_path / "artifacts")
    settings.paths.state = str(tmp_path / "state/sessions.json")

    client = TestClient(create_app(settings))

    # Huge file (51MB)
    huge_data = b"0" * (51 * 1024 * 1024)
    res = client.post("/api/upload", files={"file": ("big.png", io.BytesIO(huge_data), "image/png")})
    assert res.status_code == 413

    # Path traversal on artifact endpoint
    res_traversal = client.get("/a/../sensitive.txt")
    assert res_traversal.status_code == 404


def test_dicom_monochrome1_and_multiframe(tmp_path: Path):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset

    # Create dummy DICOM dataset
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "test.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)

    ds.Rows = 100
    ds.Columns = 100
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PixelSpacing = ["0.15", "0.20"]

    # Image where pixels are gradient from 0 to 1000
    arr = np.linspace(0, 1000, 10000, dtype=np.uint16).reshape((100, 100))
    ds.PixelData = arr.tobytes()
    dcm_path = tmp_path / "mono1.dcm"
    ds.save_as(str(dcm_path))

    # Load MONOCHROME1 image
    img, metadata = load_study_image(dcm_path)
    assert metadata["pixel_spacing_mm"] == [0.15, 0.20]
    # Because MONOCHROME1 was inverted: originally minimum (0) was at [0,0], now maximum (255)
    assert img[0, 0] == 255
    assert img[99, 99] == 0

    # Test multi-frame DICOM rejection
    ds.NumberOfFrames = 4
    multi_path = tmp_path / "multi.dcm"
    ds.save_as(str(multi_path))
    with pytest.raises(ValueError, match="multi-frame"):
        load_study_image(multi_path)


def test_segmentation_anisotropic_and_uncalibrated():
    settings = Settings()
    tool = SegmentationTool(settings)

    # Verify SAM 2 variant mapping
    assert SAM2_CONFIG_MAP["sam2.1_hiera_large"] == "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Mock generator to return fake annotations
    fake_generator = MagicMock()
    fake_generator.generate.return_value = [
        {"area": 1000.0, "bbox": [10, 10, 30, 30], "predicted_iou": 0.95, "segmentation": np.ones((50, 50), dtype=bool)}
    ]
    tool._generator = fake_generator

    # 1. Calibrated image with anisotropic spacing: 0.1 x 0.2 mm/px -> 1 px = 0.02 mm²
    img = np.zeros((50, 50), dtype=np.uint8)
    out_calibrated = tool.run(img, {"pixel_spacing_mm": [0.1, 0.2]}, "")
    struct = out_calibrated.payload["structures"][0]
    assert struct["area_px"] == 1000
    assert struct["area_mm2"] == 20.0  # 1000 * 0.1 * 0.2 = 20.0
    assert "20.0 mm²" in out_calibrated.summary
    assert "calibrated" in out_calibrated.payload["calibration"]

    # 2. Uncalibrated image (no pixel spacing) -> area_mm2 must be None, NOT fabricated mm!
    out_uncalibrated = tool.run(img, {}, "")
    struct_uncal = out_uncalibrated.payload["structures"][0]
    assert struct_uncal["area_px"] == 1000
    assert struct_uncal["area_mm2"] is None
    assert "uncalibrated" in out_uncalibrated.payload["calibration"]
    assert "uncalibrated" in out_uncalibrated.summary


def test_manifest_indexing_with_numeric_labels(tmp_path: Path):
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "MURA-v1.1/train/XR_WRIST/patient0001/study1_positive/image1.png,1\n"
        "MURA-v1.1/train/XR_WRIST/patient0002/study1_negative/image1.png,0\n"
        "MURA-v1.1/train/XR_ELBOW/patient0003/study1_positive/image1.png,1.0\n"
        "MURA-v1.1/train/XR_ELBOW/patient0004/study1_negative/image1.png,0.0\n"
        "MURA-v1.1/train/XR_FINGER/patient0005/study1_positive/image1.png,positive\n"
        "MURA-v1.1/train/XR_FINGER/patient0006/study1_negative/image1.png,normal\n"
        "MURA-v1.1/train/XR_HAND/patient0007/study1_unknown/image1.png,unusable\n",
        encoding="utf-8",
    )
    rows = read_manifest(str(manifest))
    assert len(rows) == 6  # All 6 valid rows parsed, only 'unusable' skipped
    assert rows[0][1] == "abnormal"
    assert rows[1][1] == "normal"
    assert rows[2][1] == "abnormal"
    assert rows[3][1] == "normal"
    assert rows[4][1] == "abnormal"
    assert rows[5][1] == "normal"


def test_verifier_rejects_physical_units_without_calibration():
    # Bundle without calibrated spacing
    items = [
        EvidenceItem(
            id="TRI-1",
            tool="triage",
            summary="p = 0.5",
            payload={"abnormality_probability": 0.5, "threshold": 0.33, "band": "indeterminate"},
        ),
        EvidenceItem(
            id="SEG-1",
            tool="segmentation",
            summary="1 structure (uncalibrated)",
            payload={
                "structures": [{"index": 0, "area_px": 500, "area_mm2": None, "score": 0.9}],
                "calibration": "uncalibrated (pixel units)",
            },
        ),
    ]
    bundle = EvidenceBundle(run_id="test", items=items)

    # Report attempting to claim physical millimeters mm² when uncalibrated
    data = {
        "study": {"region": "wrist"},
        "triage_summary": {"abnormality_probability": 0.5, "threshold": 0.33, "band": "indeterminate"},
        "findings": [
            {
                "structure": "radius",
                "observation": "Lesion noted",
                "evidence_ids": ["SEG-1"],
                "measurements": [{"value": 500.0, "unit": "mm²", "source": "SEG-1"}],
            }
        ],
        "impression": "Indeterminate study, score 0.5.",
        "recommendations": ["Correlate clinically"],
        "limitations": [],
    }
    report = ClinicalReport.model_validate(data)
    result = GroundingVerifier().verify(report, bundle)
    assert not result.ok
    assert any("physical measurement unit" in v.detail for v in result.violations)


def test_medgemma_streamer_and_exception_handling():
    settings = Settings()
    settings.models.structured_reports = False
    backend = MedGemmaBackend(settings)

    # Mock processor and model
    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    mock_model = MagicMock()
    backend._processor = mock_processor
    backend._model = mock_model

    # Simulate model generate raising an exception in background thread
    mock_model.generate.side_effect = RuntimeError("CUDA out of memory or device fault")

    with patch.object(backend, "_build_inputs", return_value=MagicMock()):
        with pytest.raises(RuntimeError, match="MedGemma generation failed"):
            backend.generate([{"role": "user", "content": "hello"}])


def test_openai_compat_mime_and_key_validation(tmp_path: Path):
    assert _detect_mime(Path("test.jpg")) == "image/jpeg"
    assert _detect_mime(Path("test.jpeg")) == "image/jpeg"
    assert _detect_mime(Path("test.png")) == "image/png"
    assert _detect_mime(Path("test.webp")) == "image/webp"

    settings = Settings()
    backend = OpenAICompatBackend(settings)

    # Missing API key must raise actionable error
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="Missing API key"):
            backend.generate([{"role": "user", "content": "hello"}])


def test_health_endpoint_truthful_readiness(tmp_path: Path):
    settings = Settings()
    settings.app.mode = "models"
    settings.models.reasoner = "medgemma"
    settings.paths.data = str(tmp_path / "data")
    settings.paths.artifacts = str(tmp_path / "artifacts")
    settings.paths.state = str(tmp_path / "state/sessions.json")

    client = TestClient(create_app(settings))
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["mode"] == "models"
    # Since medgemma weights are not downloaded in tmp_path, ready must be False
    assert data["ready"] is False
    assert data["medgemma_ready"] is False
