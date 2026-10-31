from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from xcortex.core.agent import Agent
from xcortex.core.execution import ExecutionCancelled, cancellation_scope, current_cancel_event
from xcortex.core.messages import Study
from xcortex.core.pipeline import AnalysisPipeline
from xcortex.rag.encoder import MedSigLIPEncoder
from xcortex.rag.indexer import read_manifest, run_indexing
from xcortex.rag.store import VectorStore
from xcortex.tools.base import ToolOutput
from xcortex.tools.rag_encoder import RAGRetrievalTool
from xcortex.utils.dicom import image_fingerprint, load_study_image


def make_dicom(path: Path):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Rows, dataset.Columns = 32, 48
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME1"
    dataset.PixelSpacing = [0.15, 0.20]
    dataset.BodyPartExamined = "WRIST"
    dataset.PatientID = "patient-query"
    dataset.StudyInstanceUID = pydicom.uid.generate_uid()
    dataset.PixelData = np.linspace(0, 4000, 32 * 48, dtype=np.uint16).reshape(32, 48).tobytes()
    dataset.save_as(path)
    return dataset


def test_agent_sends_normalized_dicom_pixels_and_preserves_source_calibration(settings, tmp_path, monkeypatch):
    import xcortex.core.pipeline as pipeline_module

    original = tmp_path / "study.dcm"
    make_dicom(original)
    source_bytes = original.read_bytes()
    real_loader = pipeline_module.load_study_image
    loads = []

    def tracked_loader(path):
        loads.append(path)
        return real_loader(path)

    monkeypatch.setattr(pipeline_module, "load_study_image", tracked_loader)
    agent = Agent(settings)
    backend_generate = agent.backend.generate
    messages_seen = []

    def capture(messages, **kwargs):
        messages_seen.append(messages)
        return backend_generate(messages, **kwargs)

    agent.backend.generate = capture
    study = Study("dicom-test", str(original), metadata={"pixel_spacing_mm": [99, 99], "rows": 999})
    run_dir = tmp_path / "run"
    agent.analyze(study, run_dir)
    image_part = next(part for part in messages_seen[0][1]["content"] if part.get("type") == "image")
    image_path = Path(image_part["path"])
    assert loads == [str(original)]
    assert image_path == run_dir / "normalized.png"
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    expected, _ = real_loader(original)
    np.testing.assert_array_equal(np.array(Image.open(image_path)), expected)
    assert original.read_bytes() == source_bytes
    bundle = agent.pipeline.run(study, tmp_path / "second", lambda *_: None)
    assert bundle.study_metadata["pixel_spacing_mm"] == [0.15, 0.20]
    assert bundle.study_metadata["rows"] == 32


def test_16bit_images_preserve_contrast_instead_of_saturating(tmp_path):
    path = tmp_path / "depth16.png"
    Image.fromarray(np.array([[0, 32768, 65535]], dtype=np.uint16)).save(path)
    image, meta = load_study_image(path)
    np.testing.assert_array_equal(image, [[0, 127, 255]])
    assert meta["normalization"] == "minmax_uint8"
    assert meta["image_sha256"] == image_fingerprint(image)
    assert image_fingerprint(image.reshape(3, 1)) != meta["image_sha256"]


def test_dicom_voi_window_is_applied_before_monochrome_inversion(tmp_path):
    path = tmp_path / "windowed.dcm"
    dataset = make_dicom(path)
    dataset.WindowCenter, dataset.WindowWidth = 2000, 1000
    dataset.PixelData = np.tile(np.array([0, 1000, 3000, 4000], np.uint16), (32, 12)).tobytes()
    dataset.save_as(path)
    image, _ = load_study_image(path)
    np.testing.assert_array_equal(image[0, :4], [255, 255, 0, 0])


def test_raster_does_not_inherit_physical_calibration_from_caller_metadata(settings, study_png, tmp_path):
    bundle = AnalysisPipeline([], settings).run(
        Study("raster", str(study_png), metadata={"pixel_spacing_mm": [0.1, 0.2]}),
        tmp_path / "raster", lambda *_: None,
    )
    assert "pixel_spacing_mm" not in bundle.study_metadata


def test_cancelled_agent_never_starts_tools_and_restores_scope(settings, study_png, tmp_path):
    agent = Agent(settings)
    agent.pipeline.run = MagicMock()
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(ExecutionCancelled):
        agent.analyze(Study("cancel", str(study_png)), tmp_path / "cancel", cancel_event=stopped)
    agent.pipeline.run.assert_not_called()
    assert current_cancel_event() is None


def test_pipeline_checks_cancellation_after_tool(settings, study_png, tmp_path):
    stop = threading.Event()
    first = MagicMock(name="first tool")
    first.name, first.label = "triage", "Triage"

    def run(*_):
        stop.set()
        return ToolOutput("finished", {})

    first.run.side_effect = run
    second = MagicMock()
    pipeline = AnalysisPipeline([first, second], settings)
    with cancellation_scope(stop), pytest.raises(ExecutionCancelled):
        pipeline.run(Study("cancel", str(study_png)), tmp_path / "cancel", lambda *_: None)
    second.run.assert_not_called()


@pytest.mark.parametrize("status,count,metadata", [
    ("empty_index", 0, {"region": "wrist"}),
    ("anatomy_unknown", 2, {}),
])
def test_retrieval_unavailable_without_loading_model(settings, status, count, metadata):
    tool = RAGRetrievalTool(settings)
    tool._encoder, tool._store = MagicMock(), MagicMock()
    tool._store.count.return_value = count
    output = tool.run(np.zeros((8, 8), np.uint8), metadata, "")
    assert output.payload["status"] == status
    assert output.payload["matches"] == []
    tool._encoder.encode.assert_not_called()


def test_filtered_out_cases_are_not_reported_as_an_empty_database(settings):
    tool = RAGRetrievalTool(settings)
    tool._encoder, tool._store = MagicMock(), MagicMock()
    tool._store.count.return_value = 7
    tool._store.query.return_value = []
    output = tool.run(np.zeros((8, 8), np.uint8), {"region": "wrist", "patient_id": "p1"}, "")
    assert output.payload["status"] == "no_eligible_references"
    assert "empty" not in output.summary
    assert tool._store.query.call_args.kwargs["exclude_patient_id"] == "p1"


def test_real_chroma_filters_anatomy_patient_study_and_duplicate_pixels(settings):
    pytest.importorskip("chromadb")
    store = VectorStore(settings)
    ids = ["same-patient", "same-study", "duplicate", "chest", "eligible"]
    metadata = [
        {"region": "wrist", "patient_id": "p1", "study_id": "s2", "image_sha256": "other"},
        {"region": "wrist", "patient_id": "p2", "study_id": "s1", "image_sha256": "other"},
        {"region": "wrist", "patient_id": "p2", "study_id": "s2", "image_sha256": "same"},
        {"region": "chest", "patient_id": "p2", "study_id": "s2", "image_sha256": "other"},
        {"region": "wrist", "patient_id": "p2", "study_id": "s2", "image_sha256": "other"},
    ]
    store.upsert(ids, [[1, 0, 0]] * len(ids), metadata)
    matches = store.query(np.array([1, 0, 0]), 99, region="WRIST", exclude_patient_id="p1", exclude_study_id="s1", exclude_image_sha256="same")
    assert [match["case_id"] for match in matches] == ["eligible"]
    assert store.query(np.array([1, 0, 0]), 0) == []


def test_populated_index_pipeline_with_real_chroma_and_synthetic_embeddings(settings, tmp_path, monkeypatch):
    """Plumbing regression only: vectors and images here are synthetic, not clinical validation."""
    pytest.importorskip("chromadb")
    encoder = MagicMock()
    encoder.encode.return_value = np.array([1, 0, 0], dtype=np.float32)
    monkeypatch.setattr(MedSigLIPEncoder, "shared", classmethod(lambda cls, _: encoder))
    source = tmp_path / "source"
    source.mkdir()
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(source / "reference.png")
    manifest = source / "references.csv"
    manifest.write_text("image_path,label,region,patient_id,study_id,split\nreference.png,1,THORAX,p-ref,s-ref,reference\n", encoding="utf-8")
    assert read_manifest(str(manifest)) == [("reference.png", "abnormal")]
    assert run_indexing(settings, str(manifest), batch=1) == 1
    output = RAGRetrievalTool(settings).run(
        np.zeros((8, 8), np.uint8), {"region": "chest", "patient_id": "p-query", "study_id": "s-query", "image_sha256": "query"}, "",
    )
    match = output.payload["matches"][0]
    assert output.payload["status"] == "available"
    assert match["reference_kind"] == "dataset_label"
    assert match["label"] == "abnormal"
    assert match["score_type"] == "cosine_similarity_not_probability"
    assert output.images[0][0] == "ref_0.png"
    assert "verified" not in match["reference"]


def test_index_rejects_test_cases_before_embedding_any_case(settings, tmp_path, monkeypatch):
    manifest = tmp_path / "mixed.csv"
    manifest.write_text("image_path,label,split\na.png,1,train\nb.png,0,test\n", encoding="utf-8")
    shared = MagicMock()
    monkeypatch.setattr(MedSigLIPEncoder, "shared", shared)
    with pytest.raises(ValueError, match="training/reference"):
        run_indexing(settings, str(manifest))
    shared.assert_not_called()


def test_official_mura_study_directories_are_indexed(settings, tmp_path, monkeypatch):
    captured = MagicMock()
    monkeypatch.setattr("xcortex.rag.indexer.VectorStore", lambda _: captured)
    encoder = MagicMock()
    encoder.encode.return_value = np.array([1, 0])
    monkeypatch.setattr(MedSigLIPEncoder, "shared", classmethod(lambda cls, _: encoder))
    root = tmp_path / "MURA-v1.1"
    study = root / "train" / "XR_WRIST" / "patient0001" / "study1_positive"
    study.mkdir(parents=True)
    for index in (1, 2):
        Image.fromarray(np.full((8, 8), index, np.uint8)).save(study / f"image{index}.png")
    manifest = root / "train_labeled.csv"
    manifest.write_text("MURA-v1.1/train/XR_WRIST/patient0001/study1_positive/,1\n", encoding="utf-8")
    assert run_indexing(settings, str(manifest), root=str(root)) == 2
    metadata = captured.upsert.call_args.args[2]
    assert {item["region"] for item in metadata} == {"wrist"}
    assert {item["patient_id"] for item in metadata} == {"patient0001"}
    assert len({item["study_id"] for item in metadata}) == 1


def test_encoder_serializes_processor_and_pooled_feature_inference(settings):
    torch = pytest.importorskip("torch")
    encoder = MedSigLIPEncoder(settings)
    encoder._device = "cpu"
    active, peak = 0, 0
    state_lock = threading.Lock()

    class Batch(dict):
        def to(self, _):
            return self

    def processor(**kwargs):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        return Batch()

    def features(**kwargs):
        nonlocal active
        with state_lock:
            active -= 1
        return SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))

    encoder._processor = processor
    encoder._model = SimpleNamespace(get_image_features=features, get_text_features=features)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(encoder.encode, np.zeros((8, 8), np.uint8)),
            pool.submit(encoder.encode_texts, ["normal radiograph"]),
            pool.submit(encoder.encode, np.zeros((8, 8), np.uint8)),
            pool.submit(encoder.encode_texts, ["abnormal radiograph"]),
        ]
        vectors = [job.result() for job in jobs]
    assert peak == 1
    for vector in vectors:
        np.testing.assert_allclose(np.linalg.norm(vector), 1.0)


def test_shared_encoder_avoids_duplicate_model_instances(settings):
    first = MedSigLIPEncoder.shared(settings)
    assert MedSigLIPEncoder.shared(settings) is first
