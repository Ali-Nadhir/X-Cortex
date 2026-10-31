"""AnomalyCLIP interface/geometry/math checks without loading model weights."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from xcortex.core.execution import ExecutionCancelled, cancellation_scope
from xcortex.models.anomalyclip import artifacts
from xcortex.models.anomalyclip.artifacts import WeightArtifact, verify_artifact
from xcortex.tools.anomalyclip import AnomalyCLIPTool, render_anomaly_overlay


def _tool(settings, score=0.7, value=0.2):
    tool = AnomalyCLIPTool(settings)
    tool._runtime = SimpleNamespace(infer=lambda _: (score, np.full((518, 518), value, np.float32)))
    return tool


def test_tool_preserves_original_geometry_raw_scores_and_no_diagnosis(settings):
    original = np.arange(32 * 64, dtype=np.uint8).reshape(32, 64)
    before = original.copy()
    tool = _tool(settings)
    result = tool.run(original, {"region": "chest"}, "")
    assert np.array_equal(original, before)
    assert result.payload["anomaly_score"] == 0.7
    assert result.payload["abnormality_probability"] == 0.7
    assert result.payload["calibrated"] is False
    assert result.payload["band"] == "unvalidated"
    assert "prediction" not in result.payload
    assert result.payload["detections"] == []
    assert result.payload["map"]["display_scale"]["per_image_minmax"] is False
    images = dict(result.images)
    assert images["overlay_anomalyclip.png"].size == (64, 32)
    assert images["anomaly_map.npy"].shape == original.shape
    assert images["anomaly_map_native.npy"].shape == (518, 518)
    assert np.allclose(images["anomaly_map.npy"], 0.2)
    assert tool._tested is True
    json.dumps(result.payload, allow_nan=False)


def test_rendering_uses_fixed_scale_not_per_image_extrema():
    source = np.full((10, 20), 120, dtype=np.uint8)
    low = np.full(source.shape, 0.1, np.float32)
    high = np.full(source.shape, 0.9, np.float32)
    rendered_low = np.asarray(render_anomaly_overlay(source, low))
    rendered_high = np.asarray(render_anomaly_overlay(source, high))
    assert np.isfinite(rendered_low).all()
    assert not np.array_equal(rendered_low, rendered_high)
    assert np.array_equal(rendered_low[0, 0], rendered_low[-1, -1])
    assert np.array_equal(source, np.full(source.shape, 120, dtype=np.uint8))


@pytest.mark.parametrize("response,alpha,scale", [
    (np.full((2, 3), np.nan), 0.3, 1),
    (np.full((2, 3), -0.1), 0.3, 1),
    (np.full((2, 3), 1.1), 0.3, 1),
    (np.ones((3, 2)), 0.3, 1),
    (np.ones((2, 3)), 1.1, 1),
    (np.ones((2, 3)), 0.3, 0),
])
def test_invalid_overlay_input_is_rejected(response, alpha, scale):
    with pytest.raises(ValueError):
        render_anomaly_overlay(np.zeros((2, 3), np.uint8), response, alpha=alpha, scale_max=scale)


def test_region_must_be_explicit_chest_before_loading(settings):
    tool = AnomalyCLIPTool(settings)
    with pytest.raises(ValueError, match="scoped"):
        tool.run(np.zeros((8, 8), np.uint8), {"region": "wrist"}, "")
    assert tool._runtime is None


@pytest.mark.parametrize("score,value", [(float("nan"), 0.2), (1.1, 0.2), (0.5, float("inf"))])
def test_nonfinite_or_invalid_model_result_is_not_published(settings, score, value):
    tool = _tool(settings, score, value)
    with pytest.raises(RuntimeError, match="invalid scores"):
        tool.run(np.zeros((8, 8), np.uint8), {"region": "chest"}, "")
    assert tool._tested is False


def test_cancellation_after_native_work_prevents_success(settings):
    tool = _tool(settings)
    cancel = threading.Event()

    def finish_then_cancel(_):
        cancel.set()
        return 0.7, np.full((518, 518), 0.2, np.float32)

    tool._runtime.infer = finish_then_cancel
    with cancellation_scope(cancel), pytest.raises(ExecutionCancelled):
        tool.run(np.zeros((8, 8), np.uint8), {"region": "chest"}, "")
    assert tool._tested is False


def _artifact(data=b"example"):
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    return WeightArtifact("example.bin", "https://example.invalid/example", len(data), hashlib.sha256(data).hexdigest(), blob)


def test_artifact_integrity_checks_size_sha256_and_git_blob(tmp_path):
    artifact = _artifact()
    path = tmp_path / artifact.filename
    path.write_bytes(b"example")
    assert verify_artifact(path, artifact)
    path.write_bytes(b"changed")
    assert not verify_artifact(path, artifact)


def test_download_resumes_and_promotes_only_verified_artifact(tmp_path, monkeypatch):
    data = b"complete bytes"
    artifact = _artifact(data)
    path = tmp_path / artifact.filename
    path.with_name(path.name + ".partial").write_bytes(data[:3])
    requests = []

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 3-{len(data)-1}/{len(data)}"}

    def open_request(request, timeout):
        requests.append(request)
        return Response(data[3:])

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", open_request)
    artifacts._download(path, artifact, None)
    assert requests[0].get_header("Range") == "bytes=3-"
    assert path.read_bytes() == data
    assert not path.with_name(path.name + ".partial").exists()


def test_ignored_range_response_restarts_instead_of_appending(tmp_path, monkeypatch):
    data = b"complete bytes"
    artifact = _artifact(data)
    path = tmp_path / artifact.filename
    path.with_name(path.name + ".partial").write_bytes(data[:3])

    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", lambda *args, **kwargs: Response(data))
    artifacts._download(path, artifact, None)
    assert verify_artifact(path, artifact)


def test_readiness_respects_custom_paths_and_does_not_claim_inference(tmp_path, monkeypatch):
    artifact = _artifact()
    monkeypatch.setattr(artifacts, "PROMPT", artifact)
    monkeypatch.setattr(artifacts, "BACKBONE", artifact)
    custom = tmp_path / "custom"
    custom.write_bytes(b"example")
    result = artifacts.inspect_anomalyclip(tmp_path / "other", checkpoint=custom, backbone=custom)
    assert result["installed"] is True
    assert result["inference_verified"] is False
    assert all(x["path"] == str(custom) for x in result["files"])
    assert "sizes only" in result["reason"]


def test_runtime_matches_official_normal_abnormal_equations_without_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    pytest.importorskip("torchvision")
    from xcortex.models.anomalyclip.runtime import AnomalyCLIPRuntime
    from PIL import Image

    runtime = AnomalyCLIPRuntime.__new__(AnomalyCLIPRuntime)
    runtime.device, runtime.image_size, runtime.features, runtime.sigma = torch.device("cpu"), 28, (24,), 0
    runtime.text_features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    class Model:
        def encode_image(self, tensor, levels, DPAM_layer):
            assert tensor.shape == (1, 3, 28, 28)
            assert levels == [24] and DPAM_layer == 20
            image = torch.tensor([[0.0, 1.0]])
            patches = image[:, None, :].expand(1, 5, 2).clone()
            return image, [patches]

    runtime.model = Model()
    score, response = runtime.infer(Image.new("RGB", (19, 35), (128, 128, 128)))
    expected = float(torch.tensor([0.0, 1.0 / 0.07]).softmax(0)[1])
    assert score == pytest.approx(expected)
    assert response.shape == (28, 28)
    assert np.allclose(response, expected)
