"""Explicit CPU staging releases models only after their evidence is saved."""
from pathlib import Path
from types import SimpleNamespace
import weakref

from PIL import Image

from xcortex.config import Settings
from xcortex.core.messages import Study
from xcortex.core.pipeline import AnalysisPipeline
from xcortex.tools.base import ToolOutput
from xcortex.tools.registry import ToolRegistry


def test_cpu_staging_preserves_evidence_before_releasing_auxiliary_model(tmp_path):
    settings = Settings()
    settings.models.cpu_release_auxiliary_models = True
    source = tmp_path / "query.png"
    Image.new("L", (32, 32), 100).save(source)
    run_dir = tmp_path / "run"
    observations = []

    class Tool:
        name = "triage"
        label = "test auxiliary"

        def run(self, image, meta, history):
            return ToolOutput(summary="fixture", payload={"status": "completed"},
                              images=[("overlay_fixture.png", Image.fromarray(image))])

        def release_models(self):
            observations.append((run_dir / "overlay_fixture.png").is_file())

    registry = ToolRegistry()
    registry.register(Tool())
    result = AnalysisPipeline(registry, settings).run(Study("stage", str(source)), run_dir, lambda *args: None)
    assert observations == [True]
    assert result.items[0].payload["status"] == "completed"
    assert result.items[0].artifacts[0].name == "overlay_fixture.png"


def test_rag_release_drops_its_strong_encoder_reference_without_losing_store():
    from xcortex.tools.rag_encoder import RAGRetrievalTool

    class Encoder:
        pass

    tool = RAGRetrievalTool(Settings())
    encoder = Encoder()
    handle = weakref.ref(encoder)
    tool._encoder = encoder
    store = object()
    tool._store = store
    del encoder
    tool.release_models()
    assert handle() is None
    assert tool._store is store
