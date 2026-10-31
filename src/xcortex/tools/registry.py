from __future__ import annotations

from .base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def build_registry(settings) -> ToolRegistry:
    registry = ToolRegistry()
    if settings.app.mode == "mock":
        from .mock_rag import MockRetrieval
        from .mock_segmentation import MockSegmentation
        from .mock_triage import MockTriage

        registry.register(MockTriage(settings))
        registry.register(MockSegmentation(settings))
        registry.register(MockRetrieval(settings))
        return registry
    from .anomaly import ZeroShotAnomalyTool
    from .rag_encoder import RAGRetrievalTool
    from .segmentation import SegmentationTool
    from .triage import TriageTool

    if settings.models.triage_backend == "anomalyclip":
        from .anomalyclip import AnomalyCLIPTool
        registry.register(AnomalyCLIPTool(settings))
    elif settings.models.triage_backend == "chestxray":
        from .chestxray import ChestXRayTool
        registry.register(ChestXRayTool(settings))
    elif settings.models.triage_backend == "mobilenet":
        registry.register(TriageTool(settings))
    elif settings.models.triage_backend == "zeroshot":
        registry.register(ZeroShotAnomalyTool(settings))
    else:
        raise ValueError(f"Unsupported triage backend: {settings.models.triage_backend}")
    if settings.models.segmentation_enabled:
        registry.register(SegmentationTool(settings))
    if settings.rag.enabled:
        registry.register(RAGRetrievalTool(settings))
    return registry
