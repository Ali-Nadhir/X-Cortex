from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Study:
    id: str
    image_path: str
    history: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Artifact:
    name: str
    url: str
    kind: str


@dataclass
class EvidenceItem:
    id: str
    tool: str
    summary: str
    payload: dict
    artifacts: list[Artifact] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    run_id: str
    items: list[EvidenceItem] = field(default_factory=list)
    normalized_image_path: str | None = None
    study_metadata: dict = field(default_factory=dict)

    def ids(self) -> list[str]:
        return [item.id for item in self.items]

    def by_tool(self, tool: str) -> EvidenceItem | None:
        for item in self.items:
            if item.tool == tool:
                return item
        return None

    def to_prompt_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "study_metadata": {
                key: value for key, value in self.study_metadata.items()
                if key in {"region", "age", "sex", "view", "modality", "pixel_spacing_mm", "rows", "columns"}
            },
            "items": [
                {"id": item.id, "tool": item.tool, "summary": item.summary, "payload": item.payload}
                for item in self.items
            ],
        }

    def overlay_paths(self, run_dir: Path) -> list[Path]:
        return [
            run_dir / artifact.name
            for item in self.items
            for artifact in item.artifacts
            if artifact.kind == "overlay"
        ]

    def reference_paths(self, run_dir: Path) -> list[Path]:
        return [
            run_dir / artifact.name
            for item in self.items
            for artifact in item.artifacts
            if artifact.kind == "reference"
        ]
