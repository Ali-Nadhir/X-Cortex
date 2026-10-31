from __future__ import annotations

from .base import BaseTool, ToolOutput

MOCK_PROBABILITY = 0.71


class MockTriage(BaseTool):
    name = "triage"
    label = "Triage"

    def run(self, image, meta: dict, history: str) -> ToolOutput:
        threshold = self.settings.models.triage_threshold
        band = "high" if MOCK_PROBABILITY >= 0.66 else ("indeterminate" if MOCK_PROBABILITY >= threshold else "low")
        payload = {
            "abnormality_probability": MOCK_PROBABILITY,
            "threshold": threshold,
            "prediction": "abnormal" if MOCK_PROBABILITY >= threshold else "normal",
            "band": band,
        }
        return ToolOutput(summary=f"p(abnormal) = {MOCK_PROBABILITY} ({band})", payload=payload)
