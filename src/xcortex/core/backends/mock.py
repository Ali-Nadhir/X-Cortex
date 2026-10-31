from __future__ import annotations

import json
import time

from .base import ReasoningBackend

MOCK_REPORT = {
    "study": {"region": "wrist", "history": "", "view": None},
    "triage_summary": {"abnormality_probability": 0.71, "threshold": 0.33, "band": "high"},
    "findings": [
        {
            "structure": "distal radius",
            "observation": "Segmented region area 512.0 mm².",
            "evidence_ids": ["SEG-1", "TRI-1"],
            "confidence_band": "high",
            "measurements": [{"value": 512.0, "unit": "mm²", "source": "SEG-1"}],
        },
        {
            "structure": "radiocarpal joint space",
            "observation": "Segmented region area 287.5 mm².",
            "evidence_ids": ["SEG-1"],
            "confidence_band": "moderate",
            "measurements": [{"value": 287.5, "unit": "mm²", "source": "SEG-1"}],
        },
    ],
    "impression": "Abnormal study with abnormality probability 0.71; the closest verified reference matches at 88% similarity.",
    "recommendations": ["Correlate with prior imaging of the same region", "Orthopedic review of the distal radius"],
    "limitations": ["Single-view assessment"],
}

MOCK_ANSWER = (
    "Based on the session evidence: triage flags an abnormal study at probability 0.71, segmentation isolates two "
    "structures (512.0 mm² and 287.5 mm²), and the closest verified reference case matches at 88% similarity."
)


class MockBackend(ReasoningBackend):
    def generate(self, messages: list, on_token=None, task: str = "report", response_schema: dict | None = None) -> str:
        report = dict(MOCK_REPORT)
        if response_schema is not None:
            supplied = response_schema.get("properties", {}).get("answer_choice", {}).get("enum", [])
            if supplied:
                # Deterministic fixture behavior, never a medical prediction.
                report["answer_choice"] = supplied[0]
        text = json.dumps(report, ensure_ascii=False) if task == "report" else MOCK_ANSWER
        for i in range(0, len(text), 24):
            part = text[i : i + 24]
            if on_token:
                on_token(part)
            time.sleep(0.004)
        return text
