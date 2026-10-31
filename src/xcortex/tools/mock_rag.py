from __future__ import annotations

import numpy as np
from PIL import Image

from .base import BaseTool, ToolOutput
from ..utils import viz

MOCK_MATCHES = [
    {"case_id": "XR_WRIST_P00001_S1", "region": "wrist", "similarity_percent": 88, "reference": "Verified label: abnormal — distal radius fracture"},
    {"case_id": "XR_WRIST_P00002_S2", "region": "wrist", "similarity_percent": 81, "reference": "Verified label: abnormal — ulnar styloid fracture"},
    {"case_id": "XR_SHOULDER_P00003_S1", "region": "shoulder", "similarity_percent": 76, "reference": "Verified label: normal"},
]


class MockRetrieval(BaseTool):
    name = "rag"
    label = "Retrieval"

    def run(self, image: np.ndarray, meta: dict, history: str) -> ToolOutput:
        h, w = image.shape[:2]
        gradients = np.linspace(0.15, 0.85, w, dtype=np.float32)
        images = []
        for i in range(3):
            frame = np.tile(gradients, (h, 1)) * (0.6 + 0.2 * i)
            images.append((f"ref_{i}.png", viz.reference_thumbnail((frame * 255).astype(np.uint8))))
        payload = {
            "matches": MOCK_MATCHES,
            "count": len(MOCK_MATCHES),
            "collection": self.settings.rag.collection,
        }
        summary = "3 reference studies (88% / 81% / 76% similarity)"
        return ToolOutput(summary=summary, payload=payload, images=images)
