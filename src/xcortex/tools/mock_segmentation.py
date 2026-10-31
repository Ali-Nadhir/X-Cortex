from __future__ import annotations

import numpy as np

from .base import BaseTool, ToolOutput
from ..utils import viz

ASSUMED_SPACING_MM = 0.2
MAX_STRUCTURES = 6

MOCK_AREAS_MM2 = [512.0, 287.5]


class MockSegmentation(BaseTool):
    name = "segmentation"
    label = "Segmentation"

    def run(self, image: np.ndarray, meta: dict, history: str) -> ToolOutput:
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        masks = []
        specs = [(0.38, 0.42, 0.075, 0.05), (0.68, 0.62, 0.055, 0.038)]
        for cy, cx, ry, rx in specs:
            mask = ((yy - h * cy) / (h * ry)) ** 2 + ((xx - w * cx) / (w * rx)) ** 2 <= 1.0
            masks.append(mask)
        overlay = viz.overlay_image(image, masks)
        structures = []
        for i, area in enumerate(MOCK_AREAS_MM2):
            structures.append(
                {
                    "index": i,
                    "area_mm2": area,
                    "bbox": [int(w * 0.34), int(h * 0.30), int(w * 0.16), int(h * 0.16)],
                    "score": 0.94 - i * 0.06,
                }
            )
        payload = {
            "structures": structures,
            "calibration": meta.get("pixel_spacing_mm") and "dicom" or f"assumed {ASSUMED_SPACING_MM} mm/px",
        }
        summary = f"2 structures segmented ({MOCK_AREAS_MM2[0]} / {MOCK_AREAS_MM2[1]} mm²)"
        return ToolOutput(summary=summary, payload=payload, images=[("overlay.png", overlay)])
