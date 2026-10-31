from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import numpy as np
from PIL import Image

from .base import BaseTool, ToolOutput
from ..utils import geometry, viz
from ..core.execution import check_cancelled, inference_lock

MAX_STRUCTURES = 6
MIN_AREA_PX = 400

SAM2_CONFIG_MAP = {
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
}


class SegmentationTool(BaseTool):
    name = "segmentation"
    label = "Segmentation"

    def __init__(self, settings):
        super().__init__(settings)
        self._generator: Any = None
        self._lock = threading.RLock()
        self._tested = False

    def _ensure_loaded(self):
        if self._generator is not None:
            return
        with self._lock:
            if self._generator is not None:
                return
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            if self.settings.models.device != "auto":
                device = self.settings.models.device
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            ckpt_path = Path(self.settings.models.sam2_checkpoint)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"SAM 2 checkpoint not found at '{ckpt_path}'. Run 'xcortex download sam2' or configure models.sam2_checkpoint."
                )

            variant = self.settings.models.sam2_variant
            config_file = SAM2_CONFIG_MAP.get(variant, variant)

            model = build_sam2(
                config_file,
                str(ckpt_path),
                device=device,
                mode="eval",
            )
            self._generator = SAM2AutomaticMaskGenerator(
                model, points_per_side=self.settings.models.sam2_points_per_side,
                points_per_batch=self.settings.models.sam2_points_per_batch,
            )

    def run(self, image: np.ndarray, meta: dict, history: str) -> ToolOutput:
        import torch

        check_cancelled()
        self._ensure_loaded()
        rgb = np.stack([image] * 3, axis=-1)
        with inference_lock(self._lock), torch.inference_mode():
            annotations = self._generator.generate(rgb)
        annotations = sorted(annotations, key=lambda a: a["area"], reverse=True)
        annotations = [a for a in annotations if a["area"] >= MIN_AREA_PX][:MAX_STRUCTURES]

        spacing = meta.get("pixel_spacing_mm")
        if spacing and len(spacing) >= 2 and float(spacing[0]) > 0 and float(spacing[1]) > 0:
            row_spacing = float(spacing[0])
            col_spacing = float(spacing[1])
            calibration = f"calibrated dicom ({row_spacing:.4f} x {col_spacing:.4f} mm/px)"
        else:
            row_spacing = None
            col_spacing = None
            calibration = "uncalibrated (pixel units)"

        masks = [a["segmentation"] for a in annotations]
        structures = []
        for i, annotation in enumerate(annotations):
            area_px = float(annotation["area"])
            area_mm2 = (
                round(area_px * row_spacing * col_spacing, 2)
                if (row_spacing is not None and col_spacing is not None)
                else None
            )
            structures.append(
                {
                    "index": i,
                    "area_px": round(area_px),
                    "area_mm2": area_mm2,
                    "bbox": [int(v) for v in annotation["bbox"]],
                    "score": round(float(annotation["predicted_iou"]), 3),
                    "contour": geometry.mask_contours(annotation["segmentation"]),
                }
            )
        overlay = viz.overlay_image(image, masks)
        images: list[tuple[str, Image.Image | np.ndarray]] = [("overlay.png", overlay)]
        if masks:
            stack = np.stack(masks)
            images.append(("masks.npy", stack))
        payload = {"structures": structures, "calibration": calibration,
                   "semantics": "Automatically generated unnamed masks; not validated lesion or anatomical labels"}
        if row_spacing is not None and col_spacing is not None:
            areas = " / ".join(f"{s['area_mm2']}" for s in structures[:3])
            summary = f"{len(structures)} regions segmented ({areas} mm²)"
        else:
            areas_px = " / ".join(f"{s['area_px']} px" for s in structures[:3])
            summary = f"{len(structures)} regions segmented ({areas_px}, uncalibrated)"
        self._tested = True
        check_cancelled()
        return ToolOutput(summary=summary, payload=payload, images=images)
