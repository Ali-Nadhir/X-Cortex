from __future__ import annotations

import numpy as np
from PIL import Image

PALETTE = [
    (79, 124, 255),
    (45, 212, 191),
    (244, 114, 182),
    (250, 204, 21),
    (167, 139, 250),
    (248, 113, 113),
]


def to_rgb(base: np.ndarray) -> np.ndarray:
    return np.stack([base] * 3, axis=-1).astype(np.float32)


def overlay_image(base: np.ndarray, masks: list[np.ndarray], alpha: float = 0.4) -> Image.Image:
    out = to_rgb(base)
    for i, mask in enumerate(masks):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        out[mask] = out[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def reference_thumbnail(image: np.ndarray, size: int = 256) -> Image.Image:
    img = Image.fromarray(image)
    img.thumbnail((size, size))
    return img
