from __future__ import annotations

import numpy as np


def bbox_of(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def mask_contours(mask: np.ndarray, sample: int = 6) -> list[list[list[float]]]:
    from skimage import measure

    contours = []
    for contour in measure.find_contours(mask.astype(np.float32), 0.5):
        points = contour[::sample]
        if len(points) < 3:
            continue
        contours.append([[round(float(x), 1), round(float(y), 1)] for y, x in points])
    return contours
