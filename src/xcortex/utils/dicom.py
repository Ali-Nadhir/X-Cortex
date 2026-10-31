from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

DICOM_SUFFIXES = {".dcm", ".dicom", ".dc3"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_SUFFIXES = DICOM_SUFFIXES | IMAGE_SUFFIXES
MAX_IMAGE_DIM = 8192

_META_FIELDS = [
    ("PatientID", "patient_id"),
    ("PatientAge", "age"),
    ("PatientSex", "sex"),
    ("BodyPartExamined", "region"),
    ("Modality", "modality"),
    ("StudyDate", "study_date"),
    ("ViewPosition", "view"),
    ("PatientPosition", "position"),
    ("StudyInstanceUID", "study_id"),
    ("SeriesInstanceUID", "series_id"),
    ("SOPInstanceUID", "image_id"),
]


def _uint8_image(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("image contains non-finite pixel values")
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def image_fingerprint(image: np.ndarray) -> str:
    """Identify identical normalized pixels, including the image dimensions."""
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _finish(image: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    meta.update(rows=int(image.shape[0]), columns=int(image.shape[1]), image_sha256=image_fingerprint(image))
    return image, meta


def load_study_image(path: str | Path) -> tuple[np.ndarray, dict]:
    p = Path(path)
    if p.suffix.lower() in DICOM_SUFFIXES:
        import pydicom
        try:
            from pydicom.pixels.processing import apply_modality_lut, apply_voi_lut
        except ImportError:  # pydicom 2.x
            from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

        ds = pydicom.dcmread(str(p))
        num_frames = getattr(ds, "NumberOfFrames", 1)
        if num_frames and int(num_frames) > 1:
            raise ValueError(f"multi-frame DICOM with {num_frames} frames is not supported; single study image expected")

        arr = ds.pixel_array
        original_shape = arr.shape
        if arr.ndim < 2:
            raise ValueError(f"unsupported DICOM pixel array shape: {arr.shape}")
        if arr.ndim > 2:
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
            else:
                raise ValueError(f"unsupported multi-frame or multi-dimensional DICOM pixel array shape: {arr.shape}")

        if arr.shape[0] > MAX_IMAGE_DIM or arr.shape[1] > MAX_IMAGE_DIM:
            raise ValueError(f"DICOM image dimensions {arr.shape} exceed maximum allowed dimension ({MAX_IMAGE_DIM}px)")

        # Honor the acquisition's display transform before converting to a common
        # uint8 image. All tools and both reasoner backends use these same pixels.
        if len(original_shape) == 2:
            arr = apply_modality_lut(arr, ds)
            arr = apply_voi_lut(arr, ds)
        arr = _uint8_image(arr)

        photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
        if photometric == "MONOCHROME1":
            arr = 255 - arr

        meta: dict[str, Any] = {
            "source_format": p.suffix.lower(),
            "normalization": "dicom_modality_voi_minmax_uint8",
            "photometric_interpretation": photometric,
        }
        for src, dst in _META_FIELDS:
            value = getattr(ds, src, None)
            if value is not None:
                meta[dst] = str(value)
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            try:
                values = [float(spacing[0]), float(spacing[1])]
                if all(np.isfinite(v) and v > 0 for v in values):
                    meta["pixel_spacing_mm"] = values
                    meta["pixel_spacing_source"] = "DICOM PixelSpacing"
            except (ValueError, TypeError):
                pass
        return _finish(arr, meta)

    with Image.open(p) as original:
        img = ImageOps.exif_transpose(original)
        if img.width > MAX_IMAGE_DIM or img.height > MAX_IMAGE_DIM:
            raise ValueError(f"image dimensions ({img.width}, {img.height}) exceed maximum allowed dimension ({MAX_IMAGE_DIM}px)")
        # Pillow's direct convert('L') clips 16-bit images at 255. Preserve their
        # contrast with the same bounded normalization used for DICOM.
        if img.mode.startswith("I") or img.mode == "F":
            image = _uint8_image(np.asarray(img))
            normalization = "minmax_uint8"
        else:
            image = np.array(img.convert("L"))
            normalization = "grayscale_uint8"
    return _finish(image, {"source_format": p.suffix.lower(), "normalization": normalization})
