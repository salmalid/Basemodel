#!/usr/bin/env python3
"""DICOM → (1024, 1024) float32 in [0, 1] """

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

TARGET = 1024


def preprocess_dcm(dcm_path, size: int = TARGET) -> np.ndarray:
    """DICOM → (size, size) float32 in [0, 1], ready for the VAE pipeline."""
    ds  = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    arr = apply_modality_lut(arr, ds).astype(np.float32)

    #  apply windowing (WindowCenter/WindowWidth) or lookup table.
    try:
        arr = apply_voi_lut(arr, ds, index=0, prefer_lut=False).astype(np.float32)
    except Exception:
        lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
        arr = np.clip(arr, lo, hi)

    # Invert MONOCHROME1: low pixel = bright (opposite of MONOCHROME2).
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr

    # Percentile clip: removes scanner-edge artefacts and extreme outliers
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip(arr, lo, hi)

    # Normalize to [0, 1] float32.
    span = hi - lo
    arr  = (arr - lo) / span if span > 0 else np.zeros_like(arr)

    # Pad to square, then resize to (size × size) 
    arr = _pad_and_resize(arr, size)

    return arr  # (size, size) float32 in [0, 1]


def _pad_and_resize(arr_01: np.ndarray, size: int) -> np.ndarray:
    """Aspect-ratio-preserving Lanczos resize, then center-zero-pad to (size × size).
    Uses PIL 'F' (float32) mode — no int8/uint8 quantization anywhere in the chain."""
    h, w  = arr_01.shape
    scale = size / max(h, w)
    new_h = min(int(round(h * scale)), size)
    new_w = min(int(round(w * scale)), size)

    pil    = Image.fromarray(arr_01, mode="F").resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("F", (size, size), 0.0)
    canvas.paste(pil, ((size - new_w) // 2, (size - new_h) // 2))

    return np.array(canvas, dtype=np.float32)