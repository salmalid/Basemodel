import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT  = _DIR.parent

PNG_ROOT      = ROOT.parent / "Data_original" / "png_chexpert_plus_chunk_4"
CAPTIONS_FILE = ROOT / "dataset" / "captions_filtered.json"
OUT_DIR       = ROOT / "dataset" / "preprocessed1024" / "train"
IMAGE_SIZE    = 1024


def preprocess_png(png_path, size: int = IMAGE_SIZE) -> np.ndarray:
    """PNG -> (size, size) float32 in [0, 1], ready for cache_vae.py.

    CheXpert+ PNGs are already windowed by the dataset providers.
    """
    img = Image.open(png_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0

    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip(arr, lo, hi)
    span = hi - lo
    arr = (arr - lo) / span if span > 0 else np.zeros_like(arr)

    return _pad_and_resize(arr, size)


def _pad_and_resize(arr_01: np.ndarray, size: int) -> np.ndarray:
    """Aspect-ratio-preserving Lanczos resize, center-zero-pad to (size x size).
    Uses PIL 'F' mode (float32)"""
    h, w  = arr_01.shape
    scale = size / max(h, w)
    new_h = min(int(round(h * scale)), size)
    new_w = min(int(round(w * scale)), size)

    pil    = Image.fromarray(arr_01, mode="F").resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("F", (size, size), 0.0)
    canvas.paste(pil, ((size - new_w) // 2, (size - new_h) // 2))

    return np.array(canvas, dtype=np.float32)

def main():
    if not CAPTIONS_FILE.exists():
        raise SystemExit(
            f"Not found: {CAPTIONS_FILE}\n"
            "Run  1-preprocess_caption/write_captions.py  then  filter_captions.py  first."
        )

    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    jobs = []
    for r in records:
        rel      = Path(r["path_to_image"])                 # train/patientXXX/studyY/viewZ.jpg
        png_path = PNG_ROOT / rel.with_suffix(".png")       # actual files are .png
        out_path = OUT_DIR / rel.relative_to("train").with_suffix(".npy")
        if png_path.exists():
            jobs.append((png_path, out_path))

    print(f"Captions in file : {len(records):,}")
    print(f"Matching PNGs    : {len(jobs):,}")
    print(f"Output directory : {OUT_DIR}  ({IMAGE_SIZE}x{IMAGE_SIZE})\n")

    ok, failed, skipped = 0, 0, 0

    for png_path, out_path in tqdm(jobs, desc="Preprocessing", unit="img"):
        if out_path.exists():
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            arr = preprocess_png(png_path)
            np.save(out_path, arr)
            ok += 1
        except Exception as e:
            tqdm.write(f"FAILED {png_path.name}: {e}")
            failed += 1

    print(f"\nDone -- {ok:,} saved, {skipped:,} already cached, {failed} failed")

    sample_files = list(OUT_DIR.rglob("*.npy"))
    if sample_files:
        s = np.load(sample_files[0])
        print(f"Sample shape : {s.shape}   dtype : {s.dtype}   range : [{s.min():.3f}, {s.max():.3f}]")


if __name__ == "__main__":
    main()