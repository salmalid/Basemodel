"""Preprocess all DICOMs referenced in captions_filtered.json.

Reads DICOMs  : C:\\...\\PFE-project\\dataset\\train\\
Writes .npy   : C:\\...\\PFE-project\\dataset\\preprocessed1024\\train\\

The preprocessed images are shared with the parent project — identical DICOM
pipeline produces identical .npy files, so no duplication is needed. Skip this
step if the parent project has already run its own preprocess_all.py.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_DIR        = Path(os.path.dirname(os.path.abspath(__file__)))
BM_ROOT     = _DIR.parent
PARENT_ROOT = BM_ROOT.parent

sys.path.insert(0, str(_DIR))
from preprocess import preprocess_dcm

CAPTIONS_FILE = BM_ROOT / "dataset" / "captions_filtered.json"
TRAIN_DIR     = PARENT_ROOT / "dataset" / "train"
OUT_DIR       = PARENT_ROOT / "dataset" / "preprocessed1024" / "train"
IMAGE_SIZE    = 1024


def main():
    if not CAPTIONS_FILE.exists():
        raise SystemExit(
            f"Not found: {CAPTIONS_FILE}\n"
            "Run 1-preprocess_caption/write_captions.py → filter_captions.py first."
        )

    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    jobs = []
    for r in records:
        rel      = Path(r["path_to_image"]).relative_to("train").with_suffix(".dcm")
        dcm_path = TRAIN_DIR / rel
        out_path = (OUT_DIR / rel).with_suffix(".npy")
        if dcm_path.exists():
            jobs.append((dcm_path, out_path))

    print(f"Captions in file  : {len(records):,}")
    print(f"Matching DICOMs   : {len(jobs):,}")
    print(f"Output directory  : {OUT_DIR}  (size={IMAGE_SIZE}×{IMAGE_SIZE})\n")

    ok, failed, skipped = 0, 0, 0

    for dcm_path, out_path in tqdm(jobs, desc="Preprocessing", unit="img"):
        if out_path.exists():
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            arr = preprocess_dcm(dcm_path, size=IMAGE_SIZE)
            np.save(out_path, arr)
            ok += 1
        except Exception as e:
            tqdm.write(f"FAILED {dcm_path.name}: {e}")
            failed += 1

    print(f"\nDone — {ok:,} saved, {skipped:,} already cached, {failed} failed")

    sample_files = list(OUT_DIR.rglob("*.npy"))
    if sample_files:
        s = np.load(sample_files[0])
        print(f"Sample shape : {s.shape}   dtype : {s.dtype}   range : [{s.min():.3f}, {s.max():.3f}]")


if __name__ == "__main__":
    main()
