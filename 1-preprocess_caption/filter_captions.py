"""Filter captions to images that actually exist on disk.

Reads  : 1-preprocess_caption/captions.json
Checks : C:\\...\\PFE-project\\dataset\\train\\**\\*.dcm
Writes : BASEMODEL/dataset/captions_filtered.json
"""
import json
import os
from pathlib import Path

_DIR        = Path(os.path.dirname(os.path.abspath(__file__)))
BM_ROOT     = _DIR.parent
PARENT_ROOT = BM_ROOT.parent

CAPTIONS  = _DIR / "captions.json"
TRAIN_DIR = PARENT_ROOT / "dataset" / "train"
OUTPUT    = BM_ROOT / "dataset" / "captions_filtered.json"

if not CAPTIONS.exists():
    raise SystemExit(f"Missing {CAPTIONS} — run write_captions.py first.")

with open(CAPTIONS, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total captions : {len(records):,}")
print(f"Matching against DICOM files in {TRAIN_DIR} ...")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
kept, missing = 0, 0
with open(OUTPUT, "w", encoding="utf-8") as out:
    for r in records:
        if not r["path_to_image"].startswith("train/"):
            missing += 1
            continue
        rel = Path(r["path_to_image"]).relative_to("train").with_suffix(".dcm")
        if (TRAIN_DIR / rel).exists():
            out.write(json.dumps(r) + "\n")
            kept += 1
        else:
            missing += 1

print(f"Kept    : {kept:,}")
print(f"Missing : {missing:,}  (DICOM not found on disk)")
print(f"Output  : {OUTPUT}")
