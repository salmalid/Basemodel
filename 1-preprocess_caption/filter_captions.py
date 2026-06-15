import json
import os
from pathlib import Path

_DIR      = Path(os.path.dirname(os.path.abspath(__file__)))
BM_ROOT   = _DIR.parent
PNG_ROOT  = BM_ROOT.parent / "Data_original" / "png_chexpert_plus_chunk_4"

CAPTIONS = _DIR / "captions.json"
OUTPUT   = BM_ROOT / "dataset" / "captions_filtered.json"


def main():
    if not CAPTIONS.exists():
        raise SystemExit(f"Missing {CAPTIONS} — run write_captions.py first.")

    with open(CAPTIONS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Total captions : {len(records):,}")
    print(f"Matching against PNGs in {PNG_ROOT} ...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    kept, missing = 0, 0
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for r in records:
            # metadata paths have .jpg extension; actual files on disk are .png
            png_path = PNG_ROOT / Path(r["path_to_image"]).with_suffix(".png")
            if png_path.exists():
                out.write(json.dumps(r) + "\n")
                kept += 1
            else:
                missing += 1

    print(f"Kept    : {kept:,}")
    print(f"Missing : {missing:,}  (PNG not found on disk)")
    print(f"Output  : {OUTPUT}")


if __name__ == "__main__":
    main()
