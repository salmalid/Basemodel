"""
Caption format:
  Single   : "Frontal CXR showing Pneumothorax"
  Multiple : "Frontal CXR showing Cardiomegaly and Pleural Effusion"
  Three+   : "Frontal CXR showing Atelectasis, Consolidation, and Pleural Effusion"
  Normal   : "Frontal CXR showing no acute findings"

Rules:
  - Frontal views only (PA / AP); laterals are discarded
  - Only confident positives (label == 1.0); uncertain (-1) are ignored
  - Records with no confident positive label (and not No Finding) are discarded
  - Support Devices is excluded whether alone or combined with other pathologies
"""
import json
import os
from pathlib import Path

_DIR        = Path(os.path.dirname(os.path.abspath(__file__)))
BM_ROOT     = _DIR.parent
REPORT_FILE = BM_ROOT.parent / "Data_original" / "metadata" / "findings_fixed.json"
OUTPUT      = _DIR / "captions.json"

PATHOLOGIES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
]


def _is_frontal(path: str) -> bool:
    p = path.lower()
    return "frontal" in p and "lateral" not in p


def _join(active: list) -> str:
    if len(active) == 1:
        return active[0]
    if len(active) == 2:
        return f"{active[0]} and {active[1]}"
    return ", ".join(active[:-1]) + f", and {active[-1]}"


def build_caption(record: dict) -> str | None:
    active     = [p for p in PATHOLOGIES if record.get(p) == 1.0]
    no_finding = record.get("No Finding") == 1.0

    if no_finding and not active:
        return "Frontal CXR showing no acute findings"
    if not active:
        return None
    return f"Frontal CXR showing {_join(active)}"


def main():
    if not REPORT_FILE.exists():
        raise SystemExit(f"Missing {REPORT_FILE}")

    with open(REPORT_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    results             = []
    skipped_lateral     = 0
    skipped_no_positive = 0

    for r in records:
        if not _is_frontal(r.get("path_to_image", "")):
            skipped_lateral += 1
            continue

        caption = build_caption(r)
        if caption is None:
            skipped_no_positive += 1
            continue

        parts      = r["path_to_image"].split("/")
        patient_id = parts[1] if len(parts) > 1 else ""
        study_id   = parts[2] if len(parts) > 2 else ""
        image_id   = Path(parts[3]).stem if len(parts) > 3 else ""

        results.append({
            "path_to_image": r["path_to_image"],
            "patient_id":    patient_id,
            "study_id":      study_id,
            "image_id":      image_id,
            "caption":       caption,
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Written {len(results):,} captions to {OUTPUT.name}")
    print(f"  Skipped {skipped_lateral:,} lateral views")
    print(f"  Skipped {skipped_no_positive:,} records with no confident positive label")


if __name__ == "__main__":
    main()
