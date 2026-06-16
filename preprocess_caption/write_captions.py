import json
import os
from pathlib import Path

_DIR         = Path(os.path.dirname(os.path.abspath(__file__)))
BM_ROOT      = _DIR.parent
REPORT_FILE  = BM_ROOT.parent / "Data_original" / "metadata" / "findings_fixed.json"
OUTPUT       = _DIR / "captions.json"
VALID_OUTPUT = _DIR / "captions_valid.json"

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
    active = [p for p in PATHOLOGIES if record.get(p) == 1.0]
    if not active:
        return None
    return f"Frontal CXR showing {_join(active)}"


def main():
    if not REPORT_FILE.exists():
        raise SystemExit(f"Missing {REPORT_FILE}")

    with open(REPORT_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    results                  = []   # train split -> feeds the training pipeline
    valid_results             = []  # valid split -> held out, written separately
    skipped_lateral          = 0
    skipped_support_devices  = 0
    skipped_no_positive      = 0
    skipped_unknown_split    = 0

    for r in records:
        if not _is_frontal(r.get("path_to_image", "")):
            skipped_lateral += 1
            continue

        if r.get("Support Devices") == 1.0:
            skipped_support_devices += 1
            continue

        caption = build_caption(r)
        if caption is None:
            skipped_no_positive += 1
            continue

        parts      = r["path_to_image"].split("/")
        split      = parts[0] if parts else ""
        patient_id = parts[1] if len(parts) > 1 else ""
        study_id   = parts[2] if len(parts) > 2 else ""
        image_id   = Path(parts[3]).stem if len(parts) > 3 else ""

        record = {
            "path_to_image": r["path_to_image"],
            "patient_id":    patient_id,
            "study_id":      study_id,
            "image_id":      image_id,
            "caption":       caption,
        }

        # findings_fixed.json covers both CheXpert+ splits — keep them in
        # separate files so "train" only ever reaches the training pipeline
        # and "valid" stays held out for evaluation.
        if split == "train":
            results.append(record)
        elif split == "valid":
            valid_results.append(record)
        else:
            skipped_unknown_split += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(VALID_OUTPUT, "w", encoding="utf-8") as f:
        for r in valid_results:
            f.write(json.dumps(r) + "\n")

    print(f"Written {len(results):,} train captions to {OUTPUT.name}")
    print(f"Written {len(valid_results):,} valid captions to {VALID_OUTPUT.name}")
    print(f"  Skipped {skipped_lateral:,} lateral views")
    print(f"  Skipped {skipped_support_devices:,} records with Support Devices (alone or combined)")
    print(f"  Skipped {skipped_no_positive:,} records with no confident positive pathology (incl. No Finding)")
    if skipped_unknown_split:
        print(f"  Skipped {skipped_unknown_split:,} records with unrecognized split prefix")


if __name__ == "__main__":
    main()
