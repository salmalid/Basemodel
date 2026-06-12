"""Validate captions fit within CLIP and T5 token limits.

"Frontal CXR showing X" captions are very short and will always pass,
but this confirms it before running the expensive text-embedding cache step.
"""
import json
import sys
from pathlib import Path

_DIR          = Path(__file__).parent
BM_ROOT       = _DIR.parent
CAPTIONS_FILE = BM_ROOT / "dataset" / "captions_filtered.json"
REPORT_FILE   = _DIR / "check_captions_report.json"

CLIP_HARD_LIMIT = 77
CLIP_WARN_TOKEN = 55
T5_HARD_LIMIT   = 256

PATHOLOGIES = [
    "enlarged cardiomediastinum", "cardiomegaly", "lung opacity", "lung lesion",
    "edema", "consolidation", "pneumonia", "atelectasis", "pneumothorax",
    "pleural effusion", "pleural other", "fracture", "support devices", "no acute",
]


def first_pathology_token(ids: list, tokenizer, n: int = CLIP_WARN_TOKEN) -> int:
    full_ids  = list(ids)
    decoded_n = tokenizer.decode(full_ids[:n]).lower()
    if not any(p in decoded_n for p in PATHOLOGIES):
        return -1
    for i in range(1, min(n + 1, len(full_ids) + 1)):
        decoded_prefix = tokenizer.decode(full_ids[:i]).lower()
        for p in PATHOLOGIES:
            if p in decoded_prefix:
                return i - 1
    return -1


def main():
    if not CAPTIONS_FILE.exists():
        sys.exit(f"Not found: {CAPTIONS_FILE}\nRun write_captions.py + filter_captions.py first.")

    from transformers import CLIPTokenizer, T5TokenizerFast

    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print("Loading tokenizers ...")
    tok_l  = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tok_g  = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    tok_t5 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

    flags = []
    t5_over, clip_buried = 0, 0

    for r in records:
        cap    = r["caption"]
        stem   = f"{r['patient_id']}_{r['study_id']}_{r['image_id']}"
        issues = []

        t5_ids = tok_t5(cap, add_special_tokens=True).input_ids
        if len(t5_ids) > T5_HARD_LIMIT:
            issues.append(f"T5 {len(t5_ids)} tokens > {T5_HARD_LIMIT}")
            t5_over += 1

        cl_ids = tok_l(cap, add_special_tokens=False).input_ids
        cl_pos = first_pathology_token(cl_ids, tok_l)
        if cl_pos > CLIP_WARN_TOKEN or cl_pos == -1:
            issues.append(f"CLIP-L pathology at token {cl_pos} (want ≤{CLIP_WARN_TOKEN})")
            clip_buried += 1

        cg_ids = tok_g(cap, add_special_tokens=False).input_ids
        cg_pos = first_pathology_token(cg_ids, tok_g)
        if cg_pos > CLIP_WARN_TOKEN or cg_pos == -1:
            issues.append(f"CLIP-G pathology at token {cg_pos} (want ≤{CLIP_WARN_TOKEN})")

        if issues:
            flags.append({"stem": stem, "issues": issues, "caption_preview": cap[:120]})

    print(f"\n{'='*60}")
    print(f"Captions checked  : {len(records):,}")
    print(f"T5 > 512 tokens   : {t5_over:,}  ({100*t5_over/max(len(records),1):.2f}%)")
    print(f"CLIP buried       : {clip_buried:,}  ({100*clip_buried/max(len(records),1):.2f}%)")
    print(f"Total flagged     : {len(flags):,}")

    if flags:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(flags, f, indent=2)
        print(f"\nReport written → {REPORT_FILE}")
        print("Fix flagged captions in write_captions.py before caching embeddings.")
    else:
        print("\nAll captions pass token-budget checks.")


if __name__ == "__main__":
    main()
