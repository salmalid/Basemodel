import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_DIR           = Path(__file__).parent
BM_ROOT        = _DIR.parent
PARENT_ROOT    = BM_ROOT.parent

CAPTIONS_FILE  = BM_ROOT / "dataset" / "captions_filtered.json"
LATENT_DIR     = PARENT_ROOT / "dataset" / "cache" / "latents" / "train"
TEXT_CACHE_DIR = BM_ROOT / "dataset" / "cache" / "text"

EXPECTED_LATENT_SHAPE = (16, 128, 128)


def _text_stem(r: dict) -> str:
    return f"{r['patient_id']}_{r['study_id']}_{r['image_id']}"


class CXRDataset(Dataset):
    def __init__(
        self,
        captions_file=CAPTIONS_FILE,
        latent_dir=LATENT_DIR,
        text_cache_dir=TEXT_CACHE_DIR,
        expected_latent_shape=EXPECTED_LATENT_SHAPE,
    ):
        self.expected_latent_shape = tuple(expected_latent_shape)

        captions_file  = Path(captions_file)
        latent_dir     = Path(latent_dir)
        text_cache_dir = Path(text_cache_dir)

        if not captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {captions_file}")

        with open(captions_file, encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        self.records = []
        missing_lat, missing_txt = 0, 0

        for r in all_records:
            rel         = Path(r["path_to_image"]).relative_to("train").with_suffix(".npy")
            latent_path = latent_dir / rel
            text_path   = text_cache_dir / f"{_text_stem(r)}.pt"

            if not latent_path.exists():
                missing_lat += 1
                continue
            if not text_path.exists():
                missing_txt += 1
                continue

            self.records.append({
                "latent_path": str(latent_path),
                "text_path":   str(text_path),
                "caption":     r["caption"],
                "image_id":    _text_stem(r),
            })

        print(
            f"CXRDataset: {len(self.records):,} samples ready  "
            f"(skipped: {missing_lat} no latent, {missing_txt} no text cache)"
        )
        if missing_lat:
            print(f"  WARNING: {missing_lat} missing latents — run 3-caching/cache_vae.py")
        if missing_txt:
            print(f"  WARNING: {missing_txt} missing text cache — run 3-caching/cache_text.py")
        if not self.records:
            raise RuntimeError(
                "No valid samples — run 3-caching/cache_vae.py and cache_text.py first.\n"
                f"  Latent dir : {latent_dir}  (exists={latent_dir.exists()})\n"
                f"  Text dir   : {text_cache_dir}  (exists={text_cache_dir.exists()})"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec    = self.records[idx]
        latent = torch.from_numpy(np.load(rec["latent_path"]))

        if tuple(latent.shape) != self.expected_latent_shape:
            raise ValueError(
                f"Latent shape mismatch: got {tuple(latent.shape)}, "
                f"expected {self.expected_latent_shape}\n"
                f"File: {rec['latent_path']}\n"
                "Delete dataset/cache/latents/ and re-run 3-caching/cache_vae.py."
            )

        text = torch.load(rec["text_path"], weights_only=True)
        return {
            "latents":              latent,
            "prompt_embeds":        text["prompt_embeds"],
            "pooled_prompt_embeds": text["pooled_prompt_embeds"],
            "caption":              rec["caption"],
            "image_id":             rec["image_id"],
        }

    def __repr__(self) -> str:
        return f"CXRDataset(n={len(self.records)}, latent_shape={self.expected_latent_shape})"


def build_dataloader(batch_size: int = 1, num_workers: int = 0, shuffle: bool = True) -> DataLoader:
    dataset = CXRDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("CXRDataset smoke test")
    print("=" * 60)
    try:
        ds = CXRDataset()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\n{ds}")
    print("\nChecking first 3 samples ...")
    for i in range(min(3, len(ds))):
        item = ds[i]
        print(
            f"  [{i}] latents={tuple(item['latents'].shape)} "
            f"pe={tuple(item['prompt_embeds'].shape)} "
            f"ppe={tuple(item['pooled_prompt_embeds'].shape)}"
        )

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch  = next(iter(loader))
    print(f"\nBatch latents              : {tuple(batch['latents'].shape)}  {batch['latents'].dtype}")
    print(f"Batch prompt_embeds        : {tuple(batch['prompt_embeds'].shape)}")
    print(f"Batch pooled_prompt_embeds : {tuple(batch['pooled_prompt_embeds'].shape)}")
    print(f"\nSample caption:\n  {batch['caption'][0]}")
    print("\nAll checks passed.")
