from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

_DIR        = Path(__file__).parent
BM_ROOT     = _DIR.parent
PARENT_ROOT = BM_ROOT.parent

PREPROC_DIR = PARENT_ROOT / "dataset" / "preprocessed1024" / "train"
OUT_DIR     = PARENT_ROOT / "dataset" / "cache" / "latents" / "train"
MODEL_PATH  = BM_ROOT / "models" / "sd3.5_medium.safetensors"

SCALING_FACTOR = 1.5305
SHIFT_FACTOR   = 0.0609
BATCH_SIZE     = 2


def load_vae(device: str) -> AutoencoderKL:
    print(f"Loading VAE from {MODEL_PATH.name} ...")
    vae = AutoencoderKL.from_single_file(
        str(MODEL_PATH),
        config="stabilityai/stable-diffusion-3.5-medium",
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device)
    vae.eval()
    print(f"  scaling_factor = {vae.config.scaling_factor}")
    print(f"  shift_factor   = {vae.config.shift_factor}")
    return vae


@torch.no_grad()
def encode_batch(vae: AutoencoderKL, arrays: list, device: str) -> np.ndarray:
    """arrays: list of (1024,1024) float32 in [0,1].
    Returns (B, 16, 128, 128) float16 — scaled x0 ready for training."""
    imgs     = np.stack([np.stack([a * 2.0 - 1.0] * 3) for a in arrays])
    tensor   = torch.from_numpy(imgs).to(device, dtype=torch.float16)
    raw_mean = vae.encode(tensor).latent_dist.mean
    x0       = (raw_mean.float() - SHIFT_FACTOR) * SCALING_FACTOR
    return x0.cpu().half().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    if not PREPROC_DIR.exists():
        raise SystemExit(
            f"Preprocessed images not found: {PREPROC_DIR}\n"
            "Run 2-preprocess_DICOM/preprocess_all.py first."
        )

    npy_files = sorted(PREPROC_DIR.rglob("*.npy"))
    print(f"Preprocessed images : {len(npy_files):,}")
    if not npy_files:
        raise SystemExit("No .npy files found — check PREPROC_DIR path.")

    sample = np.load(npy_files[0])
    if sample.shape != (1024, 1024):
        raise SystemExit(f"Expected (1024, 1024) but got {sample.shape}.")

    vae = load_vae(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0
    batch_paths: list = []
    batch_arrs:  list = []

    def flush():
        nonlocal ok
        latents = encode_batch(vae, batch_arrs, device)
        for path, latent in zip(batch_paths, latents):
            out = OUT_DIR / path.relative_to(PREPROC_DIR)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, latent)
            ok += 1
        batch_paths.clear()
        batch_arrs.clear()

    for f in tqdm(npy_files, desc="VAE encoding", unit="img"):
        out = OUT_DIR / f.relative_to(PREPROC_DIR)
        if out.exists():
            skipped += 1
            continue
        batch_paths.append(f)
        batch_arrs.append(np.load(f))
        if len(batch_paths) >= BATCH_SIZE:
            flush()

    if batch_paths:
        flush()

    print(f"\nDone — {ok:,} latents saved, {skipped:,} already cached")
    if ok > 0:
        sample_out = np.load(next(OUT_DIR.rglob("*.npy")))
        print(f"Latent shape : {sample_out.shape}   dtype : {sample_out.dtype}")
        print(f"Expected     : (16, 128, 128)  float16")
        print(f"Value range  : [{sample_out.min():.3f}, {sample_out.max():.3f}]  (scaled x0)")


if __name__ == "__main__":
    main()