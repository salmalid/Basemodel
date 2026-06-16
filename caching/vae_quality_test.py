"""
Sanity-check the image pipeline + VAE before running the full
preprocess.py -> cache_vae.py pass over the whole (balanced) dataset.

For a handful of real PNGs (sampled from dataset/captions_filtered.json, so
they're the actual images that would be trained on) this:
  1. Loads the PNG, percentile-clips + normalises to [0,1]   (same as preprocess.py)
  2. Resizes + center-pads to 1024x1024 using torchvision     (antialiased bicubic,
     replacing the current PIL/Lanczos resize in preprocess.py — swap that file's
     _pad_and_resize() over to this once you're happy with the results here)
  3. Encodes through the real SD3.5 VAE, then decodes the latent back
  4. Reports PSNR/MSE between the preprocessed input and the VAE reconstruction,
     and saves a visual grid: [preprocessed | latent (16ch avg) | reconstruction]

Nothing here writes to dataset/preprocessed1024 or dataset/cache — it's purely
a check. Run preprocess.py + cache_vae.py for real once this looks good.

Usage:
    python caching/vae_quality_test.py --n 6 --out caching/vae_quality_test.png
"""
import argparse
import random
from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from PIL import Image

_DIR    = Path(__file__).parent
BM_ROOT = _DIR.parent

PNG_ROOT      = BM_ROOT.parent / "Data_original" / "png_chexpert_plus"
CAPTIONS_FILE = BM_ROOT / "dataset" / "captions_filtered.json"
MODEL_PATH    = BM_ROOT.parent / "models" / "sd3.5_medium.safetensors"

SCALING_FACTOR = 1.5305
SHIFT_FACTOR   = 0.0609
SIZE           = 1024


def load_normalized(png_path: Path) -> np.ndarray:
    """PNG -> (H,W) float32 [0,1], percentile-clipped. Same as preprocess.py,
    minus the resize step (kept separate so we can swap resize methods)."""
    img = Image.open(png_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0

    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr  = np.clip(arr, lo, hi)
    span = hi - lo
    return (arr - lo) / span if span > 0 else np.zeros_like(arr)


def pad_and_resize_torchvision(arr_01: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Aspect-ratio-preserving antialiased resize + center zero-pad, via torchvision.
    Candidate replacement for preprocess.py's PIL/Lanczos _pad_and_resize()."""
    tensor = torch.from_numpy(arr_01).unsqueeze(0)          # (1,H,W)
    _, h, w = tensor.shape
    scale = size / max(h, w)
    new_h = min(round(h * scale), size)
    new_w = min(round(w * scale), size)

    resized = TF.resize(
        tensor, [new_h, new_w],
        interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
    ).clamp(0.0, 1.0)

    pad_top    = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left   = (size - new_w) // 2
    pad_right  = size - new_w - pad_left
    padded = TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0.0)
    return padded.squeeze(0).numpy()


@torch.no_grad()
def encode_decode(vae: AutoencoderKL, arr_01: np.ndarray, device: str):
    """Returns (latent_mean_map 128x128 in [0,1], decoded image 1024x1024 in [0,1])."""
    img    = np.stack([arr_01 * 2.0 - 1.0] * 3)[None]        # (1,3,H,W) [-1,1]
    tensor = torch.from_numpy(img).to(device, dtype=torch.float16)

    raw_mean = vae.encode(tensor).latent_dist.mean             # (1,16,128,128)
    x0       = (raw_mean.float() - SHIFT_FACTOR) * SCALING_FACTOR
    vae_in   = x0 / SCALING_FACTOR + SHIFT_FACTOR
    rec      = vae.decode(vae_in.half()).sample                # (1,3,H,W) [-1,1]

    lat_np  = raw_mean.squeeze(0).float().cpu().numpy()
    lat_map = lat_np.mean(0)
    lo, hi  = lat_map.min(), lat_map.max()
    lat_map = (lat_map - lo) / (hi - lo + 1e-8)

    rec_np = rec.squeeze(0).mean(0).float().cpu().numpy()
    rec_np = (rec_np.clip(-1, 1) + 1) / 2
    return lat_map, rec_np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=6)
    parser.add_argument("--out",  default=str(BM_ROOT / "caching" / "vae_quality_test.png"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not CAPTIONS_FILE.exists():
        raise SystemExit(f"Missing {CAPTIONS_FILE} — run the caption pipeline first.")

    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rng     = random.Random(args.seed)
    samples = rng.sample(records, min(args.n, len(records)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VAE ({device}) ...")
    vae = AutoencoderKL.from_single_file(
        str(MODEL_PATH),
        config="stabilityai/stable-diffusion-3.5-medium",
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device).eval()

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n))
    fig.suptitle(
        "Pipeline check: PNG -> percentile-clip -> torchvision resize/pad -> VAE encode/decode",
        fontsize=12, fontweight="bold", y=1.005,
    )
    col_titles = [
        "Preprocessed (torchvision resize)\nDataset input",
        "Latent mean (16ch avg, 128x128)",
        "VAE reconstruction\n(decoded from latent)",
    ]
    if n == 1:
        axes = [axes]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=10, pad=6)

    psnrs = []
    for row, rec in enumerate(samples):
        png_path = PNG_ROOT / Path(rec["path_to_image"]).with_suffix(".png")
        print(f"[{row+1}/{n}] {rec['path_to_image']}  ({rec['caption']})")

        if not png_path.exists():
            for col in range(3):
                axes[row][col].text(0.5, 0.5, "PNG not found", ha="center",
                                    va="center", transform=axes[row][col].transAxes,
                                    color="red")
                axes[row][col].axis("off")
            continue

        normed   = load_normalized(png_path)
        prepped  = pad_and_resize_torchvision(normed)
        lat_map, recon = encode_decode(vae, prepped, device)
        score = psnr(prepped, recon)
        psnrs.append(score)

        axes[row][0].imshow(prepped, cmap="gray", vmin=0, vmax=1)
        axes[row][1].imshow(lat_map, cmap="viridis")
        axes[row][2].imshow(recon,   cmap="gray", vmin=0, vmax=1)
        for col in range(3):
            axes[row][col].axis("off")
        axes[row][2].set_xlabel(f"PSNR = {score:.1f} dB", fontsize=9)
        axes[row][0].set_ylabel(rec["caption"][:40], fontsize=7, rotation=0,
                                labelpad=90, va="center")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=120, bbox_inches="tight")

    if psnrs:
        print(f"\nMean PSNR over {len(psnrs)} samples: {np.mean(psnrs):.2f} dB "
              f"(higher is better; >30 dB is generally a good encode/decode round-trip)")
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
