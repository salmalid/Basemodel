import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from tqdm import tqdm

try:
    import torchxrayvision as xrv
except ImportError:
    sys.exit(
        "torchxrayvision not installed.\n"
        "  pip install torchxrayvision"
    )

_DIR    = Path(__file__).parent
BM_ROOT = _DIR.parent

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

XRV_SIZE = 224


class DenseNetFeatures(nn.Module):
    """DenseNet-121 with classifier head removed; returns 1024-dim features."""

    def __init__(self, weights: str = "densenet121-res224-all"):
        super().__init__()
        model        = xrv.models.DenseNet(weights=weights)
        self.features = model.features
        self.relu     = nn.ReLU(inplace=True)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.relu(out)
        out = self.pool(out)
        return out.flatten(1)


def load_extractor(device: str) -> DenseNetFeatures:
    print("Loading DenseNet-121 (densenet121-res224-all) from TorchXRayVision ...")
    return DenseNetFeatures().to(device).eval()


def _npy_to_xrv(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    pil = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (XRV_SIZE, XRV_SIZE), Image.LANCZOS
    )
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = arr * 2048.0 - 1024.0
    return arr[None]


def _png_to_xrv(path: Path) -> np.ndarray:
    pil = Image.open(path).convert("L").resize((XRV_SIZE, XRV_SIZE), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = arr * 2048.0 - 1024.0
    return arr[None]


@torch.no_grad()
def extract_features(
    model: DenseNetFeatures,
    image_paths: list,
    load_fn,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    all_feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i : i + batch_size]
        imgs  = np.stack([load_fn(p) for p in batch_paths])
        t     = torch.from_numpy(imgs).to(device)
        feats = model(t).cpu().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


def compute_fid(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    if len(real_feats) < 2 or len(gen_feats) < 2:
        return float("nan")

    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)

    diff           = mu_r - mu_g
    covmean, _     = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


def main():
    parser = argparse.ArgumentParser(description="Medical FID for BASEMODEL CXR generation")
    parser.add_argument("--real",   default=str(BM_ROOT / "dataset" / "preprocessed1024" / "train"),
                        help="Directory of .npy real images")
    parser.add_argument("--gen",    default=str(BM_ROOT / "outputs"),
                        help="Directory of .png generated images")
    parser.add_argument("--labels", default=str(BM_ROOT / "dataset" / "captions_filtered.json"),
                        help="captions_filtered.json for per-pathology breakdown")
    parser.add_argument("--out",    default=str(BM_ROOT / "5-evaluation" / "fid_results.json"))
    parser.add_argument("--batch",  type=int, default=32)
    args = parser.parse_args()

    real_dir   = Path(args.real)
    gen_dir    = Path(args.gen)
    labels_f   = Path(args.labels)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_extractor(device)

    real_npy = sorted(real_dir.rglob("*.npy"))
    if not real_npy:
        sys.exit(f"No .npy files found in {real_dir}")
    print(f"\nReal images  : {len(real_npy):,}")
    real_feats = extract_features(model, real_npy, _npy_to_xrv, device, args.batch)

    gen_png = sorted(gen_dir.rglob("*.png"))
    if not gen_png:
        sys.exit(f"No .png files found in {gen_dir}")
    print(f"Generated    : {len(gen_png):,}")
    gen_feats = extract_features(model, gen_png, _png_to_xrv, device, args.batch)

    overall_fid = compute_fid(real_feats, gen_feats)
    print(f"\nOverall FID  : {overall_fid:.2f}")

    results = {"overall_fid": overall_fid, "per_pathology_fid": {}}

    if labels_f.exists():
        with open(labels_f, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        stem_pathologies: dict = {}
        for r in records:
            stem = f"{r['patient_id']}_{r['study_id']}_{r['image_id']}"
            cap  = r["caption"].lower()
            pats = [p for p in PATHOLOGIES if p.lower() in cap]
            stem_pathologies[stem] = pats

        real_stem_map: dict = {}
        for p in real_npy:
            real_stem_map[p.stem] = p

        gen_path_by_patho: dict = {p: [] for p in PATHOLOGIES}
        for p in gen_png:
            name = p.stem.lower()
            for pat in PATHOLOGIES:
                if pat.lower().replace(" ", "_") in name or pat.lower().replace(" ", "-") in name:
                    gen_path_by_patho[pat].append(p)
                    break

        for patho in PATHOLOGIES:
            real_paths_p = [
                real_stem_map[stem]
                for stem, pats in stem_pathologies.items()
                if patho in pats and stem in real_stem_map
            ]
            gen_paths_p = gen_path_by_patho[patho]

            if len(real_paths_p) < 2 or len(gen_paths_p) < 1:
                print(f"  {patho:<30} SKIP (real={len(real_paths_p)}, gen={len(gen_paths_p)})")
                results["per_pathology_fid"][patho] = None
                continue

            r_feats = extract_features(model, real_paths_p, _npy_to_xrv, device, args.batch)
            g_feats = extract_features(model, gen_paths_p,  _png_to_xrv, device, args.batch)
            fid_p   = compute_fid(r_feats, g_feats)
            print(f"  {patho:<30} FID = {fid_p:.2f}  (real={len(real_paths_p)}, gen={len(gen_paths_p)})")
            results["per_pathology_fid"][patho] = fid_p

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()


