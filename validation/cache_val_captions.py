import json
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from transformers import (
    CLIPTextConfig, CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5Config, T5EncoderModel,
    T5TokenizerFast,
)

ROOT      = Path(__file__).parent.parent
ENC_DIR   = ROOT.parent / "models" / "text_encoders"
OUT_DIR   = ROOT / "validation" / "cache" / "text"
INDEX_OUT = ROOT / "validation" / "cached_val_captions.json"

DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16
CLIP_MAX = 77
T5_MAX   = 256
T5_DIM   = 4096

PATHOLOGIES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
    "No Finding", "Support Devices",
]


def _slug(pathology: str) -> str:
    return pathology.lower().replace(" ", "_")


def _make_caption(pathology: str) -> str:
    return f"Frontal CXR showing {pathology}."


def _load_weights(path: Path, model: torch.nn.Module) -> torch.nn.Module:
    sd = {k: v.to(DTYPE) for k, v in load_file(path).items()}
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        for prefix in ("text_model.", "transformer.", "module."):
            s = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if s:
                missing, _ = model.load_state_dict(s, strict=False)
                if not missing:
                    break
        else:
            model.load_state_dict(sd, strict=False)
    return model


def _tokenize(tokenizer, text: str, max_length: int) -> torch.Tensor:
    return tokenizer(
        text, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(DEVICE)


def _write_index(captions: list, slugs: list) -> None:
    """(Re)write the JSON index for every pathology, even ones that were
    already cached on a previous run — keeps it in sync with PATHOLOGIES."""
    index = {
        pathology: {
            "caption": captions[i],
            "pt_path": str(OUT_DIR / f"{slugs[i]}.pt"),
        }
        for i, pathology in enumerate(PATHOLOGIES)
    }
    INDEX_OUT.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Index -> {INDEX_OUT}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    captions = [_make_caption(p) for p in PATHOLOGIES]
    slugs    = [_slug(p) for p in PATHOLOGIES]
    N        = len(PATHOLOGIES)

    already  = [s for s in slugs if (OUT_DIR / f"{s}.pt").exists()]
    todo_idx = [i for i, s in enumerate(slugs) if not (OUT_DIR / f"{s}.pt").exists()]

    print(f"Device           : {DEVICE}")
    print(f"Pathologies total: {N}")
    print(f"Already cached   : {len(already)}")
    print(f"To encode        : {len(todo_idx)}\n")

    if not todo_idx:
        print("All validation caption embeddings already cached.")
        _write_index(captions, slugs)
        return

    todo_caps = [captions[i] for i in todo_idx]
    M         = len(todo_idx)

    clip_l_h = [None] * M
    clip_l_p = [None] * M
    clip_g_h = [None] * M
    clip_g_p = [None] * M
    t5_h     = [None] * M

    # Pass 1 — CLIP-L
    print("Pass 1/3 — CLIP-L")
    config_l = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
    clip_l   = _load_weights(
        ENC_DIR / "clip_l.safetensors",
        CLIPTextModel(config_l).to(DTYPE),
    ).to(DEVICE).eval()
    tok_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with torch.no_grad():
        for j, cap in enumerate(todo_caps):
            out = clip_l(_tokenize(tok_l, cap, CLIP_MAX), output_hidden_states=True)
            clip_l_h[j] = out.hidden_states[-2].squeeze(0).cpu()
            clip_l_p[j] = out.pooler_output.squeeze(0).cpu()
            print(f"  [{j + 1:>2}/{M}] {slugs[todo_idx[j]]}")

    del clip_l
    torch.cuda.empty_cache()

    # Pass 2 — CLIP-G
    print("\nPass 2/3 — CLIP-G")
    config_g = CLIPTextConfig.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2")
    clip_g   = _load_weights(
        ENC_DIR / "clip_g.safetensors",
        CLIPTextModelWithProjection(config_g).to(DTYPE),
    ).to(DEVICE).eval()
    tok_g = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")

    with torch.no_grad():
        for j, cap in enumerate(todo_caps):
            out = clip_g(_tokenize(tok_g, cap, CLIP_MAX), output_hidden_states=True)
            clip_g_h[j] = out.hidden_states[-2].squeeze(0).cpu()
            clip_g_p[j] = out.text_embeds.squeeze(0).cpu()
            print(f"  [{j + 1:>2}/{M}] {slugs[todo_idx[j]]}")

    del clip_g
    torch.cuda.empty_cache()

    # Pass 3 — T5-XXL
    print(f"\nPass 3/3 — T5-XXL  (max_sequence_length={T5_MAX})")
    config_t5 = T5Config.from_pretrained("google/t5-v1_1-xxl")
    t5_model  = _load_weights(
        ENC_DIR / "t5xxl.safetensors",
        T5EncoderModel(config_t5).to(DTYPE),
    ).to(DEVICE).eval()
    tok_t5 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

    with torch.no_grad():
        for j, cap in enumerate(todo_caps):
            out    = t5_model(input_ids=_tokenize(tok_t5, cap, T5_MAX))
            t5_h[j] = out.last_hidden_state.squeeze(0).cpu()  # (512, 4096)
            print(f"  [{j + 1:>2}/{M}] {slugs[todo_idx[j]]}")

    del t5_model
    torch.cuda.empty_cache()

    # Save .pt files
    print(f"\nSaving {M} embedding files ...")
    for j, i in enumerate(todo_idx):
        slug = slugs[i]
        clip_h = torch.cat([clip_l_h[j], clip_g_h[j]], dim=-1)
        clip_h = F.pad(clip_h, (0, T5_DIM - clip_h.shape[-1]))  # (77, 4096)

        torch.save(
            {
                "prompt_embeds":        torch.cat([clip_h, t5_h[j]], dim=0),  # (589, 4096)
                "pooled_prompt_embeds": torch.cat([clip_l_p[j], clip_g_p[j]], dim=0),  # (2048,)
            },
            OUT_DIR / f"{slug}.pt",
        )
        print(f"  saved {slug}.pt")

    # Write / update JSON index (covers all pathologies, not just the new ones)
    _write_index(captions, slugs)

    sample = torch.load(OUT_DIR / f"{slugs[todo_idx[0]]}.pt", weights_only=True)
    print(f"\nDone. {M} files -> {OUT_DIR}")
    print(f"prompt_embeds        {tuple(sample['prompt_embeds'].shape)}  {sample['prompt_embeds'].dtype}")
    print(f"pooled_prompt_embeds {tuple(sample['pooled_prompt_embeds'].shape)}  {sample['pooled_prompt_embeds'].dtype}")


if __name__ == "__main__":
    main()