"""
T5_MAX is 256 instead of 512: short "Frontal CXR showing X" captions
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from transformers import (
    CLIPTextConfig, CLIPTextModel,
    CLIPTextModelWithProjection, CLIPTokenizer,
    T5Config, T5EncoderModel, T5TokenizerFast,
)

_DIR          = Path(__file__).parent
BM_ROOT       = _DIR.parent

ENC_DIR       = BM_ROOT / "models" / "text_encoders"
CAPTIONS_FILE = BM_ROOT / "dataset" / "captions_filtered.json"
CACHE_DIR     = BM_ROOT / "dataset" / "cache" / "text"

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — activate the project venv first.")

DEVICE   = "cuda:0"
DTYPE    = torch.float16
CLIP_MAX = 77
T5_MAX   = 256
T5_DIM   = 4096

CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


if not CAPTIONS_FILE.exists():
    raise SystemExit(f"Not found: {CAPTIONS_FILE}\nRun write_captions.py + filter_captions.py first.")

stems    = []
captions = []
for line in CAPTIONS_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    data = json.loads(line)
    stems.append(f"{data['patient_id']}_{data['study_id']}_{data['image_id']}")
    captions.append(data["caption"])

N             = len(stems)
todo_stems    = [s for s in stems    if not (CACHE_DIR / f"{s}.pt").exists()]
todo_captions = [captions[i] for i, s in enumerate(stems)
                 if not (CACHE_DIR / f"{s}.pt").exists()]

if not todo_stems:
    print("All cache files already exist. Nothing to do.")
    raise SystemExit(0)

print(f"Captions total : {N:,}")
print(f"Already cached : {N - len(todo_stems):,}")
print(f"To encode      : {len(todo_stems):,}")

unique_caps = list(dict.fromkeys(todo_captions))
cap_to_idx  = {c: i for i, c in enumerate(unique_caps)}
U           = len(unique_caps)
print(f"Unique captions: {U:,}  (deduplication saves {len(todo_stems) - U:,} forward passes)\n")

cpu_clip_l_h = [None] * U
cpu_clip_l_p = [None] * U
cpu_clip_g_h = [None] * U
cpu_clip_g_p = [None] * U
cpu_t5_h     = [None] * U


print("Pass 1/3 — CLIP-L")
config_l = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
clip_l   = _load_weights(
    ENC_DIR / "clip_l.safetensors",
    CLIPTextModel(config_l).to(DTYPE),
).to(DEVICE).eval()
tok_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

with torch.no_grad():
    for j, cap in enumerate(unique_caps, 1):
        out = clip_l(input_ids=_tokenize(tok_l, cap, CLIP_MAX), output_hidden_states=True)
        cpu_clip_l_h[j - 1] = out.hidden_states[-2].squeeze(0).cpu()
        cpu_clip_l_p[j - 1] = out.pooler_output.squeeze(0).cpu()
        if j % 100 == 0 or j == U:
            print(f"  [{j:>5}/{U}]")

del clip_l
torch.cuda.empty_cache()



print("Pass 2/3 — CLIP-G")
config_g = CLIPTextConfig.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2")
clip_g   = _load_weights(
    ENC_DIR / "clip_g.safetensors",
    CLIPTextModelWithProjection(config_g).to(DTYPE),
).to(DEVICE).eval()
tok_g = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")

with torch.no_grad():
    for j, cap in enumerate(unique_caps, 1):
        out = clip_g(input_ids=_tokenize(tok_g, cap, CLIP_MAX), output_hidden_states=True)
        cpu_clip_g_h[j - 1] = out.hidden_states[-2].squeeze(0).cpu()
        cpu_clip_g_p[j - 1] = out.text_embeds.squeeze(0).cpu()
        if j % 100 == 0 or j == U:
            print(f"  [{j:>5}/{U}]")

del clip_g
torch.cuda.empty_cache()



print(f"Pass 3/3 — T5-XXL  (max_sequence_length={T5_MAX})")
config_t5 = T5Config.from_pretrained("google/t5-v1_1-xxl")
t5        = _load_weights(
    ENC_DIR / "t5xxl.safetensors",
    T5EncoderModel(config_t5).to(DTYPE),
).to(DEVICE).eval()
tok_t5 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

with torch.no_grad():
    for j, cap in enumerate(unique_caps, 1):
        out = t5(input_ids=_tokenize(tok_t5, cap, T5_MAX))
        cpu_t5_h[j - 1] = out.last_hidden_state.squeeze(0).cpu()
        if j % 100 == 0 or j == U:
            print(f"  [{j:>5}/{U}]")

del t5
torch.cuda.empty_cache()



print(f"\nSaving {len(todo_stems):,} cache files ...")
for stem, cap in zip(todo_stems, todo_captions):
    uidx   = cap_to_idx[cap]
    clip_h = torch.cat([cpu_clip_l_h[uidx], cpu_clip_g_h[uidx]], dim=-1)
    clip_h = F.pad(clip_h, (0, T5_DIM - clip_h.shape[-1]))

    torch.save(
        {
            "prompt_embeds":        torch.cat([clip_h, cpu_t5_h[uidx]], dim=0),
            "pooled_prompt_embeds": torch.cat([cpu_clip_l_p[uidx], cpu_clip_g_p[uidx]], dim=0),
        },
        CACHE_DIR / f"{stem}.pt",
    )

sample = torch.load(CACHE_DIR / f"{todo_stems[0]}.pt", weights_only=True)
print(f"\nDone. {len(todo_stems):,} files → {CACHE_DIR}")
print(f"  prompt_embeds        {tuple(sample['prompt_embeds'].shape)}  {sample['prompt_embeds'].dtype}")
print(f"  pooled_prompt_embeds {tuple(sample['pooled_prompt_embeds'].shape)}  {sample['pooled_prompt_embeds'].dtype}")
