#!/usr/bin/env python3
import argparse
import csv
import gc
import os
from pathlib import Path

import torch.distributed as dist
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from safetensors.torch import load_file
from transformers import (
    CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
    T5Config, T5EncoderModel, T5Tokenizer,
)
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT.parent / "models" / "sd3.5_medium.safetensors"
ENC_DIR    = ROOT.parent / "models" / "text_encoders"
LORA_DIR   = ROOT / "checkpoints" / "best"
OUT_DIR    = ROOT / "outputs"
GEN_CSV    = ROOT / "outputs" / "generations.csv"

SCALING_FACTOR = 1.5305
SHIFT_FACTOR   = 0.0609
CLIP_MAX       = 77
T5_MAX         = 256
T5_DIM         = 4096
IMG_SIZE       = 1024
SEQ_LEN        = CLIP_MAX + T5_MAX 

PATHOLOGY_PROMPTS = {
    "Enlarged Cardiomediastinum": "Frontal CXR showing Enlarged Cardiomediastinum",
    "Cardiomegaly":               "Frontal CXR showing Cardiomegaly",
    "Lung Opacity":               "Frontal CXR showing Lung Opacity",
    "Lung Lesion":                "Frontal CXR showing Lung Lesion",
    "Edema":                      "Frontal CXR showing Edema",
    "Consolidation":              "Frontal CXR showing Consolidation",
    "Pneumonia":                  "Frontal CXR showing Pneumonia",
    "Atelectasis":                "Frontal CXR showing Atelectasis",
    "Pneumothorax":               "Frontal CXR showing Pneumothorax",
    "Pleural Effusion":           "Frontal CXR showing Pleural Effusion",
    "Pleural Other":              "Frontal CXR showing Pleural Other",
    "Fracture":                   "Frontal CXR showing Fracture",
    "Support Devices":            "Frontal CXR showing Support Devices",
}

ALL_PATHOLOGIES = list(PATHOLOGY_PROMPTS.keys())


def _load_weights(path: Path, model: torch.nn.Module, dtype) -> torch.nn.Module:
    sd = {k: v.to(dtype) for k, v in load_file(path).items()}
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


def _tokenize(tokenizer, text: str, max_length: int, device) -> torch.Tensor:
    return tokenizer(
        text, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)


@torch.no_grad()
def encode_prompt(prompt: str, device, dtype):
    """Encode a single prompt through CLIP-L, CLIP-G, and T5-XXL."""
    print("  [1/3] CLIP-L ...")
    config_l = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
    clip_l   = _load_weights(
        ENC_DIR / "clip_l.safetensors", CLIPTextModel(config_l).to(dtype), dtype
    ).to(device).eval()
    tok_l    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    out_l    = clip_l(_tokenize(tok_l, prompt, CLIP_MAX, device), output_hidden_states=True)
    clip_l_h = out_l.hidden_states[-2]
    clip_l_p = out_l.pooler_output
    del clip_l; torch.cuda.empty_cache()

    print("  [2/3] CLIP-G ...")
    config_g = CLIPTextConfig.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2")
    clip_g   = _load_weights(
        ENC_DIR / "clip_g.safetensors", CLIPTextModelWithProjection(config_g).to(dtype), dtype
    ).to(device).eval()
    tok_g    = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    out_g    = clip_g(_tokenize(tok_g, prompt, CLIP_MAX, device), output_hidden_states=True)
    clip_g_h = out_g.hidden_states[-2]
    clip_g_p = out_g.text_embeds
    del clip_g; torch.cuda.empty_cache()

    print("  [3/3] T5-XXL ...")
    config_t5 = T5Config.from_pretrained("google/t5-v1_1-xxl")
    t5        = _load_weights(
        ENC_DIR / "t5xxl.safetensors", T5EncoderModel(config_t5).to(dtype), dtype
    ).to(device).eval()
    tok_t5    = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    t5_h      = t5(_tokenize(tok_t5, prompt, T5_MAX, device)).last_hidden_state
    del t5; torch.cuda.empty_cache()

    clip_h        = torch.cat([clip_l_h, clip_g_h], dim=-1)
    clip_h        = F.pad(clip_h, (0, T5_DIM - clip_h.shape[-1]))
    prompt_embeds = torch.cat([clip_h, t5_h], dim=1)
    pooled_embeds = torch.cat([clip_l_p, clip_g_p], dim=-1)
    return prompt_embeds, pooled_embeds


@torch.no_grad()
def generate_one(
    pe: torch.Tensor,
    ppe: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    device,
    dtype,
    transformer,
    scheduler,
    vae,
) -> np.ndarray:
    sched = type(scheduler)(num_train_timesteps=1000, shift=3.0)
    sched.set_timesteps(num_steps, device=device)
    sched.sigmas = torch.cat([sched.sigmas, sched.sigmas.new_zeros(1)])

    gen       = torch.Generator(device=device).manual_seed(seed)
    latent_c  = transformer.config.in_channels
    latent_hw = IMG_SIZE // 8
    xt = torch.randn(1, latent_c, latent_hw, latent_hw,
                     device=device, dtype=dtype, generator=gen)

    t_in_1 = torch.zeros(1, device=device, dtype=dtype)
    for t in sched.timesteps:
        t_in_1[0] = t
        if guidance_scale > 1.0:
            v_uncond = transformer(
                hidden_states=xt, timestep=t_in_1,
                encoder_hidden_states=pe[:1], pooled_projections=ppe[:1],
                return_dict=False,
            )[0]
            v_cond = transformer(
                hidden_states=xt, timestep=t_in_1,
                encoder_hidden_states=pe[1:], pooled_projections=ppe[1:],
                return_dict=False,
            )[0]
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = transformer(
                hidden_states=xt, timestep=t_in_1,
                encoder_hidden_states=pe, pooled_projections=ppe,
                return_dict=False,
            )[0]
        xt = sched.step(v, t, xt).prev_sample

    torch.cuda.empty_cache()

    vae_was_cpu = next(vae.parameters()).device.type == "cpu"
    if vae_was_cpu:
        vae.to(device)

    vae_input  = xt / SCALING_FACTOR + SHIFT_FACTOR
    img_tensor = vae.decode(vae_input.to(torch.float32)).sample

    if vae_was_cpu:
        vae.cpu()
        torch.cuda.empty_cache()

    img = img_tensor.squeeze(0).float().clamp(-1, 1)
    img = ((img + 1) / 2 * 255).byte().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img.mean(axis=2).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="BASEMODEL CXR generation — SD3.5 + LoRA")
    parser.add_argument("--pathology", nargs="+", default=ALL_PATHOLOGIES,
                        help="Pathologies to generate (default: all)")
    parser.add_argument("--prompt",    type=str, default=None,
                        help="Custom prompt (overrides built-in for each --pathology)")
    parser.add_argument("--cfg",   nargs="+", type=float, default=[7.0, 4.5])
    parser.add_argument("--seeds", type=int, default=4,
                        help="Number of seeds per (pathology, cfg) pair")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--out",   default=str(OUT_DIR))
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    local_rank = 0
    world_size = 1
    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    if local_rank == 0:
        print(f"Device: {device}  |  world_size: {world_size}\n")

    if not LORA_DIR.exists():
        raise SystemExit(f"No LoRA checkpoint found at {LORA_DIR}\nRun 4-lora_training/run.py first.")

    prompt_override = args.prompt
    pathologies = []
    for p in args.pathology:
        if prompt_override or p in PATHOLOGY_PROMPTS:
            pathologies.append(p)
            if prompt_override:
                PATHOLOGY_PROMPTS[p] = prompt_override
        else:
            print(f"WARNING: '{p}' not in PATHOLOGY_PROMPTS and no --prompt given — skipping.")
    need_cfg = any(c > 1.0 for c in args.cfg)

    if local_rank == 0:
        print("Encoding prompts ...")

    null_pe = null_ppe = None
    if need_cfg:
        if local_rank == 0:
            print("  [null prompt for CFG negative]")
            null_pe, null_ppe = encode_prompt("", device, dtype)
        else:
            null_pe  = torch.empty(1, SEQ_LEN, T5_DIM, device=device, dtype=dtype)
            null_ppe = torch.empty(1, 2048,         device=device, dtype=dtype)
        if args.deepspeed:
            dist.broadcast(null_pe,  src=0)
            dist.broadcast(null_ppe, src=0)
        null_pe  = null_pe.cpu()
        null_ppe = null_ppe.cpu()

    embed_cache: dict = {}
    for pathology in pathologies:
        if local_rank == 0:
            print(f"  [{pathology}] — \"{PATHOLOGY_PROMPTS[pathology]}\"")
            pe, ppe = encode_prompt(PATHOLOGY_PROMPTS[pathology], device, dtype)
        else:
            pe  = torch.empty(1, SEQ_LEN, T5_DIM, device=device, dtype=dtype)
            ppe = torch.empty(1, 2048,         device=device, dtype=dtype)
        if args.deepspeed:
            dist.broadcast(pe,  src=0)
            dist.broadcast(ppe, src=0)
        embed_cache[pathology] = (pe.cpu(), ppe.cpu())

    if local_rank == 0:
        print("\nLoading transformer + LoRA ...")

    transformer = SD3Transformer2DModel.from_single_file(str(MODEL_PATH))
    transformer = transformer.to(dtype)
    gc.collect()
    transformer = transformer.to(device)
    torch.cuda.empty_cache()

    if args.deepspeed:
        import deepspeed
        peft_model  = PeftModel.from_pretrained(transformer, str(LORA_DIR))
        transformer = peft_model.merge_and_unload()
        del peft_model; gc.collect(); torch.cuda.empty_cache()
        transformer = transformer.to(dtype); gc.collect(); torch.cuda.empty_cache()
        transformer = deepspeed.init_inference(
            transformer, mp_size=world_size, dtype=dtype, replace_with_kernel_inject=False,
        ).module
    else:
        transformer = PeftModel.from_pretrained(transformer, str(LORA_DIR)).to(dtype)
        gc.collect(); torch.cuda.empty_cache()

    transformer.eval()

    if local_rank == 0:
        print("Loading VAE (CPU — moved to GPU only for decode) ...")
    vae = AutoencoderKL.from_single_file(
        str(MODEL_PATH),
        config="stabilityai/stable-diffusion-3.5-medium",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval()

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

    out_dir = Path(args.out)
    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for pathology in pathologies:
        prompt  = PATHOLOGY_PROMPTS[pathology]
        slug    = pathology.lower().replace(" ", "_")
        pos_pe, pos_ppe = embed_cache[pathology]

        for cfg in args.cfg:
            if cfg > 1.0:
                pe  = torch.cat([null_pe, pos_pe]).to(device=device, dtype=dtype)
                ppe = torch.cat([null_ppe, pos_ppe]).to(device=device, dtype=dtype)
            else:
                pe  = pos_pe.to(device=device, dtype=dtype)
                ppe = pos_ppe.to(device=device, dtype=dtype)

            for seed_idx in range(args.seeds):
                seed     = abs(hash(f"{slug}_{cfg}_{seed_idx}")) % (2**31)
                image_id = f"{slug}_cfg{cfg:.1f}_{seed_idx:04d}"

                if local_rank == 0:
                    print(f"\n[{pathology}]  cfg={cfg}  seed={seed}")

                img_gray = generate_one(
                    pe=pe, ppe=ppe,
                    num_steps=args.steps,
                    guidance_scale=cfg,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    transformer=transformer,
                    scheduler=scheduler,
                    vae=vae,
                )

                if local_rank == 0:
                    out_path = out_dir / f"{image_id}.png"
                    Image.fromarray(img_gray, mode="L").save(out_path)
                    print(f"  -> {out_path}")
                    csv_rows.append({
                        "image_id":  image_id,
                        "pathology": pathology,
                        "cfg":       cfg,
                        "seed":      seed,
                        "caption":   prompt,
                    })

    if local_rank == 0:
        csv_path = GEN_CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "pathology", "cfg", "seed", "caption"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nGenerated {len(csv_rows)} images -> {out_dir}")
        print(f"Log       -> {csv_path}")


if __name__ == "__main__":
    main()