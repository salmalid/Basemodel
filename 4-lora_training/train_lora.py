#!/usr/bin/env python3
import copy
import csv
import datetime
import json
import math
import os
import random
import sys

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, ProjectConfiguration
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig, get_peft_model

from dataset import CXRDataset

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "sd3.5_medium.safetensors"
CKPT_DIR   = ROOT / "checkpoints"
RESUME_DIR = CKPT_DIR / "resume"
LOG_DIR    = ROOT / "logs"
VAL_DIR    = ROOT / "validation"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

RANK           = 64
LORA_ALPHA     = 64
LORA_DROPOUT   = 0.1
LR             = 1e-4
WARMUP_RATIO   = 0.05
BATCH_SIZE     = 1
GRAD_ACCUM     = 16
NUM_EPOCHS     = 50
PATIENCE       = 5
IMG_SIZE       = 1024

SCALING_FACTOR = 1.5305
SHIFT_FACTOR   = 0.0609

CFG_DROPOUT_PROB = 0.1

VAL_EVERY_EPOCHS = 5
VAL_STEPS        = 20
VAL_GUIDANCE     = 7.0
VAL_SEED         = 0
NUM_VAL_SAMPLES  = 3

CKPT_EVERY_STEPS = 50

# ── wandb config ──────────────────────────────────────────────────────────────
WANDB_ENTITY  = "salma-lidame-university-of-technology-belfort-montbeliard"
WANDB_PROJECT = "basemodel-cxr-sd35"



def get_sigmas(timesteps, sched, device, n_dim: int = 4):
    sigmas   = sched.sigmas.to(device=device, dtype=torch.float32)
    sched_ts = sched.timesteps.to(device)
    idx      = [(sched_ts == t).nonzero(as_tuple=False)[0, 0].item() for t in timesteps]
    sigma    = sigmas[idx].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(transformer, vae, val_records, sched_ref, accelerator, epoch, g_step, wb_run):
    if not accelerator.is_main_process:
        return

    transformer.eval()
    device = accelerator.device
    dtype  = torch.float16

    out_dir = VAL_DIR / f"epoch_{epoch:03d}_step_{g_step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_sched = copy.deepcopy(sched_ref)
    infer_sched.set_timesteps(VAL_STEPS, device=device)
    infer_sched.sigmas = torch.cat([infer_sched.sigmas, infer_sched.sigmas.new_zeros(1)])

    vae.to(device)
    wandb_images = []

    for i, rec in enumerate(val_records):
        text = torch.load(rec["text_path"], weights_only=True)
        pe   = text["prompt_embeds"].unsqueeze(0).to(device, dtype)
        ppe  = text["pooled_prompt_embeds"].unsqueeze(0).to(device, dtype)

        neg_pe  = torch.zeros_like(pe)
        neg_ppe = torch.zeros_like(ppe)
        pe_in   = torch.cat([neg_pe, pe])
        ppe_in  = torch.cat([neg_ppe, ppe])

        latent_hw = IMG_SIZE // 8
        latent_c  = accelerator.unwrap_model(transformer).config.in_channels
        gen = torch.Generator(device=device).manual_seed(VAL_SEED + i)
        xt  = torch.randn(1, latent_c, latent_hw, latent_hw,
                          device=device, dtype=dtype, generator=gen)

        for t in infer_sched.timesteps:
            model_in = torch.cat([xt, xt])
            t_in     = t.expand(model_in.shape[0])
            v = accelerator.unwrap_model(transformer)(
                hidden_states=model_in, timestep=t_in,
                encoder_hidden_states=pe_in, pooled_projections=ppe_in,
                return_dict=False,
            )[0]
            v_uncond, v_cond = v.chunk(2)
            v  = v_uncond + VAL_GUIDANCE * (v_cond - v_uncond)
            xt = infer_sched.step(v, t, xt).prev_sample

        vae_input  = xt / SCALING_FACTOR + SHIFT_FACTOR
        img_tensor = vae.decode(vae_input.to(torch.float32)).sample

        img      = img_tensor.squeeze(0).float().clamp(-1, 1)
        img      = ((img + 1) / 2 * 255).byte().cpu().numpy()
        img      = np.transpose(img, (1, 2, 0))
        img_gray = img.mean(axis=2).astype(np.uint8)

        img_path = out_dir / f"sample_{i}.png"
        Image.fromarray(img_gray, mode="L").save(img_path)
        (out_dir / f"sample_{i}_caption.txt").write_text(rec["caption"][:200], encoding="utf-8")

        caption_short = rec["caption"][:80]
        wandb_images.append(wandb.Image(str(img_path), caption=caption_short))

    if wb_run is not None and wandb_images:
        wb_run.log({"val/samples": wandb_images, "val/epoch": epoch, "val/g_step": g_step})

    vae.cpu()
    torch.cuda.empty_cache()
    transformer.train()
    accelerator.print(f"  Validation images -> {out_dir}")



def _save_resume(accelerator, meta_path, *, epoch, epoch_done, g_step, best_loss, no_improve, wandb_run_id):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        RESUME_DIR.mkdir(exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "epoch":        epoch,
                "epoch_done":   epoch_done,
                "g_step":       g_step,
                "best_loss":    best_loss,
                "no_improve":   no_improve,
                "wandb_run_id": wandb_run_id,
            }, f)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(RESUME_DIR))



def main():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    on_linux     = sys.platform == "linux"
    dist_backend = "nccl" if on_linux else "gloo"

    ds_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_clipping=1.0,
    )
    if not on_linux:
        ds_plugin.deepspeed_config["zero_optimization"].update({
            "reduce_scatter": False,
            "contiguous_gradients": False,
        })

    pg_kwargs = InitProcessGroupKwargs(
        backend=dist_backend, timeout=datetime.timedelta(seconds=1800)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM,
        mixed_precision="fp16",
        project_config=ProjectConfiguration(project_dir=str(CKPT_DIR)),
        kwargs_handlers=[pg_kwargs],
        deepspeed_plugin=ds_plugin,
    )
    device = accelerator.device

    dataset = CXRDataset()
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=True)

    rng         = random.Random(42)
    val_records = rng.sample(dataset.records, min(NUM_VAL_SAMPLES, len(dataset.records)))

    steps_per_epoch   = math.ceil(len(dataset) / (BATCH_SIZE * accelerator.num_processes))
    total_optim_steps = math.ceil(steps_per_epoch / GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps      = max(1, int(total_optim_steps * WARMUP_RATIO))

    accelerator.print(
        f"\nDataset  : {len(dataset):,} samples\n"
        f"GPUs     : {accelerator.num_processes}\n"
        f"Batch    : {BATCH_SIZE}/GPU × {accelerator.num_processes} GPUs "
        f"× {GRAD_ACCUM} accum = {BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM} effective\n"
        f"Steps    : {steps_per_epoch}/epoch  |  total optimizer steps: {total_optim_steps:,}\n"
        f"Warmup   : {warmup_steps} steps ({WARMUP_RATIO:.0%})\n"
        f"LR       : {LR}  (constant after warmup)\n"
        f"LoRA     : rank={RANK}  alpha={LORA_ALPHA}\n"
        f"CFG drop : {CFG_DROPOUT_PROB:.0%}   weighting: logit_normal  shift=3.0\n"
        f"Val      : every {VAL_EVERY_EPOCHS} epochs  ({NUM_VAL_SAMPLES} images, {VAL_STEPS} steps)\n"
    )

    accelerator.print(f"Loading transformer from {MODEL_PATH.name} ...")
    transformer = SD3Transformer2DModel.from_single_file(
        str(MODEL_PATH), torch_dtype=torch.float16
    )
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    from functools import partial
    transformer._gradient_checkpointing_func = partial(
        torch.utils.checkpoint.checkpoint, use_reentrant=False
    )
    transformer.enable_xformers_memory_efficient_attention()

    lora_cfg = LoraConfig(
        r=RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        init_lora_weights="gaussian",
    )
    transformer = get_peft_model(transformer, lora_cfg)

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in transformer.parameters())
    accelerator.print(f"LoRA params : {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")

    if accelerator.is_main_process:
        accelerator.print("Loading VAE (CPU, only moved to GPU during validation) ...")
        vae = AutoencoderKL.from_single_file(
            str(MODEL_PATH),
            config="stabilityai/stable-diffusion-3.5-medium",
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        vae.requires_grad_(False)
        vae.eval()
    else:
        vae = None

    noise_sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    noise_sched.set_timesteps(1000)
    sched_ref   = copy.deepcopy(noise_sched)

    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(
        lora_params, lr=LR, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    transformer, optimizer, loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, loader, lr_scheduler
    )

    resume_meta_path = RESUME_DIR / "training_meta.json"
    resume_meta      = None
    if RESUME_DIR.exists() and resume_meta_path.exists():
        with open(resume_meta_path) as f:
            resume_meta = json.load(f)
        accelerator.print(
            f"\nResume checkpoint found — restarting after epoch {resume_meta['epoch']} "
            f"(g_step={resume_meta['g_step']}, best_loss={resume_meta['best_loss']:.5f})"
        )
        accelerator.load_state(str(RESUME_DIR))

    if resume_meta:
        epoch_done     = resume_meta.get("epoch_done", True)
        start_epoch    = resume_meta["epoch"] + (1 if epoch_done else 0)
        g_step         = resume_meta["g_step"]
        best_loss      = resume_meta["best_loss"]
        no_improve     = resume_meta["no_improve"]
        prev_wandb_id  = resume_meta.get("wandb_run_id")
        if not epoch_done:
            accelerator.print(
                f"  (mid-epoch checkpoint at step {g_step} — epoch {start_epoch} will restart from batch 0)"
            )
    else:
        start_epoch   = 1
        g_step        = 0
        best_loss     = float("inf")
        no_improve    = 0
        prev_wandb_id = None

    wb_run = None
    if accelerator.is_main_process:
        wb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            id=prev_wandb_id,      # resumes the same run on restart
            resume="allow",
            config={
                "rank":             RANK,
                "lora_alpha":       LORA_ALPHA,
                "lora_dropout":     LORA_DROPOUT,
                "lr":               LR,
                "warmup_ratio":     WARMUP_RATIO,
                "batch_size":       BATCH_SIZE,
                "grad_accum":       GRAD_ACCUM,
                "effective_batch":  BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM,
                "num_epochs":       NUM_EPOCHS,
                "patience":         PATIENCE,
                "cfg_dropout_prob": CFG_DROPOUT_PROB,
                "img_size":         IMG_SIZE,
                "lora_target_mods": 8,
                "caption_style":    "Frontal CXR showing {pathology}",
                "dataset_size":     len(dataset),
                "num_gpus":         accelerator.num_processes,
            },
        )
        accelerator.print(f"wandb run: {wb_run.name}  ({wb_run.url})")

    wandb_run_id = wb_run.id if wb_run is not None else None

    log_path = LOG_DIR / "train_log.csv"
    if accelerator.is_main_process and not resume_meta:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "lr", "g_step", "best"])

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        transformer.train()
        epoch_loss  = 0.0
        epoch_steps = 0

        bar = tqdm(loader, desc=f"Ep {epoch:>3}/{NUM_EPOCHS}",
                   disable=not accelerator.is_main_process, leave=False)

        for batch in bar:
            with accelerator.accumulate(transformer):
                x0  = batch["latents"].to(device, dtype=torch.float16)
                pe  = batch["prompt_embeds"].to(device, dtype=torch.float16)
                ppe = batch["pooled_prompt_embeds"].to(device, dtype=torch.float16)
                B   = x0.shape[0]

                if CFG_DROPOUT_PROB > 0:
                    mask = torch.rand(B, device=device) < CFG_DROPOUT_PROB
                    pe[mask]  = 0.0
                    ppe[mask] = 0.0

                noise = torch.randn_like(x0)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=B,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * sched_ref.config.num_train_timesteps).long().clamp(0, 999)
                ts      = sched_ref.timesteps[indices].to(device)
                sigmas  = get_sigmas(ts, sched_ref, device, n_dim=x0.ndim).to(dtype=x0.dtype)

                xt = (1.0 - sigmas) * x0 + sigmas * noise

                v_pred = transformer(
                    hidden_states=xt, timestep=ts,
                    encoder_hidden_states=pe, pooled_projections=ppe,
                    return_dict=False,
                )[0]

                pred_x0 = v_pred * (-sigmas) + xt
                w       = compute_loss_weighting_for_sd3(
                    weighting_scheme="logit_normal", sigmas=sigmas
                )
                loss = torch.mean(
                    (w.float() * (pred_x0.float() - x0.float()) ** 2).reshape(B, -1), 1
                ).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    g_step   += 1
                    _mid_ckpt = (g_step % CKPT_EVERY_STEPS == 0)

                    # Log per-step metrics to wandb
                    if wb_run is not None:
                        wb_run.log({
                            "train/loss":    loss.item(),
                            "train/lr":      optimizer.param_groups[0]["lr"],
                            "train/g_step":  g_step,
                            "train/epoch":   epoch,
                        }, step=g_step)
                else:
                    _mid_ckpt = False

            if _mid_ckpt:
                _save_resume(accelerator, resume_meta_path,
                             epoch=epoch, epoch_done=False,
                             g_step=g_step, best_loss=best_loss,
                             no_improve=no_improve, wandb_run_id=wandb_run_id)
                accelerator.print(f"  [step {g_step}] mid-epoch checkpoint saved")

            epoch_loss  += loss.detach().item()
            epoch_steps += 1
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_t   = torch.tensor(epoch_loss / max(epoch_steps, 1), device=device)
        avg     = accelerator.reduce(avg_t, reduction="mean").item()
        lr_now  = optimizer.param_groups[0]["lr"]
        is_best = avg < best_loss

        accelerator.print(
            f"Epoch {epoch:>3}/{NUM_EPOCHS}  loss={avg:.5f}  lr={lr_now:.2e}  step={g_step}"
            + ("  <- best" if is_best else f"  (no improve {no_improve + 1}/{PATIENCE})")
        )

        if is_best:
            best_loss  = avg
            no_improve = 0
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.unwrap_model(transformer).save_pretrained(str(CKPT_DIR / "best"))
                accelerator.print(f"  -> saved  checkpoints/best")
        else:
            no_improve += 1

        # Log per-epoch metrics to wandb
        if wb_run is not None:
            wb_run.log({
                "epoch/avg_loss": avg,
                "epoch/lr":       lr_now,
                "epoch/best":     int(is_best),
                "epoch/best_loss_so_far": best_loss,
                "epoch/no_improve": no_improve,
                "epoch/epoch":    epoch,
            }, step=g_step)

        if accelerator.is_main_process:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{avg:.6f}", f"{lr_now:.2e}", g_step, int(is_best)])

        _save_resume(accelerator, resume_meta_path,
                     epoch=epoch, epoch_done=True,
                     g_step=g_step, best_loss=best_loss,
                     no_improve=no_improve, wandb_run_id=wandb_run_id)
        accelerator.print(f"  -> epoch checkpoint saved  (checkpoints/resume)")

        if epoch % VAL_EVERY_EPOCHS == 0:
            accelerator.wait_for_everyone()
            run_validation(transformer, vae, val_records, sched_ref,
                           accelerator, epoch, g_step, wb_run)
            accelerator.wait_for_everyone()

        if no_improve >= PATIENCE:
            accelerator.print(f"\nEarly stopping: no improvement for {PATIENCE} consecutive epochs.")
            break

    accelerator.wait_for_everyone()
    run_validation(transformer, vae, val_records, sched_ref, accelerator, epoch, g_step, wb_run)

    if accelerator.is_main_process:
        accelerator.print(f"\nDone. Best loss={best_loss:.5f}  ->  checkpoints/best")
        if wb_run is not None:
            wb_run.summary["best_loss"] = best_loss
            wb_run.finish()


if __name__ == "__main__":
    main()