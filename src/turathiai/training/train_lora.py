"""LoRA fine-tuning of SDXL on the heritage dataset (Section 3.2).

Implements the adaptation pipeline used for the main results and the ablation
study: SDXL UNet + LoRA adapters, masked denoising loss, min-SNR loss weighting,
Prodigy / AdamW optimizers, and validation-loss monitoring for best-checkpoint
selection.

The validation loss is a fixed-condition denoising error computed on the held-out
split at fixed timesteps and seeds (Section 3.2), giving a clean, comparable
curve whose minima mark the best checkpoints.

Example
-------
    accelerate launch -m turathiai.training.train_lora --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader

from ..config import Config, load_config
from ..data.dataset import HeritageTriplesDataset, index_triples, train_val_split
from .losses import masked_mse_loss, min_snr_weights
from .optimizers import build_optimizer


def _fixed_val_timesteps(n: int, num_train_timesteps: int, seed: int = 0) -> torch.Tensor:
    """Deterministic, evenly spread timesteps so val loss is repeatable."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_train_timesteps, (n,), generator=g)


def train(cfg: Config) -> str:
    from accelerate import Accelerator
    from diffusers import (AutoencoderKL, DDPMScheduler,
                           StableDiffusionXLPipeline, UNet2DConditionModel)
    from peft import LoraConfig, get_peft_model_state_dict

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
    )
    device = accelerator.device
    weight_dtype = torch.bfloat16 if cfg.train.mixed_precision == "bf16" else torch.float16

    # --- components ---
    tok1 = _tokenizer(cfg.model.base_model, "tokenizer")
    tok2 = _tokenizer(cfg.model.base_model, "tokenizer_2")
    te1 = _text_encoder(cfg.model.base_model, "text_encoder").to(device, weight_dtype)
    te2 = _text_encoder(cfg.model.base_model, "text_encoder_2", second=True).to(device, weight_dtype)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae_model).to(device, torch.float32)
    unet = UNet2DConditionModel.from_pretrained(cfg.model.base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    te1.requires_grad_(False)
    te2.requires_grad_(False)
    unet.requires_grad_(False)

    # --- attach LoRA to the UNet attention projections ---
    lora_cfg = LoraConfig(
        r=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(cfg.model.target_modules),
    )
    unet.add_adapter(lora_cfg)
    unet.to(device)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    accelerator.print(f"Trainable LoRA params: {n_params/1e6:.2f}M (rank {cfg.model.lora_rank})")

    # --- data ---
    triples = index_triples(cfg.data.dataset_dir, cfg.data.mask_suffix,
                            cfg.data.caption_ext, cfg.data.image_exts)
    train_t, val_t = train_val_split(triples, cfg.data.val_split, cfg.data.seed)
    accelerator.print(f"Dataset: {len(triples)} triples -> {len(train_t)} train / {len(val_t)} val")

    train_ds = HeritageTriplesDataset(train_t, cfg.data.resolution, cfg.data.random_flip,
                                      return_mask=cfg.data.use_masked_loss)
    val_ds = HeritageTriplesDataset(val_t, cfg.data.resolution, random_flip=False,
                                    return_mask=cfg.data.use_masked_loss)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.train_batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=2, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.train_batch_size, shuffle=False,
                        collate_fn=_collate, num_workers=2)

    optimizer = build_optimizer(cfg.train.optimizer, trainable, cfg.train.learning_rate)
    lr_scheduler = _lr_scheduler(cfg, optimizer)

    unet, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dl, lr_scheduler)

    encode_prompt = _make_prompt_encoder(tok1, tok2, te1, te2, device, weight_dtype)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    best_val = math.inf
    global_step = 0
    unet.train()

    while global_step < cfg.train.max_train_steps:
        for batch in train_dl:
            with accelerator.accumulate(unet):
                loss = _step(batch, vae, unet, noise_scheduler, encode_prompt,
                             cfg, device, weight_dtype)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % cfg.train.validation_steps == 0:
                    val = _validate(val_dl, vae, unet, noise_scheduler, encode_prompt,
                                    cfg, device, weight_dtype)
                    accelerator.print(f"step {global_step}: train {loss.item():.4f} | val {val:.4f}")
                    if accelerator.is_main_process and val < best_val:
                        best_val = val
                        _save_lora(accelerator, unet, cfg.train.output_dir,
                                   StableDiffusionXLPipeline, get_peft_model_state_dict)
                        accelerator.print(f"  new best val {best_val:.4f} -> saved")

                if global_step >= cfg.train.max_train_steps:
                    break

    accelerator.print(f"Done. Best validation loss: {best_val:.4f}")
    _write_val_record(cfg, best_val)
    return cfg.train.output_dir


def _step(batch, vae, unet, noise_scheduler, encode_prompt, cfg, device, weight_dtype):
    pixels = batch["pixel_values"].to(device, torch.float32)
    with torch.no_grad():
        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
    latents = latents.to(weight_dtype)

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
    noisy = noise_scheduler.add_noise(latents, noise, timesteps)

    prompt_embeds, pooled, add_time_ids = encode_prompt(batch["caption"], cfg.data.resolution, bsz)
    model_pred = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds,
                      added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample

    target = noise  # epsilon prediction
    weights = None
    if cfg.train.snr_gamma is not None:
        weights = min_snr_weights(noise_scheduler, timesteps, cfg.train.snr_gamma)
    mask = batch.get("mask")
    if mask is not None:
        mask = mask.to(device)
    return masked_mse_loss(model_pred, target, mask=mask, weights=weights)


@torch.no_grad()
def _validate(val_dl, vae, unet, noise_scheduler, encode_prompt, cfg, device, weight_dtype):
    unet.eval()
    losses = []
    for i, batch in enumerate(val_dl):
        pixels = batch["pixel_values"].to(device, torch.float32)
        latents = vae.encode(pixels).latent_dist.mean * vae.config.scaling_factor
        latents = latents.to(weight_dtype)
        bsz = latents.shape[0]
        # fixed timesteps/seed -> repeatable, comparable val curve
        g = torch.Generator(device=device).manual_seed(cfg.data.seed + i)
        noise = torch.randn(latents.shape, generator=g, device=device, dtype=latents.dtype)
        timesteps = _fixed_val_timesteps(bsz, noise_scheduler.config.num_train_timesteps,
                                         seed=cfg.data.seed + i).to(device)
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        prompt_embeds, pooled, add_time_ids = encode_prompt(batch["caption"], cfg.data.resolution, bsz)
        pred = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        weights = None
        if cfg.train.snr_gamma is not None:
            weights = min_snr_weights(noise_scheduler, timesteps, cfg.train.snr_gamma)
        losses.append(masked_mse_loss(pred, noise, mask=mask, weights=weights).item())
    unet.train()
    return float(sum(losses) / max(len(losses), 1))


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _tokenizer(base_model, subfolder):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_model, subfolder=subfolder, use_fast=False)


def _text_encoder(base_model, subfolder, second=False):
    if second:
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection.from_pretrained(base_model, subfolder=subfolder)
    from transformers import CLIPTextModel
    return CLIPTextModel.from_pretrained(base_model, subfolder=subfolder)


def _make_prompt_encoder(tok1, tok2, te1, te2, device, weight_dtype):
    """SDXL dual text-encoder prompt encoding, returning the three UNet inputs."""
    def encode(captions, resolution, bsz):
        embeds_list, pooled = [], None
        for tok, te in ((tok1, te1), (tok2, te2)):
            ids = tok(list(captions), padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(device)
            out = te(ids, output_hidden_states=True)
            pooled = out[0]                     # last encoder's pooled output
            embeds_list.append(out.hidden_states[-2])
        prompt_embeds = torch.cat(embeds_list, dim=-1).to(weight_dtype)
        pooled = pooled.to(weight_dtype)
        # SDXL micro-conditioning: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
        add_time_ids = torch.tensor(
            [resolution, resolution, 0, 0, resolution, resolution],
            device=device, dtype=weight_dtype).repeat(bsz, 1)
        return prompt_embeds, pooled, add_time_ids
    return encode


def _collate(batch):
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "sample": [b["sample"] for b in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([b["mask"] for b in batch])
    return out


def _lr_scheduler(cfg, optimizer):
    from diffusers.optimization import get_scheduler
    return get_scheduler(
        cfg.train.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps,
        num_training_steps=cfg.train.max_train_steps,
    )


def _save_lora(accelerator, unet, output_dir, PipelineClass, get_peft_state):
    unwrapped = accelerator.unwrap_model(unet)
    state = get_peft_state(unwrapped)
    PipelineClass.save_lora_weights(save_directory=output_dir, unet_lora_layers=state,
                                    safe_serialization=True)


def _write_val_record(cfg, best_val):
    import csv
    path = os.path.join(os.path.dirname(cfg.train.output_dir.rstrip("/")), "ablation_val.csv")
    ablation = os.path.basename(cfg.train.output_dir).replace("lora_", "").replace("_best", "")
    rows = {}
    if os.path.exists(path):
        with open(path) as fh:
            rows = {r["ablation"]: r["best_val"] for r in csv.DictReader(fh)}
    rows[ablation] = f"{best_val:.4f}"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ablation", "best_val"])
        for k, v in rows.items():
            w.writerow([k, v])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a TurathiAI LoRA on SDXL.")
    ap.add_argument("--config", required=True, help="Path to a YAML config.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
