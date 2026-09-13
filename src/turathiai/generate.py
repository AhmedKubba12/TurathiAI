"""Generate images from a trained TurathiAI LoRA (or the base SDXL model).

Loads SDXL, optionally applies a LoRA checkpoint, and generates images for a set
of prompts x seeds using the paper's inference settings (25 steps, guidance 6.0,
1024x1024, DPMSolverMultistep).

Example
-------
    python -m turathiai.generate \
        --lora outputs/lora_default_best \
        --prompts data/test_prompts.txt \
        --out generations/turathiai --seeds 42 43 44 45
"""
from __future__ import annotations

import argparse
import os

import torch


def load_pipeline(base_model: str, vae_model: str, lora_ckpt: str | None,
                  device: str, dtype):
    from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                           StableDiffusionXLPipeline)

    vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, vae=vae, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if lora_ckpt:
        pipe.load_lora_weights(lora_ckpt)
    return pipe


def read_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def generate(pipe, prompts, seeds, out_dir, steps=25, guidance=6.0,
             resolution=1024, device="cuda") -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for pi, prompt in enumerate(prompts):
        for si, sd in enumerate(seeds):
            fp = os.path.join(out_dir, f"{pi:03d}_{si}.png")
            if not os.path.exists(fp):
                g = torch.Generator(device=device).manual_seed(sd)
                img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                           generator=g, height=resolution, width=resolution).images[0]
                img.save(fp)
            records.append({"path": fp, "prompt": prompt})
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate images with TurathiAI / base SDXL.")
    ap.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--vae-model", default="madebyollin/sdxl-vae-fp16-fix")
    ap.add_argument("--lora", default=None, help="LoRA checkpoint dir (omit for base model).")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = load_pipeline(args.base_model, args.vae_model, args.lora, device, dtype)
    prompts = read_prompts(args.prompts)
    recs = generate(pipe, prompts, args.seeds, args.out, args.steps, args.guidance,
                    args.resolution, device)
    print(f"Generated {len(recs)} images -> {args.out}")


if __name__ == "__main__":
    main()
