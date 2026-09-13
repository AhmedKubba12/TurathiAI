"""End-to-end evaluation across all checkpoints (Sections 4.1, 4.3).

Discovers every ``lora_*_best`` checkpoint in an output directory, generates the
shared test set for each (plus the base SDXL zero-shot reference), scores
CLIP / FID / LPIPS / SSIM per model, then runs the omnibus ANOVA + Friedman and
the Holm-corrected pairwise tests over the per-image CLIP scores.

Outputs (written to ``--output-dir``):

* ``ablation_iqa.csv``       per-model CLIP / FID / LPIPS / SSIM (Table 26)
* ``clip_per_image_all.csv`` aligned per-image CLIP for every model
* ``anova_omnibus.json``     F, p, eta-squared (+ Friedman chi2, Kendall's W)
* ``pairwise_stats.csv``     every pair: t, p_raw, p_holm, Cohen's d

This is the scripted counterpart of ``notebooks/TurathiAI_Ablation_IQA_and_ANOVA``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from ..generate import generate, load_pipeline, read_prompts
from .metrics import clip_scores, fid_against_real, lpips_ssim_vs_base
from .stats_tests import omnibus, pairwise_holm


def find_checkpoints(output_dir: str) -> dict[str, str]:
    models: dict[str, str] = {}
    for d in sorted(glob.glob(os.path.join(output_dir, "lora_*_best"))):
        name = os.path.basename(d)[len("lora_"):-len("_best")]
        models["baseline" if name == "default" else name] = d
    return models


def find_real_images(root: str) -> list[str]:
    exts = ("jpg", "jpeg", "JPG", "JPEG", "png", "PNG")
    paths: list[str] = []
    for e in exts:
        paths += glob.glob(os.path.join(root, "**", f"*.{e}"), recursive=True)
    paths = [p for p in paths if "-masklabel" not in os.path.basename(p).lower()]
    return sorted(set(paths))


def run(args) -> None:
    import pandas as pd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    prompts = read_prompts(args.prompts)
    if args.quick:
        prompts = prompts[:25]
    seeds = args.seeds[:1] if args.quick else args.seeds

    real_paths = find_real_images(args.dataset_dir)
    print(f"Real reference images: {len(real_paths)}")
    models = find_checkpoints(args.output_dir)
    print(f"Checkpoints: {list(models)}")

    gen_root = os.path.join(args.output_dir, "eval_gen")

    # base (zero-shot) reference for LPIPS/SSIM
    pipe = load_pipeline(args.base_model, args.vae_model, None, device, dtype)
    base = generate(pipe, prompts, seeds, os.path.join(gen_root, "base"),
                    args.steps, args.guidance, args.resolution, device)
    clip_base = clip_scores(base, device)
    fid_base = fid_against_real(base, real_paths, device)

    per_image = {"base": clip_base}
    rows = [{"model": "base (zero-shot)", "CLIP_mean": clip_base.mean(), "CLIP_sd": clip_base.std(),
             "FID_vs_real": fid_base, "LPIPS_vs_base": 0.0, "SSIM_vs_base": 1.0}]

    for name, ckpt in models.items():
        pipe = load_pipeline(args.base_model, args.vae_model, ckpt, device, dtype)
        gen = generate(pipe, prompts, seeds, os.path.join(gen_root, name),
                       args.steps, args.guidance, args.resolution, device)
        c = clip_scores(gen, device)
        f = fid_against_real(gen, real_paths, device)
        lp, ss = lpips_ssim_vs_base(gen, base, device)
        per_image[name] = c
        rows.append({"model": name, "CLIP_mean": c.mean(), "CLIP_sd": c.std(),
                     "FID_vs_real": f, "LPIPS_vs_base": lp.mean(), "SSIM_vs_base": ss.mean()})
        print(f"{name:16s} CLIP {c.mean():.3f} | FID {f:.2f} | "
              f"LPIPS {lp.mean():.3f} | SSIM {ss.mean():.3f}")

    iqa = pd.DataFrame(rows).round(4)
    iqa.to_csv(os.path.join(args.output_dir, "ablation_iqa.csv"), index=False)
    pd.DataFrame(per_image).to_csv(os.path.join(args.output_dir, "clip_per_image_all.csv"), index=False)

    omni = omnibus(per_image)
    print(f"OMNIBUS ANOVA: F={omni['anova_F']:.3f} p={omni['anova_p']:.4g} eta2={omni['eta2']:.3f}")
    print(f"FRIEDMAN: chi2={omni['friedman_chi2']:.3f} "
          f"p={omni['friedman_p']:.4g} Kendall's W={omni['kendalls_W']:.3f}")
    json.dump(omni, open(os.path.join(args.output_dir, "anova_omnibus.json"), "w"), indent=2)

    pw = pd.DataFrame(pairwise_holm(list(per_image), list(per_image.values()))).round(4)
    pw.to_csv(os.path.join(args.output_dir, "pairwise_stats.csv"), index=False)
    print(f"Saved evaluation artifacts to {args.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all TurathiAI checkpoints.")
    ap.add_argument("--output-dir", required=True, help="Dir holding lora_*_best checkpoints.")
    ap.add_argument("--dataset-dir", required=True, help="Real residential images (FID reference).")
    ap.add_argument("--prompts", required=True, help="Test-prompt file (100 prompts).")
    ap.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--vae-model", default="madebyollin/sdxl-vae-fp16-fix")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--quick", action="store_true", help="25 prompts x 1 seed smoke test.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
