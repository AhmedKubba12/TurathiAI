# Methodology

A condensed map from the paper's methodology (Section 3) to the code.

## Dataset curation (§3.1)

1. **Acquisition (§3.1.1)** — 634 residential-building photos manually captured
   across Sharjah, Dubai, and Abu Dhabi (JPEG, up to 6000×4000).
2. **Annotation (§3.1.2)** — an expert-informed template (Table 2) drives
   consistent captions describing location, structure, fenestration, materials
   and ornament. → analysed by `turathiai.data.stats`.
3. **Segmentation (§3.1.3)** — automated masking to enable **masked training**:
   - downscale to 1024×1024;
   - SAM automatic mask generation → many candidates;
   - rank candidates with the heuristic **score = area × (area / bbox_area)**
     (Equation 1) → `turathiai.segmentation.scoring`;
   - refine: 15×15 morphological closing + largest-contour polygon, upscale to
     native resolution → `turathiai.segmentation.postprocess`;
   - validate vs. ground truth (IoU/Dice/Precision/Recall) →
     `turathiai.segmentation.evaluate_masks`.

## Model training (§3.2)

- **Backbone:** Stable Diffusion XL + `sdxl-vae-fp16-fix`.
- **Adaptation:** LoRA on the UNet attention projections (`to_q/k/v/out`) via
  `peft`. Full fine-tuning is compared against LoRA across ranks {16, 32, 64,
  128, 256}.
- **Optimizers:** Prodigy (learning-rate-free) and AdamW / AdamW-8bit →
  `turathiai.training.optimizers`.
- **Losses** (`turathiai.training.losses`), the two decisive components in the
  ablation:
  - **Masked MSE** — the denoising loss is restricted to building pixels using
    the segmentation mask downsampled to the latent grid;
  - **min-SNR-γ weighting** (γ = 5) — per-timestep loss reweighting.
- **Checkpoint selection:** a fixed-condition **validation loss** (fixed
  timesteps + seeds on the ~10% held-out split) gives a clean curve; the best
  checkpoint is saved (`save_best_on_val`).

## Evaluation (§3.3)

- **CLIP score** (open_clip ViT-B-32) — text–image alignment.
- **FID** — distance to the 634 real residential images (InceptionV3 2048-dim).
- **LPIPS / SSIM** — perceptual/structural distance vs. base-SDXL outputs.
- **Statistics** — one-way ANOVA + η², Friedman + Kendall's W, Holm-corrected
  pairwise paired t-tests with Cohen's d → `turathiai.evaluation.stats_tests`.
- **Protocol** — 100 held-out prompts × 4 seeds = 400 images per model
  (30% traditional exteriors, 30% modern reinterpretations, 30% interiors,
  10% close-up details).

See `docs/results.md` for all numbers.
