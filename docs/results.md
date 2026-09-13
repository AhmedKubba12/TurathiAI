# Results

All numbers are reproduced from the paper. Regenerate them with
`scripts/evaluate.sh` and `notebooks/TurathiAI_Ablation_IQA_and_ANOVA.ipynb`.

## 1. Segmentation quality (Table 6, n = 30)

| Metric                    | Mean  | Median | SD    |
|---------------------------|:-----:|:------:|:-----:|
| Intersection-over-Union   | 0.768 | 0.845  | 0.207 |
| Dice coefficient          | 0.852 | 0.916  | 0.149 |
| Precision                 | 0.838 | 0.996  | 0.240 |
| Recall                    | 0.928 | 0.956  | 0.088 |

IoU 95% bootstrap CI: [0.691, 0.840]. High recall confirms the masks capture the
full building extent — the property that matters most for masked training.

## 2. AdamW experiments — CLIP score (Tables 14–15)

ANOVA: **F = 5.9436, p = 0.0006, η² = 0.043**.

| Model         | Mean CLIP | SD      | Median  |
|---------------|:---------:|:-------:|:-------:|
| Base SDXL     | 0.3401    | 0.0192  | 0.3392  |
| Fine-tuned    | 0.3321    | 0.0228  | 0.3331  |
| LoRA rank 16  | 0.3423    | 0.0193  | 0.3427  |
| LoRA rank 128 | 0.3426    | 0.0197  | 0.3432  |

Pairwise vs. base (two-sample t-test, df = 198): full fine-tuning **degrades**
alignment significantly (p = 0.007, d = −0.384), while both LoRA models keep
statistical parity (p > 0.05) — LoRA avoids the fine-tuning overfitting pitfall.

LPIPS / SSIM vs. base (Table 17) and FID vs. real (Table 18):

| Model         | LPIPS ↓ | SSIM ↑ | FID ↓  |
|---------------|:-------:|:------:|:------:|
| Fine-tuned    | 0.6927  | 0.4040 | 178.16 |
| LoRA rank 16  | 0.5718  | 0.4540 | 203.06 |
| LoRA rank 128 | 0.5728  | 0.4502 | 203.95 |
| Base SDXL     | —       | —      | 209.69 |

## 3. Prodigy experiments — CLIP score (Tables 19–22)

ANOVA: **F = 12.1014, p < 0.0001, η² = 0.089**.

| Model          | Mean CLIP | LPIPS ↓ | SSIM ↑ | FID ↓  |
|----------------|:---------:|:-------:|:------:|:------:|
| Base SDXL      | 0.3401    | —       | —      | 209.69 |
| LoRA rank 32   | 0.3267    | 0.7268  | 0.3563 | 185.16 |
| **LoRA rank 64** | 0.3316  | 0.7136  | 0.3662 | **163.41** |
| LoRA rank 128  | 0.3210    | 0.7379  | 0.3282 | 168.01 |
| LoRA rank 256  | 0.3317    | 0.7150  | 0.3851 | 167.19 |

**LoRA rank 64 (Prodigy)** is the overall best: it is perceptually closest to the
real dataset (lowest FID) while preserving base-model perceptual/structural
behaviour — learning the domain without collapsing generalization.

## 4. Ablation study (Table 26)

Baseline = rank 32, Prodigy, constant schedule, masked loss, min-SNR. Each
variant changes one factor and is trained for 9,000 steps.

| Configuration                | CLIP  | FID   | LPIPS | SSIM  | Val loss | Δ vs. baseline |
|------------------------------|:-----:|:-----:|:-----:|:-----:|:--------:|:--------------:|
| Base SDXL (zero-shot)        | 31.53 | 152.2 | —     | —     | —        | —              |
| **Baseline**                 | 30.97 | 129.2 | 0.586 | 0.567 | 0.0607   | —              |
| LoRA rank 16                 | 30.89 | 126.5 | 0.608 | 0.566 | 0.0607   | +0.0%          |
| LoRA rank 64                 | 31.21 | 130.8 | 0.601 | 0.552 | 0.0607   | +0.0%          |
| Optimizer: AdamW-8bit        | 30.76 | 123.6 | 0.626 | 0.538 | 0.0607   | −0.1%          |
| Scheduler: cosine            | 30.71 | 127.2 | 0.634 | 0.538 | 0.0607   | −0.0%          |
| Half training data           | 31.35 | 134.2 | 0.580 | 0.577 | 0.0608   | +0.2%          |
| **Without masked loss**      | 30.79 | 136.0 | 0.603 | 0.543 | 0.0735   | **+21.2%**     |
| **Without min-SNR weighting**| 30.93 | 130.0 | 0.589 | 0.565 | 0.0945   | **+55.7%**     |

**Takeaway:** rank, optimizer, scheduler and a moderate data reduction are nearly
inconsequential; **masked training** and **min-SNR weighting** are the decisive
components.

Omnibus over per-image CLIP (n = 400/model): ANOVA **F = 9.76, p < 10⁻¹²,
η² = 0.021**; Friedman **χ² = 124, p < 10⁻²², Kendall's W = 0.039**. Differences
are statistically detectable but small — the design-relevant signal is the large
**FID improvement** of every adapted model over the base.

## 5. Comparison with baselines (Table 27)

| Approach                             | CLIP ↑ | FID ↓  |
|--------------------------------------|:------:|:------:|
| **TurathiAI LoRA (ours)**            | 33.56  | **123.34** |
| Base SDXL (zero-shot)                | 33.89  | 151.99 |
| Base SDXL + prompt engineering       | 31.36  | 197.67 |

Dataset-based adaptation gives a clearer, more reliable advantage than prompt
engineering (which even *reduces* CLIP while producing more heritage-oriented
imagery — evidence of CLIP's limits for cultural authenticity).

## 6. Blind expert evaluation (Table 29)

Three domain experts, 12 images, blind, 1–5 scale. ICC(2,1) = 0.79.

| Criterion              | TurathiAI | Base | Δ      |
|------------------------|:---------:|:----:|:------:|
| Image Quality          | 3.88      | 3.92 | −0.04  |
| Architectural Accuracy | 4.12      | 3.94 | +0.18  |
| Prompt Adherence       | 3.92      | 3.77 | +0.15  |
| Creativity             | 3.43      | 3.43 |  0.00  |
| **Overall**            | **3.84**  | 3.76 | +0.08  |

## 7. Generalizability across backbones (Section 4.5)

The same dataset + adaptation pipeline transfers to **Qwen-Image, FLUX, HiDream,
OmniGen2, and Z-Image**, reproducing heritage concepts (four-sided barjeel,
coral-stone/gypsum facades, mashrabiya, majlis interiors) on all of them —
evidence the contribution is a reusable resource, not a single-model artifact.
