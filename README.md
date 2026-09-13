<h1 align="center">TurathiAI</h1>

<p align="center">
  <b>Diffusion-Based Text-to-Image Model for Regional Residential Cultural-Heritage Architectural Design</b><br>
  <i>Fine-tuning Stable Diffusion XL on a purpose-built, expert-annotated dataset of Emirati residential heritage architecture.</i>
</p>

<p align="center">
  <a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="SDXL + LoRA" src="https://img.shields.io/badge/model-SDXL%20%2B%20LoRA-orange.svg">
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6081896"><img alt="Paper" src="https://img.shields.io/badge/paper-SSRN-green.svg"></a>
</p>

---

*Turathi* (تراثي) means *"my heritage"* in Arabic. TurathiAI adapts large
pretrained text-to-image diffusion models to generate residential architecture
that fuses **traditional Emirati design** with **contemporary form** as a
counterweight to the Westernized aesthetic bias of mainstream generative models.

The repository provides the full pipeline behind the paper: **dataset curation
and statistics**, **automated building segmentation** (SAM + a heuristic scoring
function), **LoRA fine-tuning of SDXL** with masked training and min-SNR
weighting, and a complete **image-quality + statistical evaluation** suite.

## Highlights

- 🏛️ **CHDB dataset**: 634 complete triples (image + caption + mask) of Emirati
  residential heritage buildings, manually captured and expert-annotated.
- ✂️ **Automated segmentation**: SAM candidate masks ranked by a large-compact-
  connected heuristic (paper Eq. 1), refined and validated (mean IoU 0.768).
- 🎯 **Adaptation study**: full fine-tuning vs. LoRA across ranks
  {16, 32, 64, 128, 256} and two optimizer families (Prodigy, AdamW).
- 🧪 **Ablation**: isolates the decisive components, **masked training** and
  **min-SNR weighting** (removing them raises validation loss by 21% / 56%).
- 📊 **Rigorous evaluation**: CLIP, FID, LPIPS, SSIM + ANOVA / Friedman /
  Holm-corrected pairwise tests, plus a blind expert study.
- 🔁 **Backbone-agnostic**: the same dataset + pipeline transfers to Qwen-Image,
  FLUX, HiDream, OmniGen2, and Z-Image.

## Repository structure

```
TurathiAI/
├── src/turathiai/
│   ├── config.py                 # dataclass configs + YAML loader
│   ├── segmentation/             # SAM masking (§3.1.3)
│   │   ├── scoring.py            #   heuristic mask score (Eq. 1)
│   │   ├── postprocess.py        #   morphological closing + largest contour
│   │   ├── sam_segment.py        #   end-to-end mask generation CLI
│   │   └── evaluate_masks.py     #   IoU / Dice / Precision / Recall (Table 6)
│   ├── data/
│   │   ├── dataset.py            #   image/caption/mask triples, masked training
│   │   └── stats.py              #   dataset statistics + diversity/bias (§3.1.4)
│   ├── training/
│   │   ├── train_lora.py         #   SDXL LoRA training loop (§3.2)
│   │   ├── losses.py             #   masked MSE + min-SNR weighting
│   │   └── optimizers.py         #   Prodigy / AdamW(-8bit) factory
│   ├── evaluation/
│   │   ├── metrics.py            #   CLIP / FID / LPIPS / SSIM (§3.3)
│   │   ├── stats_tests.py        #   ANOVA, Friedman, Holm pairwise
│   │   └── evaluate.py           #   all-checkpoints evaluation driver
│   └── generate.py               # inference CLI
├── configs/                      # default.yaml + ablations/
├── scripts/                      # run_segmentation / train / evaluate
├── notebooks/                    # dataset-statistics & ablation-IQA notebooks
├── data/README.md                # CHDB layout + statistics
└── docs/                         # methodology.md, results.md
```

## Installation

```bash
git clone https://github.com/AhmedKubba12/TurathiAI.git
cd TurathiAI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # exposes the turathiai-* CLIs
```

A CUDA GPU is required for training/generation. Experiments in the paper ran on
NVIDIA A10G (AWS EC2 G5), evaluation notebooks run on an L4. 8-bit optimizers
(`bitsandbytes`) need Linux + CUDA.

## Quick start

### 1. Get the data and the SAM checkpoint

Place the `CHDB_Full` folder under `data/` (see [`data/README.md`](data/README.md))
and download a SAM checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 2. Generate segmentation masks

```bash
bash scripts/run_segmentation.sh data/CHDB_Full checkpoints/sam_vit_h_4b8939.pth
```

### 3. Inspect the dataset

Open `notebooks/TurathiAI_Dataset_Statistics.ipynb` (completeness, caption/vocab
stats, resolution, mask coverage, diversity/bias), or use the library:

```python
from turathiai.data.dataset import index_triples
from turathiai.data.stats import caption_stats, diversity_balance

triples = index_triples("data/CHDB_Full")
captions = {t["sample"]: open(t["caption"]).read() for t in triples}
print(caption_stats(captions)["vocab_size"])
print(diversity_balance(captions))
```

### 4. Train

```bash
# baseline: LoRA rank 32, Prodigy, constant LR, masked loss, min-SNR
bash scripts/train.sh configs/default.yaml

# or an ablation variant
bash scripts/train.sh configs/ablations/no_snr.yaml

# or the whole grid (baseline + all ablations)
bash scripts/train_all_ablations.sh
```

### 5. Generate images

```bash
turathiai-generate \
  --lora outputs/lora_default_best \
  --prompts data/test_prompts.txt \
  --out generations/turathiai --seeds 42 43 44 45
```

### 6. Evaluate all checkpoints

```bash
bash scripts/evaluate.sh outputs data/CHDB_Full data/test_prompts.txt
```

This writes `ablation_iqa.csv`, `clip_per_image_all.csv`, `anova_omnibus.json`
and `pairwise_stats.csv`. The same analysis is available interactively in
`notebooks/TurathiAI_Ablation_IQA_and_ANOVA.ipynb`.

## Configuration

Everything is driven by small YAML files that override the dataclass defaults in
`src/turathiai/config.py`. The baseline (`configs/default.yaml`) reflects the
paper's best setup; each file in `configs/ablations/` changes exactly one factor:

| Config                     | Change vs. baseline            |
|----------------------------|--------------------------------|
| `rank16.yaml` / `rank64.yaml` | LoRA rank                   |
| `opt_adamw8bit.yaml`       | Prodigy → AdamW-8bit           |
| `sched_cosine.yaml`        | constant → cosine schedule     |
| `half_data.yaml`           | halve the training set         |
| `no_masked_loss.yaml`      | disable masked training        |
| `no_snr.yaml`              | disable min-SNR weighting      |

## Methodology Overview

Building photos are segmented by generating many SAM candidate masks and keeping
the one that best matches a building, **large, compact and connected**, via the
score `area × (area / bounding_box_area)`, the winner is closed, reduced to its
largest contour and upscaled. SDXL is then adapted with LoRA while the denoising
loss is **restricted to building pixels** (masked training) and **reweighted per
timestep** with min-SNR-γ. Checkpoints are chosen by a fixed-condition validation
loss. Models are scored with CLIP/FID/LPIPS/SSIM and compared with ANOVA,
Friedman and Holm-corrected pairwise tests. See [`docs/methodology.md`](docs/methodology.md).

## Key results

LoRA adaptation (best: **Prodigy rank 64**) moves generations substantially
closer to the real residential distribution while preserving base-model
generalization, and beats a prompt-engineering baseline by a wide FID margin:

| Approach                         | CLIP ↑ | FID ↓  |
|----------------------------------|:------:|:------:|
| **TurathiAI LoRA (ours)**        | 33.56  | **123.34** |
| Base SDXL (zero-shot)            | 33.89  | 151.99 |
| Base SDXL + prompt engineering   | 31.36  | 197.67 |

The ablation shows **masked training** and **min-SNR weighting** are the decisive
components (all other factors change validation loss by ≤0.2%). Full numbers,
tables and the blind expert study are in [`docs/results.md`](docs/results.md).

## Responsible use

Generated designs are **early-stage conceptual/ideation aids**, not
construction-ready outputs — structural, ventilation and constructability
validation are out of scope. The authors advocate **human-in-the-loop review by
domain experts**, provenance labelling of outputs, and care around cultural
appropriation, fabrication of inauthentic heritage, and privacy of photographed
residences (paper, Section 5). The SDXL base model is governed by the CreativeML
Open RAIL++-M license.

## Citation

```bibtex
@article{abutalib2025turathiai,
  title   = {TurathiAI: Diffusion-Based Text-to-Image Model for Regional
             Residential Cultural Heritage Architectural Building Design},
  author  = {Abu Talib, Manar and Ibrahim, Iman and Ammar, Ahmed and
             Tabet Aoul, Kheira Anissa and Abuimara, Tareq},
  year    = {2025},
  note    = {Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6081896}
}
```

## License

Code released under the [MIT License](LICENSE). Models and the curated dataset
carry their own terms — see [`LICENSE`](LICENSE) and [`data/README.md`](data/README.md).
