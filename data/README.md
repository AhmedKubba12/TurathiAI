# CHDB — Cultural Heritage Design Database

The **CHDB** dataset is the curated multimodal resource introduced in the paper
(Section 3.1). It consists of **634 complete triples** (1,902 files in total),
each pairing a manually captured residential-building photograph with a
descriptive caption and a segmentation mask.

## Layout

All samples live in a single flat folder. Files that share a base name form one
triple:

| File                  | Role                          |
|-----------------------|-------------------------------|
| `NAME.jpg`            | building image (JPEG)         |
| `NAME-masklabel.png`  | segmentation mask (PNG)       |
| `NAME.txt`            | caption / annotation (UTF-8)  |

```
data/CHDB_Full/
├── Al Aboudi house (West elevation 1).JPG
├── Al Aboudi house (West elevation 1)-masklabel.png
├── Al Aboudi house (West elevation 1).txt
├── ...
└── test_prompts.txt        # (optional) 100 held-out evaluation prompts
```

The dataset itself is **not** committed to this repository (see `.gitignore`); it
is distributed separately under its own terms (contact the authors). Place the extracted `CHDB_Full`
folder here, or point the config/CLIs at your own path.

## Key statistics (Section 3.1.4)

| Property                          | Value                              |
|-----------------------------------|------------------------------------|
| Complete triples                  | 634                                |
| Emirates covered                  | Sharjah, Dubai, Abu Dhabi          |
| Median image resolution           | 6000 × 4000 px (24 MP)             |
| Minimum image resolution          | 3024 × 3024 px (12.2 MP)           |
| Median aspect ratio               | 1.50                               |
| Caption length (mean / median)    | 80.8 / 69 words (496 chars mean)   |
| Vocabulary size                   | 919 unique tokens (51,251 total)   |
| Near-duplicate pairs (pHash ≤ 5)  | 1                                  |
| Mean mask foreground coverage     | 0.72 (median 0.75)                 |
| Mean bounding-box fill ratio      | 0.88                               |
| Mean connected components / mask  | 1.08                               |

Diversity (normalized-entropy **balance**, 0–1, and **Gini**, 0–1):

| Dimension        | Balance | Gini | Most frequent classes                     |
|------------------|:-------:|:----:|-------------------------------------------|
| View/perspective |  0.90   | 0.25 | close-up/detail (564)                     |
| Building element |  0.83   | 0.48 | niche (363), arch (259)                   |
| Material         |  0.85   | 0.35 | wood (342), coral stone (271)             |

Reproduce all of the above with
`notebooks/TurathiAI_Dataset_Statistics.ipynb` or `turathiai.data.stats`.

## Segmentation masks

Masks are generated automatically with SAM + a heuristic scoring function
(`turathiai.segmentation`). To regenerate them you need a SAM checkpoint:

```bash
mkdir -p checkpoints
# ViT-H (default). See https://github.com/facebookresearch/segment-anything
wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

bash scripts/run_segmentation.sh data/CHDB_Full checkpoints/sam_vit_h_4b8939.pth
```

Validate mask quality against manually annotated ground truth (Table 6):

```bash
python -m turathiai.segmentation.evaluate_masks \
  --pred-dir data/CHDB_Full --gt-dir data/gt_masks --out-csv outputs/mask_eval.csv
```
