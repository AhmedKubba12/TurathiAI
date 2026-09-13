#!/usr/bin/env bash
# Score every lora_*_best checkpoint (CLIP/FID/LPIPS/SSIM) + ANOVA/Friedman/Holm.
set -euo pipefail
OUTPUT_DIR="${1:-outputs}"
DATASET_DIR="${2:-data/CHDB_Full}"
PROMPTS="${3:-data/test_prompts.txt}"
python -m turathiai.evaluation.evaluate \
    --output-dir "$OUTPUT_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --prompts "$PROMPTS"
