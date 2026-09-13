#!/usr/bin/env bash
# Generate building masks for the whole dataset with SAM + heuristic scoring.
set -euo pipefail
IMAGES_DIR="${1:-data/CHDB_Full}"
SAM_CKPT="${2:-checkpoints/sam_vit_h_4b8939.pth}"
python -m turathiai.segmentation.sam_segment \
    --images-dir "$IMAGES_DIR" \
    --sam-checkpoint "$SAM_CKPT" \
    --model-type vit_h
