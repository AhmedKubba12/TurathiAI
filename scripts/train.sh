#!/usr/bin/env bash
# Train the baseline LoRA, or pass an ablation config as $1.
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
accelerate launch -m turathiai.training.train_lora --config "$CONFIG"
