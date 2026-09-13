#!/usr/bin/env bash
# Train the baseline + every ablation variant (Table 26).
set -euo pipefail
accelerate launch -m turathiai.training.train_lora --config configs/default.yaml
for cfg in configs/ablations/*.yaml; do
    echo "=== $cfg ==="
    accelerate launch -m turathiai.training.train_lora --config "$cfg"
done
