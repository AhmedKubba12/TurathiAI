"""Optimizer factory for the training experiments (Section 3.2).

Supports the two families compared in the paper:

* **Prodigy** — a learning-rate-free optimizer (D-Adaptation family). The base
  LR is fixed to 1.0 and Prodigy adapts the effective step size to the dataset.
* **AdamW / AdamW-8bit** — the classic adaptive baseline; the 8-bit variant
  (via ``bitsandbytes``) reduces optimizer-state VRAM.

The wider set of optimizers surveyed in the paper (SGD, AdEMAMix, Lion, RMSProp,
Adagrad) can be added here following the same pattern.
"""
from __future__ import annotations

from typing import Iterable

import torch


def build_optimizer(name: str, params: Iterable[torch.nn.Parameter],
                    learning_rate: float, weight_decay: float = 1e-2):
    name = name.lower()

    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install prodigy: pip install prodigyopt") from exc
        # Prodigy is learning-rate-free: lr acts as a multiplier, keep it at 1.0.
        return Prodigy(params, lr=learning_rate, weight_decay=weight_decay,
                       decouple=True, use_bias_correction=True, safeguard_warmup=True)

    if name in ("adamw8bit", "adamw_8bit"):
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install bitsandbytes for 8-bit AdamW.") from exc
        return bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay,
                                   betas=(0.9, 0.999), eps=1e-8)

    if name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8)

    raise ValueError(f"Unknown optimizer '{name}'. "
                     "Choose from: prodigy, adamw8bit, adamw.")
