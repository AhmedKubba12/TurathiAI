"""Diffusion training losses: masked MSE and min-SNR weighting.

The ablation study (Section 4.3, Table 26) identified these two components as the
principal contributors to model quality: removing masked training raised the
validation loss by ~21%, and removing min-SNR weighting raised it by ~56%. Rank,
optimizer, scheduler and a moderate dataset reduction were comparatively
inconsequential.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_snr(scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """Signal-to-noise ratio at each timestep, from the scheduler's alphas.

    SNR(t) = alpha_bar_t / (1 - alpha_bar_t).
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    return (sqrt_alphas / sqrt_one_minus) ** 2


def min_snr_weights(scheduler, timesteps: torch.Tensor, gamma: float = 5.0,
                    prediction_type: str = "epsilon") -> torch.Tensor:
    """Per-sample min-SNR-gamma loss weights (Hang et al., 2023).

    weight = min(SNR, gamma) / SNR         (epsilon-prediction)
    weight = min(SNR, gamma) / (SNR + 1)   (v-prediction)
    """
    snr = compute_snr(scheduler, timesteps)
    clamped = torch.clamp(snr, max=gamma)
    if prediction_type == "v_prediction":
        return clamped / (snr + 1)
    return clamped / snr


def downsample_mask(mask: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
    """Downsample a pixel-space mask to the latent grid (e.g. 1024 -> 128)."""
    return F.interpolate(mask, size=latent_hw, mode="nearest")


def masked_mse_loss(model_pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None,
                    weights: torch.Tensor | None = None) -> torch.Tensor:
    """MSE denoising loss, optionally masked to building pixels and SNR-weighted.

    Parameters
    ----------
    model_pred, target : torch.Tensor
        ``(B, C, H, W)`` predicted and target noise (or v).
    mask : torch.Tensor, optional
        ``(B, 1, H, W)`` latent-grid mask in {0, 1}. If ``None`` the loss is
        computed over the whole latent (i.e. masked training disabled).
    weights : torch.Tensor, optional
        ``(B,)`` per-sample min-SNR weights. If ``None`` all samples weigh 1.
    """
    err = (model_pred.float() - target.float()) ** 2  # (B,C,H,W)

    if mask is not None:
        mask = mask.to(err.dtype)
        if mask.shape[-2:] != err.shape[-2:]:
            mask = downsample_mask(mask, err.shape[-2:])
        err = err * mask
        denom = mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
        per_sample = err.sum(dim=[1, 2, 3]) / denom
    else:
        per_sample = err.mean(dim=[1, 2, 3])

    if weights is not None:
        per_sample = per_sample * weights.to(per_sample.dtype)

    return per_sample.mean()
