"""Heuristic mask-scoring function for building segmentation.

Implements the candidate-mask ranking described in Section 3.1.3 of the paper
(Equation 1). The heuristic is that a building forms a *large*, *compact*, and
*connected* region: it should cover most of the scene and fill most of its own
bounding box. Candidate masks produced by SAM are ranked with this score and the
top-scoring mask is kept.

    score(mask) = mask_area * (mask_area / bounding_box_area)

where ``mask_area`` is the number of foreground pixels and ``bounding_box_area``
is the area of the smallest axis-aligned rectangle enclosing the mask. The right
factor (``mask_area / bounding_box_area``) is the bounding-box fill ratio: it is
close to 1 for a solid, rectangular object and small for a fragmented mask (e.g.
tree branches), so it penalises noisy candidates.
"""
from __future__ import annotations

import numpy as np


def bounding_box_area(mask: np.ndarray) -> int:
    """Area of the tightest axis-aligned bounding box around the foreground."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    return int(width * height)


def bbox_fill_ratio(mask: np.ndarray) -> float:
    """Fraction of the bounding box that is filled by the mask (0..1)."""
    area = int(mask.sum())
    bbox = bounding_box_area(mask)
    return float(area / bbox) if bbox else 0.0


def mask_score(mask: np.ndarray) -> float:
    """Equation 1: reward large, compact, connected candidate masks.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or {0, 1} 2-D array where True/1 marks the candidate region.

    Returns
    -------
    float
        The heuristic score. Larger is better.
    """
    mask = mask.astype(bool)
    area = float(mask.sum())
    if area == 0:
        return 0.0
    return area * bbox_fill_ratio(mask)


def rank_masks(masks: list[np.ndarray]) -> list[int]:
    """Return indices of ``masks`` sorted by descending heuristic score."""
    scores = [mask_score(m) for m in masks]
    return sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)


def select_best_mask(masks: list[np.ndarray]) -> tuple[int, float]:
    """Return the ``(index, score)`` of the highest-scoring candidate mask."""
    if not masks:
        raise ValueError("No candidate masks to score.")
    order = rank_masks(masks)
    best = order[0]
    return best, mask_score(masks[best])
