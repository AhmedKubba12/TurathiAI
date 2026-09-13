"""Automated building segmentation (SAM + heuristic scoring, Section 3.1.3)."""
from .scoring import mask_score, select_best_mask, rank_masks, bbox_fill_ratio
from .postprocess import refine_mask

__all__ = ["mask_score", "select_best_mask", "rank_masks", "bbox_fill_ratio", "refine_mask"]
