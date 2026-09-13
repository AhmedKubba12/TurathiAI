"""Image-quality metrics and statistical tests."""
from .metrics import clip_scores, fid_against_real, lpips_ssim_vs_base
from .stats_tests import omnibus, pairwise_holm, anova_eta2, friedman_kendall

__all__ = ["clip_scores", "fid_against_real", "lpips_ssim_vs_base",
           "omnibus", "pairwise_holm", "anova_eta2", "friedman_kendall"]
