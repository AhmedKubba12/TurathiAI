"""Statistical analysis of per-image CLIP scores (Sections 4.1, 4.3).

Provides the omnibus and pairwise tests reported in the paper:

* One-way **ANOVA** with **eta-squared** effect size.
* Non-parametric **Friedman** test with **Kendall's W**.
* **Holm-corrected** pairwise **paired t-tests** with **Cohen's d**.

These mirror the ``TurathiAI_Ablation_IQA_and_ANOVA`` notebook.
"""
from __future__ import annotations

import numpy as np


def anova_eta2(groups: list[np.ndarray]) -> dict:
    """One-way ANOVA F/p plus eta-squared (share of variance explained)."""
    from scipy import stats

    F, p = stats.f_oneway(*groups)
    grand = np.concatenate(groups)
    gm = grand.mean()
    ss_between = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_total = ((grand - gm) ** 2).sum()
    eta2 = float(ss_between / ss_total) if ss_total else 0.0
    return {"anova_F": float(F), "anova_p": float(p), "eta2": eta2}


def friedman_kendall(groups: list[np.ndarray]) -> dict:
    """Friedman chi-square and Kendall's W on aligned (seed-matched) scores."""
    from scipy import stats

    minlen = min(len(g) for g in groups)
    aligned = [g[:minlen] for g in groups]
    chi, p = stats.friedmanchisquare(*aligned)
    W = float(chi / (minlen * (len(groups) - 1))) if minlen else 0.0
    return {"friedman_chi2": float(chi), "friedman_p": float(p), "kendalls_W": W}


def pairwise_holm(names: list[str], groups: list[np.ndarray]):
    """Holm-corrected pairwise paired t-tests with Cohen's d.

    Returns a list of dicts, one per model pair.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    minlen = min(len(g) for g in groups)
    aligned = [g[:minlen] for g in groups]
    recs, pvals = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = aligned[i], aligned[j]
            t, p = stats.ttest_rel(a, b)
            d = (a - b).mean() / (a - b).std(ddof=1)
            recs.append({"model_a": names[i], "model_b": names[j],
                         "t": float(t), "p_raw": float(p), "cohens_d": float(d)})
            pvals.append(p)
    rej, p_holm, *_ = multipletests(pvals, method="holm")
    for r, ph, rj in zip(recs, p_holm, rej):
        r["p_holm"] = float(ph)
        r["significant"] = bool(rj)
    return recs


def omnibus(per_image_clip: dict[str, np.ndarray]) -> dict:
    """Full omnibus report combining ANOVA and Friedman/Kendall."""
    names = list(per_image_clip.keys())
    groups = [np.asarray(per_image_clip[n]) for n in names]
    out = {"models": names, "n_per_model": [len(g) for g in groups]}
    out.update(anova_eta2(groups))
    out.update(friedman_kendall(groups))
    return out
