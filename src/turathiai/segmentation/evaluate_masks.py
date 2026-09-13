"""Quantitative validation of the automated segmentation pipeline.

Reproduces the mask-quality evaluation in Section 3.1.3 / Table 6: given a set of
predicted masks and manually annotated ground-truth masks, compute
Intersection-over-Union, Dice, Precision and Recall, and summarise them with the
mean, median, standard deviation and a bootstrap confidence interval for IoU.

The paper reports (n = 30 stratified images): mean IoU 0.768 (median 0.845),
mean Dice 0.852, mean Precision 0.838 (median 0.996), mean Recall 0.928.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 127)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _binary(pred), _binary(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union else 1.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _binary(pred), _binary(gt)
    denom = p.sum() + g.sum()
    return float(2 * np.logical_and(p, g).sum() / denom) if denom else 1.0


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _binary(pred), _binary(gt)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    return float(tp / (tp + fp)) if (tp + fp) else 1.0


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _binary(pred), _binary(gt)
    tp = np.logical_and(p, g).sum()
    fn = np.logical_and(~p, g).sum()
    return float(tp / (tp + fn)) if (tp + fn) else 1.0


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _load(path: str, like: np.ndarray | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if like is not None and im.size != (like.shape[1], like.shape[0]):
        im = im.resize((like.shape[1], like.shape[0]), Image.NEAREST)
    return np.asarray(im)


def evaluate(pred_dir: str, gt_dir: str, mask_suffix: str = "-masklabel.png"):
    import pandas as pd

    gts = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    rows = []
    for gt_path in gts:
        name = os.path.basename(gt_path).replace(mask_suffix, "").replace(".png", "")
        pred_path = os.path.join(pred_dir, name + mask_suffix)
        if not os.path.exists(pred_path):
            print(f"  [skip] no prediction for {name}")
            continue
        gt = _load(gt_path)
        pred = _load(pred_path, like=gt)
        rows.append({
            "sample": name,
            "IoU": iou(pred, gt),
            "Dice": dice(pred, gt),
            "Precision": precision(pred, gt),
            "Recall": recall(pred, gt),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No matched prediction/ground-truth pairs found.")
        return df

    summary = df[["IoU", "Dice", "Precision", "Recall"]].agg(["mean", "median", "std"]).round(3)
    lo, hi = bootstrap_ci(df["IoU"].to_numpy())
    print(f"Evaluated {len(df)} masks\n")
    print(summary.to_string())
    print(f"\nIoU 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate predicted masks vs. ground truth.")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    df = evaluate(args.pred_dir, args.gt_dir)
    if args.out_csv and not df.empty:
        df.round(4).to_csv(args.out_csv, index=False)
        print(f"\nSaved per-mask scores to {args.out_csv}")


if __name__ == "__main__":
    main()
