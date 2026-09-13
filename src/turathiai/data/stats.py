"""Dataset statistics & diversity/bias analysis (Section 3.1.4).

Scripted counterpart of ``notebooks/TurathiAI_Dataset_Statistics``. Computes:

* completeness of image/mask/caption triples,
* caption length + vocabulary statistics,
* image resolution / aspect-ratio / brightness statistics,
* mask foreground coverage / bounding-box fill / component counts,
* keyword-based diversity: normalized-entropy **balance** (Shannon equitability)
  and the **Gini** coefficient per attribute category.
"""
from __future__ import annotations

import collections
import math
import os
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ɏ']+")

KEYWORD_CATEGORIES = {
    "View / perspective": {
        "Exterior": ["exterior", "elevation", "facade", "façade", "front view", "side view"],
        "Interior": ["interior", "indoor", "inside", "majlis"],
        "Close-up / detail": ["close-up", "close up", "detail", "motif"],
    },
    "Building element": {
        "Wind tower (barjeel)": ["wind tower", "windtower", "wind catcher", "barjeel", "barjeil"],
        "Mashrabiya / screen": ["mashrabiya", "wooden screen", "screen"],
        "Arch": ["arch"], "Dome": ["dome"], "Courtyard": ["courtyard"],
        "Parapet": ["parapet"], "Vaulted roof": ["vault"], "Flat roof": ["flat roof"],
        "Window": ["window"], "Door": ["door"], "Niche": ["niche"],
    },
    "Material": {
        "Coral stone": ["coral stone", "coral"], "Gypsum": ["gypsum"],
        "Mud / mortar": ["mud", "mortar"], "Plaster": ["plaster"],
        "Sandstone": ["sandstone"], "Wood": ["wood", "wooden", "timber"],
    },
}


def entropy(counts) -> float:
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0)


def gini(counts) -> float:
    xs = sorted(counts)
    n, s = len(xs), sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


def balance_score(counts) -> float:
    """Normalized entropy (Shannon equitability), 0..1. Higher = more even."""
    vals = list(counts)
    H = entropy(vals)
    Hmax = math.log2(len(vals)) if len(vals) > 1 else 1
    return (H / Hmax) if Hmax > 0 else 0.0


def caption_stats(captions: dict[str, str]) -> dict:
    all_tokens: list[str] = []
    lengths = []
    for text in captions.values():
        toks = [t.lower() for t in TOKEN_RE.findall(text)]
        all_tokens.extend(toks)
        lengths.append((len(toks), len(text)))
    words = np.array([w for w, _ in lengths]) if lengths else np.array([0])
    chars = np.array([c for _, c in lengths]) if lengths else np.array([0])
    return {
        "n_captions": len(captions),
        "mean_words": float(words.mean()), "median_words": float(np.median(words)),
        "mean_chars": float(chars.mean()),
        "vocab_size": len(set(all_tokens)), "total_tokens": len(all_tokens),
        "top_keywords": collections.Counter(all_tokens).most_common(30),
    }


def diversity_balance(captions: dict[str, str],
                      categories: dict = KEYWORD_CATEGORIES) -> list[dict]:
    texts = [t.lower() for t in captions.values()]
    summary = []
    for cat, classes in categories.items():
        counts = {cls: sum(1 for t in texts if any(k in t for k in kws))
                  for cls, kws in classes.items()}
        vals = list(counts.values())
        summary.append({
            "category": cat, "classes": len(vals),
            "classes_present": int(sum(1 for v in vals if v > 0)),
            "samples_tagged": int(sum(vals)),
            "balance_0to1": round(balance_score(vals), 3),
            "gini_0to1": round(gini(vals), 3),
            "top_class": max(counts, key=counts.get) if counts else None,
            "top_count": max(vals) if vals else 0,
        })
    return summary
