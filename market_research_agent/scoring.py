"""Scoring utilities: normalization and aggregate score computation."""
from typing import Dict
import numpy as np


def normalize_zscore(values: Dict[str, float]) -> Dict[str, float]:
    """Given a dict of factor_name->value, return z-scored dict (single-step using np.mean/std)."""
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0:
        std = 1.0
    normalized = (arr - mean) / std
    return dict(zip(values.keys(), normalized.tolist()))


def aggregate_score(normalized: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute weighted sum of normalized features. Missing weights default to 0."""
    score = 0.0
    for k, v in normalized.items():
        w = weights.get(k, 0.0)
        score += w * float(v)
    return float(score)


def components(normalized: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in normalized.items():
        out[k] = {"norm": float(v), "weight": float(weights.get(k, 0.0)), "contribution": float(v * weights.get(k, 0.0))}
    return out
