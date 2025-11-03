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


class FactorScorer:
    """Factor scoring engine for async API usage."""
    def __init__(self):
        pass

    def normalize_zscore(self, values: Dict[str, float]) -> Dict[str, float]:
        return normalize_zscore(values)

    def aggregate_score(self, normalized: Dict[str, float], weights: Dict[str, float]) -> float:
        return aggregate_score(normalized, weights)

    def components(self, normalized: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        return components(normalized, weights)

    async def compute_scores(self, features, factors=None, weights=None):
        # features: pd.DataFrame or dict-like
        # factors: list of factor names to use
        # weights: dict of factor->weight
        import pandas as pd
        if isinstance(features, dict):
            features = pd.DataFrame(features)
        if factors is None:
            factors = list(features.columns)
        if weights is None:
            weights = {f: 1.0 / len(factors) for f in factors}
        # Use last row if DataFrame
        row = features[factors].iloc[-1].to_dict()
        norm = self.normalize_zscore(row)
        score = self.aggregate_score(norm, weights)
        expl = self.components(norm, weights)
        return pd.Series({"score": score, **{f: expl[f]["contribution"] for f in factors}})

    async def get_available_factors(self):
        # In a real system, this would query config or DB
        return [
            "sma_20", "sma_50", "sma_200", "ema_20", "rsi_14", "macd", "atr", "beta", "volume_ratio"
        ]
