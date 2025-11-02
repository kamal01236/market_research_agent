"""Feature computation utilities for the market research agent."""
from typing import Iterable
import numpy as np
import pandas as pd


def sma(series: Iterable[float], window: int) -> pd.Series:
    s = pd.Series(series).astype(float)
    return s.rolling(window=window, min_periods=1).mean()


def rsi(series: Iterable[float], window: int = 14) -> pd.Series:
    prices = pd.Series(series).astype(float)
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(window - 1), adjust=False).mean()
    ma_down = down.ewm(com=(window - 1), adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def momentum(series: Iterable[float], period: int = 5) -> pd.Series:
    s = pd.Series(series).astype(float)
    return s.pct_change(periods=period)


def atr(high: Iterable[float], low: Iterable[float], close: Iterable[float], window: int = 14) -> pd.Series:
    high_s = pd.Series(high).astype(float)
    low_s = pd.Series(low).astype(float)
    close_s = pd.Series(close).astype(float)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def zscore(series: Iterable[float], window: int = 90) -> pd.Series:
    s = pd.Series(series).astype(float)
    mean = s.rolling(window=window, min_periods=1).mean()
    std = s.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    return ((s - mean) / std).fillna(0)
