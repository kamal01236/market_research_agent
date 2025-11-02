import pandas as pd
import numpy as np
from market_research_agent import features


def test_sma():
    data = [1, 2, 3, 4, 5]
    res = features.sma(data, window=3)
    assert isinstance(res, pd.Series)
    # last value is mean of [3,4,5]
    assert np.isclose(res.iloc[-1], (3 + 4 + 5) / 3)


def test_rsi_constant():
    # constant series should return near 50 for our implementation
    data = [100] * 20
    res = features.rsi(data, window=14)
    assert isinstance(res, pd.Series)
    assert np.allclose(res.fillna(50), 50)


def test_momentum():
    data = [1, 2, 4, 8, 16]
    res = features.momentum(data, period=1)
    # first is NaN -> pct_change yields NaN, later values should be positive
    assert np.isnan(res.iloc[0])
    assert np.isclose(res.iloc[1], 1.0)  # 2/1 -1 =1


def test_atr():
    high = [10, 11, 12, 13, 15]
    low = [9, 9.5, 11, 12, 13]
    close = [9.5, 10.5, 11.5, 12.5, 14]
    res = features.atr(high, low, close, window=3)
    assert isinstance(res, pd.Series)
    assert res.iloc[-1] >= 0


def test_zscore():
    data = list(range(1, 101))
    res = features.zscore(data, window=20)
    assert isinstance(res, pd.Series)
    # With increasing sequence, last zscore should be positive
    assert res.iloc[-1] > 0
