from .technical import TechnicalFeatures
import pandas as pd
import numpy as np

def sma(data, window=20):
    s = pd.Series(data)
    return s.rolling(window=window).mean()

def ema(data, window=20):
    s = pd.Series(data)
    return s.ewm(span=window, adjust=False).mean()

def rsi(data, window=14):
    s = pd.Series(data)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def momentum(data, period=1):
    s = pd.Series(data)
    return s.pct_change(periods=period)

def atr(high, low, close, window=14):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def zscore(data, window=20):
    s = pd.Series(data)
    roll = s.rolling(window=window)
    return (s - roll.mean()) / roll.std()