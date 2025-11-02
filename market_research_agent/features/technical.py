"""Technical analysis feature computation module."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


from ..base import Feature, FeatureComputer


class TechnicalFeatures(FeatureComputer):
    """Technical analysis feature computer."""

    def __init__(self):
        self.feature_map = {
            "sma_20": self._compute_sma,
            "sma_50": self._compute_sma,
            "sma_200": self._compute_sma,
            "ema_20": self._compute_ema,
            "rsi_14": self._compute_rsi,
            "macd": self._compute_macd,
            "atr": self._compute_atr,
            "beta": self._compute_beta,
            "volume_ratio": self._compute_volume_ratio,
        }

    def compute_features(
        self,
        data: Dict[str, pd.DataFrame],
        features: List[Feature]
    ) -> Dict[str, pd.DataFrame]:
        """Compute technical features from OHLCV data."""
        results = {}
        for feature in features:
            if feature.name in self.feature_map:
                compute_fn = self.feature_map[feature.name]
                try:
                    result = compute_fn(data, feature.window)
                    results[feature.name] = result
                except Exception as e:
                    # Log error and continue
                    print(f"Error computing {feature.name}: {str(e)}")
                    continue
        return results

    def _compute_sma(
        self,
        data: Dict[str, pd.DataFrame],
        window: Optional[int]
    ) -> pd.DataFrame:
        """Compute Simple Moving Average."""
        df = data["prices"]  # Assuming OHLCV DataFrame
        return df["close"].rolling(window=window).mean()

    def _compute_ema(
        self,
        data: Dict[str, pd.DataFrame],
        window: Optional[int]
    ) -> pd.DataFrame:
        """Compute Exponential Moving Average."""
        df = data["prices"]
        return df["close"].ewm(span=window, adjust=False).mean()

    def _compute_rsi(
        self,
        data: Dict[str, pd.DataFrame],
        window: Optional[int] = 14
    ) -> pd.DataFrame:
        """Compute Relative Strength Index (RSI) using pandas only."""
        df = data["prices"]
        close = df["close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_macd(
        self,
        data: Dict[str, pd.DataFrame],
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """Compute MACD (Moving Average Convergence Divergence) using pandas only."""
        df = data["prices"]
        close = df["close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    def _compute_atr(
        self,
        data: Dict[str, pd.DataFrame],
        window: Optional[int] = 14
    ) -> pd.DataFrame:
        """Compute Average True Range (ATR) using pandas only."""
        df = data["prices"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def _compute_beta(
        self,
        data: Dict[str, pd.DataFrame],
        window: Optional[int] = 252
    ) -> pd.DataFrame:
        """Compute beta against market index."""
        stock_returns = data["prices"]["close"].pct_change()
        market_returns = data["market"]["close"].pct_change()
        
        # Rolling beta calculation
        cov = stock_returns.rolling(window=window).cov(market_returns)
        market_var = market_returns.rolling(window=window).var()
        return cov / market_var

    def _compute_volume_ratio(
        self,
        data: Dict[str, pd.DataFrame],
        window: Optional[int] = 20
    ) -> pd.DataFrame:
        """Compute volume ratio vs moving average."""
        df = data["prices"]
        avg_volume = df["volume"].rolling(window=window).mean()
        return df["volume"] / avg_volume

    def validate_features(
        self,
        features: Dict[str, pd.DataFrame]
    ) -> tuple[bool, Dict[str, List[str]]]:
        """Validate computed features."""
        errors = {}
        for name, feature_df in features.items():
            feature_errors = []
            
            # Check for nulls
            null_pct = feature_df.isnull().mean()
            if null_pct > 0.1:  # More than 10% nulls
                feature_errors.append(f"High null ratio: {null_pct:.2%}")
            
            # Check for infinities
            inf_count = np.isinf(feature_df).sum()
            if inf_count > 0:
                feature_errors.append(f"Contains {inf_count} infinite values")
            
            if feature_errors:
                errors[name] = feature_errors
        
        return len(errors) == 0, errors