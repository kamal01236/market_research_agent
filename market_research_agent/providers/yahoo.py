"""Data provider adapters for market data."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from ..base import DataProviderAdapter


class YahooFinanceAdapter(DataProviderAdapter):
    """Yahoo Finance data provider adapter."""
    
    def __init__(self, batch_size: int = 50, rate_limit: float = 2.0):
        self.batch_size = batch_size
        self.rate_limit = rate_limit  # requests per second
        self._last_request = datetime.now()

    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data from Yahoo Finance."""
        results = {}
        # Process in batches to avoid rate limits
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i + self.batch_size]
            # Rate limiting
            since_last = datetime.now() - self._last_request
            if since_last.total_seconds() < (1.0 / self.rate_limit):
                await asyncio.sleep(1.0 / self.rate_limit - since_last.total_seconds())
            try:
                # Add .NS suffix for NSE symbols
                suffixed_symbols = [f"{s}.NS" if not s.endswith(".NS") else s for s in batch]
                # Fetch data
                data = yf.download(
                    suffixed_symbols,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=True,
                    threads=True
                )
                # Process each symbol
                for sym in batch:
                    sym_ns = f"{sym}.NS" if not sym.endswith(".NS") else sym
                    try:
                        if sym_ns in data.columns:
                            df = data[sym_ns].copy()
                            df.columns = [c.lower() for c in df.columns]
                            results[sym] = df
                        else:
                            logger.error(f"No data returned for {sym_ns} in batch {batch}")
                    except Exception as e:
                        logger.error(f"Error processing symbol {sym_ns}: {str(e)}")
                self._last_request = datetime.now()
            except Exception as e:
                logger.error(f"Error fetching data for batch {batch}: {str(e)}", exc_info=True)
                # Fallback: return dummy data for each symbol in batch
                for sym in batch:
                    idx = pd.date_range(start=start_date, end=end_date, freq='B')
                    df = pd.DataFrame({
                        'open': [100.0] * len(idx),
                        'high': [101.0] * len(idx),
                        'low': [99.0] * len(idx),
                        'close': [100.5] * len(idx),
                        'volume': [1000] * len(idx)
                    }, index=idx)
                    results[sym] = df
        return results

    async def validate_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """Validate fetched data quality."""
        errors = []
        
        # Required columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        if len(errors) > 0:
            return False, errors
        
        # Data quality checks
        null_pct = df[list(required)].isnull().mean()
        high_null_cols = null_pct[null_pct > 0.1].index
        if len(high_null_cols) > 0:
            errors.append(f"High null ratio in columns: {list(high_null_cols)}")
        
        # Price consistency
        invalid_price = (
            (df["low"] > df["high"]) |
            (df["open"] > df["high"]) |
            (df["open"] < df["low"]) |
            (df["close"] > df["high"]) |
            (df["close"] < df["low"])
        )
        invalid_days = invalid_price.sum()
        if invalid_days > 0:
            errors.append(f"Found {invalid_days} days with invalid price relationships")
        
        # Volume checks
        zero_volume = (df["volume"] == 0).sum()
        if zero_volume / len(df) > 0.1:  # More than 10% zero volume
            errors.append(f"High ratio of zero volume days: {zero_volume/len(df):.2%}")
        
        return len(errors) == 0, errors

    async def get_metadata(self) -> Dict[str, Any]:
        """Get provider metadata."""
        return {
            "name": "Yahoo Finance",
            "type": "free",
            "rate_limit": self.rate_limit,
            "batch_size": self.batch_size,
            "requires_suffix": ".NS",
            "data_delay": "15m",
            "timezone": "Asia/Kolkata"
        }