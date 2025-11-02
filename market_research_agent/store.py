"""Feature store implementation."""
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import text

from .base import Feature


class FeatureStore:
    """TimescaleDB-backed feature store."""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self._init_tables()

    def _init_tables(self):
        """Initialize feature tables if they don't exist."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS features (
                    symbol TEXT,
                    ts TIMESTAMPTZ,
                    category TEXT,
                    feature_name TEXT,
                    value DOUBLE PRECISION,
                    metadata JSONB,
                    PRIMARY KEY (symbol, ts, feature_name)
                );
                
                SELECT create_hypertable('features', 'ts', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 week'
                );
                
                CREATE INDEX IF NOT EXISTS idx_features_lookup 
                ON features (feature_name, ts DESC);
            """))

    async def store_features(
        self,
        features: Dict[str, pd.DataFrame],
        metadata: Optional[Dict] = None
    ):
        """Store computed features."""
        for feature_name, feature_df in features.items():
            # Convert to long format for storage
            df_long = feature_df.reset_index()
            df_long.columns = ["ts", "value"]
            df_long["feature_name"] = feature_name
            df_long["metadata"] = str(metadata or {})
            
            # Batch insert
            with self.engine.connect() as conn:
                df_long.to_sql(
                    "features",
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi"
                )

    async def get_features(
        self,
        symbols: List[str],
        features: List[str],
        start_ts: datetime,
        end_ts: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve features for given symbols and timerange."""
        query = text("""
            SELECT symbol, ts, feature_name, value
            FROM features
            WHERE symbol = ANY(:symbols)
            AND feature_name = ANY(:features)
            AND ts BETWEEN :start_ts AND :end_ts
            ORDER BY ts ASC
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "symbols": symbols,
                    "features": features,
                    "start_ts": start_ts,
                    "end_ts": end_ts
                }
            )
        
        # Pivot to wide format
        df_wide = df.pivot(
            index="ts",
            columns=["symbol", "feature_name"],
            values="value"
        )
        
        if interval != "1d":
            df_wide = df_wide.resample(interval).last()
        
        return df_wide

    async def delete_features(
        self,
        older_than: datetime,
        features: Optional[List[str]] = None
    ):
        """Delete old features based on retention policy."""
        query = text("""
            DELETE FROM features
            WHERE ts < :older_than
            AND (:features IS NULL OR feature_name = ANY(:features))
        """)
        
        with self.engine.connect() as conn:
            conn.execute(
                query,
                parameters={
                    "older_than": older_than,
                    "features": features
                }
            )