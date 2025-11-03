"""Feature store implementation."""
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import text

from .base import Feature


class FeatureStore:
    """TimescaleDB-backed feature store."""

    def __init__(self, db_url: Optional[str] = None):
        # Use a persistent SQLite file if db_url is not provided
        if db_url is None:
            db_url = 'sqlite:///market_features.db'
        self.engine = create_engine(db_url)
        self._init_tables()

    def _init_tables(self):
        """Initialize feature tables if they don't exist."""
        with self.engine.begin() as conn:
            # Always create the features table
            if self.engine.url.get_backend_name().startswith("sqlite"):
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS features (
                        symbol TEXT,
                        ts TEXT,
                        category TEXT,
                        feature_name TEXT,
                        value DOUBLE PRECISION,
                        metadata TEXT,
                        PRIMARY KEY (symbol, ts, feature_name)
                    )
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_features_lookup 
                    ON features (feature_name, ts DESC)
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS features (
                        symbol TEXT,
                        ts TIMESTAMPTZ,
                        category TEXT,
                        feature_name TEXT,
                        value DOUBLE PRECISION,
                        metadata JSONB,
                        PRIMARY KEY (symbol, ts, feature_name)
                    )
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_features_lookup 
                    ON features (feature_name, ts DESC)
                """))

    async def store_features(
        self,
        features: Dict[str, pd.DataFrame],
        symbol: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Store computed features, replacing any existing rows for (symbol, ts, feature_name)."""
        for feature_name, feature_df in features.items():
            # Convert to long format for storage
            df_long = feature_df.reset_index()
            # If symbol is provided, add it as a column
            if symbol is not None:
                df_long["symbol"] = symbol
            # Only keep ts, value, symbol (if present)
            value_col = feature_df.columns[0]
            cols = ["ts", value_col]
            if symbol is not None:
                cols.append("symbol")
            df_long = df_long[cols]
            # Rename value column to 'value'
            df_long = df_long.rename(columns={value_col: "value"})
            df_long["feature_name"] = feature_name
            df_long["metadata"] = str(metadata or {})
            # Drop rows with NaN in ts or value
            df_long = df_long.dropna(subset=["ts", "value"])
            # Always set and order columns as symbol, ts, feature_name, value, metadata
            if symbol is not None:
                df_long = df_long[["symbol", "ts", "feature_name", "value", "metadata"]]
                # Convert ts to string (ISO format) for SQLite compatibility and deduplication
                df_long["ts"] = df_long["ts"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
                # Drop duplicates on all PK columns
                df_long = df_long.drop_duplicates(subset=["symbol", "ts", "feature_name"], keep="last")
            else:
                df_long = df_long[["ts", "feature_name", "value", "metadata"]]
                df_long["ts"] = df_long["ts"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
                df_long = df_long.drop_duplicates(subset=["ts", "feature_name"], keep="last")
            # Delete existing rows for (symbol, ts, feature_name) before insert
            with self.engine.begin() as conn:
                for _, row in df_long.iterrows():
                    ts_val = row["ts"]
                    # Convert pandas Timestamp to Python datetime if needed
                    if hasattr(ts_val, 'to_pydatetime'):
                        ts_val = ts_val.to_pydatetime()
                    del_query = text("""
                        DELETE FROM features
                        WHERE ts = :ts AND feature_name = :feature_name
                        """ + ("AND symbol = :symbol" if "symbol" in df_long.columns else "") + ";"
                    )
                    params = {
                        "ts": ts_val,
                        "feature_name": row["feature_name"]
                    }
                    if "symbol" in df_long.columns:
                        params["symbol"] = row["symbol"]
                    conn.execute(del_query, params)
                # Batch insert
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
        if self.engine.url.get_backend_name().startswith("sqlite"):
            symbol_list = ','.join([f"'{s}'" for s in symbols])
            feature_list = ','.join([f"'{f}'" for f in features])
            # Convert datetimes to ISO strings for SQLite string comparison
            start_ts_str = start_ts.isoformat() if hasattr(start_ts, "isoformat") else str(start_ts)
            end_ts_str = end_ts.isoformat() if hasattr(end_ts, "isoformat") else str(end_ts)
            query = text(f"""
                SELECT symbol, ts, feature_name, value
                FROM features
                WHERE symbol IN ({symbol_list})
                AND feature_name IN ({feature_list})
                AND ts >= :start_ts AND ts <= :end_ts
                ORDER BY ts ASC
            """)
            params = {
                "start_ts": start_ts_str,
                "end_ts": end_ts_str
            }
        else:
            query = text("""
                SELECT symbol, ts, feature_name, value
                FROM features
                WHERE symbol = ANY(:symbols)
                AND feature_name = ANY(:features)
                AND ts BETWEEN :start_ts AND :end_ts
                ORDER BY ts ASC
            """)
            params = {
                "symbols": symbols,
                "features": features,
                "start_ts": start_ts,
                "end_ts": end_ts
            }
        with self.engine.connect() as conn:
            import logging
            logging.warning(f"[get_features] SQL: {query}")
            logging.warning(f"[get_features] params: {params}")
            df = pd.read_sql(
                query,
                conn,
                params=params
            )
            logging.warning(f"[get_features] raw df before pivot:\n{df}")
        # Pivot to wide format
        if not df.empty:
            df_wide = df.pivot(
                index="ts",
                columns=["symbol", "feature_name"],
                values="value"
            )
        else:
            df_wide = pd.DataFrame()
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