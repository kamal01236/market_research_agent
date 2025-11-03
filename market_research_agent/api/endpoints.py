"""API endpoints for the market research agent."""
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..store import FeatureStore
from ..config import Settings
from ..providers import YahooFinanceAdapter
from market_research_agent.features.technical import TechnicalFeatures
from ..scoring import FactorScorer

router = APIRouter()

class DataRequest(BaseModel):
    """Request model for data fetching."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")

class ScoringRequest(BaseModel):
    """Request model for factor scoring."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    factors: List[str] = Field(..., description="List of factors to compute")
    weights: Optional[Dict[str, float]] = Field(None, description="Factor weights")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")

async def get_provider():
    """Get data provider instance."""
    return YahooFinanceAdapter()

async def get_feature_store():
    """Get feature store instance."""
    # Use SQLite for dev/test; in prod, use env/config
    db_url = "sqlite:///market_features.db"
    return FeatureStore(db_url)

async def get_scorer():
    """Get factor scorer instance."""
    return FactorScorer()

@router.post("/fetch-data")
async def fetch_market_data(
    request: DataRequest,
    provider: YahooFinanceAdapter = Depends(get_provider)
) -> Dict[str, dict]:
    """Fetch market data for given symbols."""
    try:
        data = await provider.fetch_data(
            request.symbols,
            request.start_date,
            request.end_date
        )
        
        # Convert DataFrames to dict for JSON response
        return {
            sym: df.to_dict(orient="index")
            for sym, df in data.items()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching data: {str(e)}"
        )

@router.post("/compute-features")
async def compute_features(
    request: DataRequest,
    feature_store: FeatureStore = Depends(get_feature_store),
    provider: YahooFinanceAdapter = Depends(get_provider)
) -> Dict[str, dict]:
    """Compute technical features for given symbols."""
    try:
        # Fetch data
        data = await provider.fetch_data(
            request.symbols,
            request.start_date,
            request.end_date
        )
        
        # Initialize feature computer
        feature_computer = TechnicalFeatures()
        
        # Compute features
        results = {}
        for sym, df in data.items():
            # Build feature list for all available features with correct window values
            from market_research_agent.base import Feature
            def get_window(name):
                # Try to extract window from feature name (e.g., sma_20 -> 20)
                parts = name.split("_")
                if len(parts) > 1 and parts[-1].isdigit():
                    return int(parts[-1])
                if name.startswith("atr"):
                    return 14
                if name.startswith("macd"):
                    return None
                if name.startswith("beta"):
                    return 252
                if name.startswith("volume_ratio"):
                    return 20
                return 20
            feature_list = [
                Feature(
                    name=name,
                    category="technical",
                    description=f"{name} factor",
                    value_type="float",
                    frequency="1d",
                    window=get_window(name)
                )
                for name in feature_computer.feature_map.keys()
            ]
            features = feature_computer.compute_features(df, feature_list)
            await feature_store.store_features(features, symbol=sym)
            # For response, just show the computed features as dict
            results[sym] = {k: v.to_dict() for k, v in features.items()}
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing features: {str(e)}"
        )

@router.post("/compute-scores")
async def compute_scores(
    request: ScoringRequest,
    feature_store: FeatureStore = Depends(get_feature_store),
    scorer: FactorScorer = Depends(get_scorer)
) -> Dict[str, dict]:
    """Compute factor scores for given symbols."""
    try:
        import logging
        results = {}
        for sym in request.symbols:
            features_result = await feature_store.get_features(
                [sym], request.factors, request.start_date, request.end_date
            )
            # Handle both dict-of-DataFrames and DataFrame return types
            features_df = None
            if isinstance(features_result, dict):
                features_df = features_result.get(sym)
            else:
                features_df = features_result
            logging.warning(f"[compute-scores] features_df for {sym}:\n{features_df}")
            if features_df is None or features_df.empty:
                logging.warning(f"[compute-scores] features_df is empty for {sym}")
                continue
            try:
                # If columns are multi-indexed, extract symbol level
                if hasattr(features_df.columns, 'nlevels') and features_df.columns.nlevels > 1:
                    sym_df = features_df.xs(sym, axis=1, level=0)
                else:
                    sym_df = features_df
            except Exception as e:
                logging.warning(f"[compute-scores] Error extracting sym_df for {sym}: {e}")
                continue
            logging.warning(f"[compute-scores] sym_df for {sym}:\n{sym_df}")
            valid_rows = sym_df.dropna(subset=request.factors, how="any")
            logging.warning(f"[compute-scores] valid_rows for {sym}:\n{valid_rows}")
            if valid_rows.empty:
                logging.warning(f"[compute-scores] No valid rows for {sym}")
                continue
            last_row = valid_rows.iloc[-1]
            raw_values = last_row.to_dict()
            norm = scorer.normalize_zscore(raw_values)
            weights = request.weights or {f: 1.0 / len(request.factors) for f in request.factors}
            score = scorer.aggregate_score(norm, weights)
            comps = scorer.components(norm, weights)
            factors_out = {}
            for f in request.factors:
                factors_out[f] = {
                    "value": raw_values.get(f),
                    "normalized": norm.get(f),
                    "weight": weights.get(f, 0.0),
                    "contribution": comps.get(f, {}).get("contribution")
                }
            results[sym] = {
                "score": score,
                "factors": factors_out
            }
        logging.warning(f"[compute-scores] Final results: {results}")
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing scores: {str(e)}"
        )

@router.get("/available-factors")
async def get_available_factors(
    scorer: FactorScorer = Depends(get_scorer)
) -> List[dict]:
    """Get list of available factors with metadata."""
    try:
        # Example metadata; in real system, fetch from config or DB
        factors = await scorer.get_available_factors()
        meta = [
            {"name": f, "description": f"{f} factor", "window": 20 if "sma" in f or "ema" in f or "zscore" in f else 14, "value_type": "float"}
            for f in factors
        ]
        return meta
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting factors: {str(e)}"
        )