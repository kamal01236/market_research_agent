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
    return FeatureStore()

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
            features = await feature_computer.compute(df)
            await feature_store.store_features(sym, features)
            results[sym] = features.to_dict(orient="index")
        
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
        results = {}
        for sym in request.symbols:
            # Get features from store
            features = await feature_store.get_features(
                sym,
                request.start_date,
                request.end_date
            )
            
            if features is None or features.empty:
                continue
                
            # Compute scores
            scores = await scorer.compute_scores(
                features,
                request.factors,
                request.weights
            )
            results[sym] = scores.to_dict(orient="index")
            
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing scores: {str(e)}"
        )

@router.get("/available-factors")
async def get_available_factors(
    scorer: FactorScorer = Depends(get_scorer)
) -> List[str]:
    """Get list of available factors."""
    try:
        return await scorer.get_available_factors()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting factors: {str(e)}"
        )