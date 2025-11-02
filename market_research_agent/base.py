"""Core interfaces and base classes for the market research agent."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class Feature:
    """Feature definition with metadata."""
    name: str
    category: str
    description: str
    value_type: str  # float, int, bool, str
    frequency: str  # 1d, 1h, etc.
    window: Optional[int] = None  # for rolling/lookback features
    requires: List[str] = None  # dependency features


class DataProviderAdapter(ABC):
    """Abstract base class for data provider adapters."""
    
    @abstractmethod
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for given symbols and date range."""
        pass

    @abstractmethod
    async def validate_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """Validate fetched data quality."""
        pass

    @abstractmethod
    async def get_metadata(self) -> Dict[str, Any]:
        """Get provider metadata (rate limits, etc)."""
        pass


class FeatureComputer(ABC):
    """Abstract base class for feature computation."""

    @abstractmethod
    def compute_features(
        self,
        data: Dict[str, pd.DataFrame],
        features: List[Feature]
    ) -> Dict[str, pd.DataFrame]:
        """Compute requested features from input data."""
        pass

    @abstractmethod
    def validate_features(
        self,
        features: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate computed features."""
        pass


class ModelManager(ABC):
    """Abstract base class for model management."""

    @abstractmethod
    async def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model and return weights/artifacts."""
        pass

    @abstractmethod
    async def get_latest_weights(self) -> Dict[str, float]:
        """Get latest model weights."""
        pass

    @abstractmethod
    async def validate_weights(
        self,
        weights: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Validate weight constraints."""
        pass


class ScoreComputer(ABC):
    """Abstract base class for score computation."""

    @abstractmethod
    def compute_scores(
        self,
        features: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Compute scores and explanations."""
        pass

    @abstractmethod
    def validate_scores(
        self,
        scores: pd.Series,
        threshold: float = None
    ) -> Tuple[bool, List[str]]:
        """Validate computed scores."""
        pass