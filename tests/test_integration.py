import pytest
import pandas as pd
from market_research_agent.providers.yahoo import YahooFinanceAdapter
from market_research_agent.features.technical import TechnicalFeatures
from market_research_agent.store import FeatureStore
from market_research_agent.scoring import aggregate_score, normalize_zscore
from market_research_agent.base import Feature

@pytest.mark.asyncio
async def test_end_to_end_flow(tmp_path, monkeypatch):
    # Patch YahooFinanceAdapter to avoid real API call
    adapter = YahooFinanceAdapter()
    async def mock_fetch_data(symbols, start_date, end_date, **kwargs):
        return {
            s: pd.DataFrame({'open':[1,2], 'high':[2,3], 'low':[1,2], 'close':[2,3], 'volume':[1000,1100]}) for s in symbols
        }
    monkeypatch.setattr(adapter, 'fetch_data', mock_fetch_data)
    data = await adapter.fetch_data(["RELIANCE"], '2023-01-01', '2023-01-02')
    feature_computer = TechnicalFeatures()
    # Example: compute all features in the feature_map with a default window if needed
    features_list = [
        Feature(
            name,
            14 if 'rsi' in name else 20,
            description=f"Test {name}",
            value_type="float",
            frequency="1d"
        ) for name in feature_computer.feature_map.keys()
    ]
    features = feature_computer.compute_features(data, features_list)
    store = FeatureStore(f"sqlite:///{tmp_path}/test.db")
    await store.store_features(features)
    # For get_features, use symbol, feature names, and date range
    out = await store.get_features(["RELIANCE"], list(features.keys()), '2023-01-01', '2023-01-02')
    norm = normalize_zscore(out.mean().to_dict())
    score = aggregate_score(norm, {k: 1/len(norm) for k in norm})
    assert isinstance(score, float)
