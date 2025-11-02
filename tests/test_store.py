import pytest
import pandas as pd
from market_research_agent.store import FeatureStore

@pytest.mark.asyncio
async def test_store_and_get_features(tmp_path):
    # Use a temporary SQLite DB for testing
    db_url = f"sqlite:///{tmp_path}/test.db"
    store = FeatureStore(db_url)
    df = pd.DataFrame({
        'sma_20': [1.1, 1.2, 1.3],
        'rsi_14': [50, 55, 60]
    }, index=pd.date_range('2023-01-01', periods=3))
    df.index.name = "ts"
    # Split into dict of feature_name: DataFrame
    features = {col: df[[col]].copy() for col in df.columns}
    await store.store_features(features, symbol='RELIANCE')
    out = await store.get_features(["RELIANCE"], list(features.keys()), '2023-01-01', '2023-01-03')
    assert not out.empty
    assert 'sma_20' in out.columns.get_level_values(-1)
    assert 'rsi_14' in out.columns.get_level_values(-1)
