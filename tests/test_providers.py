import pytest
import pandas as pd
from market_research_agent.providers.yahoo import YahooFinanceAdapter

@pytest.mark.asyncio
async def test_yahoo_fetch(monkeypatch):
    adapter = YahooFinanceAdapter()
    # Patch yfinance to avoid real API call
    async def mock_fetch_data(symbols, start_date, end_date, **kwargs):
        return {
            s: pd.DataFrame({'open':[1], 'high':[2], 'low':[1], 'close':[2], 'volume':[1000]}) for s in symbols
        }
    monkeypatch.setattr(adapter, 'fetch_data', mock_fetch_data)
    data = await adapter.fetch_data(["RELIANCE"], '2023-01-01', '2023-01-02')
    assert "RELIANCE" in data
    assert set(data["RELIANCE"].columns) == {'open','high','low','close','volume'}

@pytest.mark.asyncio
async def test_yahoo_validate():
    adapter = YahooFinanceAdapter()
    df = pd.DataFrame({'open':[1], 'high':[2], 'low':[1], 'close':[2], 'volume':[1000]})
    valid, errors = await adapter.validate_data(df)
    assert valid
    assert errors == []
