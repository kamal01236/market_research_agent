import pytest
from fastapi.testclient import TestClient
from market_research_agent.api import app
from datetime import datetime, timedelta

client = TestClient(app)

def test_fetch_market_data():
    req = {
        "symbols": ["RELIANCE"],
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-02T00:00:00"
    }
    response = client.post("/api/fetch-data", json=req)
    assert response.status_code == 200
    assert "RELIANCE" in response.json()

def test_available_factors():
    response = client.get("/api/available-factors")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_compute_scores_full_breakdown():
    # Use last 30 trading days
    today = datetime.utcnow().date()
    end_date = today - timedelta(days=1)  # yesterday
    start_date = end_date - timedelta(days=44)  # ~30 trading days (account for weekends)
    start_str = start_date.strftime("%Y-%m-%dT00:00:00")
    end_str = end_date.strftime("%Y-%m-%dT00:00:00")
    # Try RELIANCE.NS and TCS.NS, find latest date with both factors present, and score
    test_symbols = ["RELIANCE.NS", "TCS.NS"]
    found = False
    for symbol in test_symbols:
        feature_req = {
            "symbols": [symbol],
            "start_date": start_str,
            "end_date": end_str
        }
        resp = client.post("/api/compute-features", json=feature_req)
        print(f"/api/compute-features response for {symbol}: {resp.status_code} {resp.json()}")
        features_json = resp.json().get(symbol, {})
        # Extract factor time series from the nested response
        import pandas as pd
        sma_20 = features_json.get("sma_20", {}).get("sma_20", {})
        rsi_14 = features_json.get("rsi_14", {}).get("rsi_14", {})
        # Build DataFrame with aligned dates
        df = pd.DataFrame({
            "sma_20": sma_20,
            "rsi_14": rsi_14
        })
        print(f"Feature DataFrame for {symbol}:\n{df}")
        valid_rows = df.dropna(subset=["sma_20", "rsi_14"], how="any")
        print(f"valid_rows for {symbol}:\n{valid_rows}")
        if not valid_rows.empty:
            last_valid_date = valid_rows.index[-1]
            print(f"last_valid_date for {symbol}: {last_valid_date}")
            # Score only for that date
            req = {
                "symbols": [symbol],
                "factors": ["sma_20", "rsi_14"],
                "weights": {"sma_20": 0.6, "rsi_14": 0.4},
                "start_date": last_valid_date,
                "end_date": last_valid_date
            }
            response = client.post("/api/compute-scores", json=req)
            print(f"/api/compute-scores response for {symbol}: {response.status_code} {response.json()}")
            if response.status_code == 200 and symbol in response.json():
                data = response.json()
                stock = data[symbol]
                assert "score" in stock
                assert "factors" in stock
                for factor in req["factors"]:
                    assert factor in stock["factors"]
                    fdata = stock["factors"][factor]
                    assert "value" in fdata
                    assert "normalized" in fdata
                    assert "weight" in fdata
                    assert "contribution" in fdata
                found = True
                break
    assert found, "No valid data found for RELIANCE.NS or TCS.NS in test_compute_scores_full_breakdown"

def test_available_factors_metadata():
    response = client.get("/api/available-factors")
    assert response.status_code == 200
    factors = response.json()
    assert isinstance(factors, list)
    for f in factors:
        assert "name" in f
        assert "description" in f
        assert "window" in f
        assert "value_type" in f
