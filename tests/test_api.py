import pytest
from fastapi.testclient import TestClient
from market_research_agent.api import app

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
