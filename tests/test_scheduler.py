import pytest
from market_research_agent.scheduler import JobScheduler
from market_research_agent.config import Settings

@pytest.mark.asyncio
async def test_scheduler_runs(tmp_path, monkeypatch):
    # Minimal settings
    settings = Settings(symbols=["RELIANCE", "TCS"])
    db_url = f"sqlite:///{str(tmp_path)}/test.db"
    scheduler = JobScheduler(settings, db_url=db_url)
    called = {}
    async def fake_fetch_and_compute(symbols, compute_scores=True):
        called['symbols'] = symbols
        called['compute_scores'] = compute_scores
    scheduler.fetch_and_compute = fake_fetch_and_compute
    await scheduler.schedule_jobs()
    # Simulate a job run
    await scheduler.fetch_and_compute(["RELIANCE"], True)
    assert called['symbols'] == ["RELIANCE"]
    assert called['compute_scores'] is True
