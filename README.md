Market Research Agent - Prototype

This repository contains a prototype scaffold for a factor-scoring market research agent targeted at the Indian equity market.

Structure:
- `market_research_agent/` - core package (config, features, scoring)
- `config.yaml` - example runtime config (universe, providers, cadence, weights)
- `tests/` - pytest-based unit tests for features and scoring

How to run tests (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```
