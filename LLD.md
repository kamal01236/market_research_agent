## Market Research Agent — Low-Level Design (Factor-Scoring Framework)

This document defines a detailed low-level design to implement a factor-scoring quantitative agent for Indian equity markets. The agent computes multi-category factor scores for every stock, aggregates them into ranked signals, stores analysis/results, and learns from past outcomes daily. It supports pre-market, intraday, and post-market runs.

### Goals / Success Criteria
- Provide per-stock daily scores that quantify likelihood of positive short/medium-term movement (configurable horizons).
- Support scheduled runs: pre-market (before open), during-market (intraday periodic), and post-market (after close/backtest update).
- Persist raw data, features, scores, model versions, and outcomes for continual learning.
- Be explainable (per-factor contribution) and auditable.

### High-level Architecture

Components:
- Data Ingestion Layer
	- Price/time-series (OHLCV) from exchanges or data providers
	- Fundamentals (annual & quarterly financials)
	- Corporate actions (splits, dividends)
	- Market data: options chains, F&O, implied vol, orderbook snapshots (if available)
	- Macro indicators (rates, CPI, GDP) and commodity prices
	- News & social sentiment (news APIs, X/Twitter, Reddit-like sources)

- ETL / Feature Engineering
	- Cleaning, alignment, adjustment (corporate actions)
	- Factor computation (per the categories below)

- Feature Store (time-series optimized)
	- Raw tables and derived features for rapid retrieval

- Scoring Engine
	- Normalization + weighting + aggregation -> factor score
	- Config-driven weights (JSON) and dynamic re-weighting via learning

- Learning & Backtest Module
	- Offline/backtest to learn weights, validate signals; produce model artifacts

- API / Dashboard
	- Expose scores, factor explanations, watchlists, historical analyses

- Storage & Ops
	- DB (Postgres/Timescale), object store (S3 or local), cache (Redis), message bus (Kafka/RabbitMQ optional), scheduler (Airflow/Cron/Celery)

### Tech Stack (recommended)
- Language: Python 3.11+
- Data: PostgreSQL (TimescaleDB if heavy time-series), S3 (or local FS) for raw files
- Orchestration: Airflow or Prefect for complex schedules; Cron/Celery for simpler setups
- ML: scikit-learn, xgboost/lightgbm, optuna for hyperparam tuning; SHAP for explainability
- APIs: FastAPI for serving; Streamlit/React for dashboards
- Containerization: Docker; k8s optional for scale

### Data Sources (Indian-market focused)
- Price / OHLCV
	- NSE/BSE official feeds (preferred, paid)
	- Yahoo Finance / Alpha Vantage / Tiingo (free and limited) — verify ticker mappings for Indian symbols
- Fundamentals
	- Exchange disclosures, financial statements (nseindia live filings), public datasets (screener.in, moneycontrol scraped with care), commercial providers
- Options/F&O
	- NSE option chain API or exchange data for Greeks and open interest
- News & Social
	- NewsAPI, Google News scraping, X/Twitter API, public finance forums
- Macro & Commodities
	- RBI, Ministry of Statistics, Bloomberg/Refinitiv for paid options

Assumption: A paid, reliable market data subscription is recommended for production (clean OHLCV, corporate actions, and option chains). The design supports fallback to free sources for prototyping.

### Data Schema (key tables)

Note: use TimescaleDB hypertables for OHLCV and feature timeseries.

- raw_prices (symbol, ts, open, high, low, close, volume, adj_close, source)
- raw_financials (symbol, period_end, fiscal_type, field_name, value, source)
- corporate_actions (symbol, date, type, details)
- option_chain_snapshots (symbol, ts, strikes[], calls[], puts[])
- sentiment_raw (symbol, ts, source, text, sentiment_score)

- features (symbol, ts, feature_name, value)
- scores (symbol, ts, score_type, score_value, components JSON)
- analysis_runs (run_id, run_type, start_ts, end_ts, config_id, status)
- models (model_id, version, weights_blob, metadata, created_ts)
- outcomes (symbol, ts, horizon, return, label)

### Factor List & Derived Features

Map the user's high-level categories to concrete computed features:

1) Fundamental Factors (company-specific)
- Financial Performance
	- revenue_ttm, revenue_yoy, eps_ttm, eps_yoy, gross_margin, op_margin, net_margin
	- fcf_ttm, fcf_margin, operating_cashflow
	- roe, roa, leverage_ratio (total_debt / total_equity)
	- dividend_yield, buyback_announcements (binary), payout_ratio
- Valuation
	- pe_ttm, ps_ttm, pb, ev_ebitda
	- peg (pe / earnings_growth_rate)
	- implied_dcf_gap = (market_price - dcf_intrinsic) / dcf_intrinsic
	- forward_pe, normalized_pe
- Management & Strategy
	- ceo_change_12m (binary), cfo_change_12m, insider_trading_score, r_and_d_to_revenue

2) Technical Factors
- Price & Volume
	- sma_20, sma_50, sma_200, ema_20
	- crossover_signals (20/50, 50/200)
	- volume_rel = volume / average_volume_20
	- rsi_14, macd, price_momentum_1w/1m/3m
	- ATR (volatility), beta (against NIFTY)
- Microstructure
	- bid_ask_spread, depth_ratio (where level2 available), short_interest_ratio

3) Macro & External
- interest_rate_change_3m, cpi_yoy, gdp_qoq, fx_inr_usd_change, oil_price_change

4) Sentiment
- news_sentiment_24h, social_sentiment_24h, sentiment_volume_ratio

5) Event-Driven
- earnings_surprise_pct, next_earnings_date_proximity, major_announcement_flag

6) Sector & Industry
- sector_relative_strength, sector_momentum, industry_pe

7) Market-wide & Liquidity
- etf_inflow_ratio, index_inclusion_flag, avg_daily_turnover

8) Quant & Options
- options_oi_change, put_call_ratio, gamma_exposure_estimate, options_iv_surface_slope

Each feature computed with time-windows (e.g., TTM, 3m, 12m) and tagged with confidence/quality metadata.

### Normalization & Scoring

Normalization options (choose/configurable per factor):
- z-score across universe (mean/std) using rolling window (e.g., 90d)
- rank percentile (0-1) across universe
- min-max scaling with clipping to avoid outliers

Score composition:
- Each factor gets a normalized score s_i in [-1,1] or [0,1] depending on transform.
- Weights w_i are defined per factor and optionally grouped by category. Weights sum to 1 (or normalized after aggregation).

Aggregate score computation (config-driven):

score(symbol, t) = sum_i w_i * s_i(symbol, t)

Where w_i are loaded from a configuration file (JSON) and can be static or produced by the Learning Module.

Component explanations stored with each score: components JSON = [{"factor":"pe_ttm","weight":0.05,"value":12.4,"norm":0.7}] so users can drill into what drove the score.

Example weight config (JSON):

{
	"categories": {
		"fundamental": 0.35,
		"technical": 0.25,
		"macro": 0.10,
		"sentiment": 0.10,
		"events": 0.10,
		"liquidity": 0.05,
		"quant": 0.05
	},
	"factors": {
		"eps_yoy": 0.12,
		"revenue_yoy": 0.08,
		"pe_ttm": 0.05,
		"sma_50_200_crossover": 0.06,
		"rsi_14": 0.04,
		"news_sentiment_24h": 0.05
	}
}

Weights can be constrained (e.g., no negative weights for certain categories) and normalized automatically by the engine.

### Learning Weights (daily learning loop)

Approach A (supervised ranking):
- Create training dataset where input X = factor vector at time t, label Y = realized return over horizon H (e.g., next 5 days) or binary target (top decile returns = 1).
- Train a model (e.g., gradient boosting) to predict returns or probability of positive outcome. Extract feature importances or prediction coefficients and convert to weights.

Approach B (constrained optimization):
- Optimize weights w to maximize backtest objective (e.g., information ratio, Sharpe, cumulative return) subject to risk constraints. Use CV or walk-forward.

Implementation details:
- Keep separate training windows (rolling) and a validation window to avoid lookahead.
- Regularize weights to avoid overfitting; use L1/L2 or limit number of active factors.
- Produce model artifacts (weights_blob) stored in `models` table with versioning.

Online learning & adaptation:
- Recompute weights weekly or monthly, and optionally allow daily small adjustments by an online learner with very limited step-size.

### Evaluation & Backtesting

Backtest engine must simulate trading realistically:
- Use transaction cost model (slippage, spread) and position sizing (equal-weight, volatility scaled, risk-parity).
- Evaluate metrics: cumulative return, annualized return, volatility, Sharpe, max drawdown, hit rate, precision@k, NDCG for ranking.
- Walk-forward/backtest with rolling windows and report performance by sector and market regimes (bull/bear).

Validation checks:
- Check for lookahead bias and survivorship bias.
- Use only data available up to decision time.

### Scheduling (pre/during/post-market flows)

- Pre-market (before market open):
	1. Ingest overnight data and news
	2. Update fundamentals if new filings
	3. Recompute features and scores
	4. Run models to update weights (if scheduled for that day)
	5. Store `scores` with `run_type=pre-market`

- During-market (intraday periodic, e.g., every 15m or 60m):
	1. Stream/ingest latest tick/1m OHLCV
	2. Recompute intraday features (momentum, volume spikes, orderbook imbalance)
	3. Recompute and publish intraday scores for monitoring and tactical signals

- Post-market (after close):
	1. Compute daily returns and outcomes labels
	2. Run backtest updates and retrain weights (offline)
	3. Persist learning artifacts and produce daily report (performance and model changes)

### APIs and Data Contracts

REST API endpoints (FastAPI example):
- GET /v1/scores?date=YYYY-MM-DD&universe=nse500 -> list of scores and top-k
- GET /v1/scores/{symbol}?date=... -> score and components for symbol
- GET /v1/features/{symbol}?feature=pe_ttm&start=&end= -> feature time series
- POST /v1/run -> trigger manual run, with payload {"run_type":"pre-market","config_id":x}

Payload formats are JSON; components use compact JSON for factor breakdown.

### Storage & Retention
- Raw ticks/1m pricedata: retain 6-12 months in DB, archive older to object store
- Daily OHLCV & computed features: retain indefinitely in TimescaleDB
- Models & artifacts: retain versions with metadata
- Retention policy configurable via S3 lifecycle

### Observability, Monitoring & Alerts
- Logging: structured logs for ingestion failures and data quality flags
- Metrics: run durations, data freshness, number of symbols processed, missing data ratios
- Alerts: data source outage, major drop in backtest performance, model drift detection

### Security & Compliance
- Protect API keys and credentials in secrets manager
- Respect exchange data licensing; do not re-distribute paid data
- Sanitize scraped sources and conform to robots.txt where applicable

### Edge Cases & Quality Controls
- Missing fundamentals: impute conservatively or mark factor as low-confidence
- Delisted symbols: archive and stop scoring
- Corporate actions: always adjust historical prices for splits/dividends
- Outliers: clip extreme z-scores and use robust scaling when necessary
- Surviving/selection bias: ensure backtests include delisted companies or use survivorship-corrected universe

### Tests & Validation
- Unit tests for feature computations (sample inputs -> expected outputs)
- Integration tests for ingestion pipeline using small recorded snapshots
- Backtest regression tests to ensure metrics don't change unexpectedly

### Explainability & Reporting
- Store component contributions for each symbol-day for auditing
- Provide SHAP or feature-importance snapshots for model-based weight learners
- Daily digest report: top/bottom ranked stocks with reasons, model changes, market context

### Minimal Viable Implementation Plan
1. Prototype data ingestion for daily OHLCV for NSE 200 symbols (Yahoo/AlphaVantage or exchange feed)
2. Implement core feature computations (fundamental ratios, SMA, RSI, momentum)
3. Implement normalization, static weight scoring engine and store scores
4. Build simple FastAPI endpoints to serve scores and explanations
5. Implement backtest harness and train a small supervised learner for weights
6. Add scheduled runs (pre/during/post) and persistence

### Implementation Timeline (suggested)
- Week 1: Data ingestion + feature prototypes for limited universe
- Week 2: Scoring engine + simple UI/API
- Week 3: Backtest framework + supervised weighting prototype
- Week 4: Explainability, reporting, and QA tests

### Assumptions & Limitations
- High-quality exchange or paid data is assumed for production accuracy.
- Intraday orderbook-level features are optional and require access to Level-2 feeds.
- Legal/licensing responsibilities for scraping or storing exchange data are the user's responsibility.

### Next Steps (recommended)
1. Confirm target universe (NSE200, NIFTY500, custom watchlist).
2. Pick data provider(s) for prices and fundamentals (paid vs free).
3. Decide execution cadence (every 15m intraday or hourly) and retention policy.
4. If you want, I can scaffold a starter prototype (Python package, basic ingestion + scoring + FastAPI) in this repo.

## Appendix: Example scoring pseudocode

1. Load universe symbols and latest data
2. For each symbol, compute features f_i
3. Normalize features -> s_i
4. Load weights w_i (from config or latest trained model)
5. Compute score = sum(w_i * s_i)
6. Save score row with components for explainability

## Appendix: Key Evaluation Metrics for Production Monitoring
- Precision@K, Recall@K, NDCG
- Daily/Weekly return of top-k portfolio (net of costs)
- Turnover, average holding period
- Model stability: KL divergence between weight distributions across retrains

---

File created/updated to implement the user's requested LLD for the factor-scoring agent.
