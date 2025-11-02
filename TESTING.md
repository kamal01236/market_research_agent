# Testing Strategy

## 1. Unit Tests

### 1.1 Data Provider Tests
```python
class TestYahooFinanceAdapter:
    @pytest.mark.asyncio
    async def test_fetch_data():
        # Test successful data fetch
        adapter = YahooFinanceAdapter()
        data = await adapter.fetch_data(
            symbols=["RELIANCE.NS", "TCS.NS"],
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31)
        )
        assert len(data) == 2
        assert all(isinstance(df, pd.DataFrame) for df in data.values())
        
    @pytest.mark.asyncio
    async def test_data_validation():
        adapter = YahooFinanceAdapter()
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        is_valid, errors = await adapter.validate_data(df)
        assert is_valid
        assert len(errors) == 0
```

### 1.2 Feature Computation Tests
```python
class TestTechnicalFeatures:
    def test_sma():
        calculator = TechnicalFeatures()
        df = pd.DataFrame({
            'close': range(100),
            'volume': range(100)
        })
        features = calculator.compute(df)
        assert 'sma_20' in features.columns
        assert 'sma_50' in features.columns
        assert len(features) == len(df)
        
    def test_rsi():
        calculator = TechnicalFeatures()
        # Test RSI bounds
        features = calculator.compute(df)
        assert all(0 <= x <= 100 for x in features['rsi_14'].dropna())
```

### 1.3 Feature Store Tests
```python
class TestFeatureStore:
    @pytest.mark.asyncio
    async def test_store_features():
        store = FeatureStore()
        features = pd.DataFrame({
            'sma_20': range(10),
            'rsi_14': range(10)
        })
        await store.store_features('RELIANCE', features)
        
        # Verify retrieval
        stored = await store.get_features(
            'RELIANCE',
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31)
        )
        assert not stored.empty
        assert all(col in stored.columns for col in features.columns)
```

### 1.4 Scoring Engine Tests
```python
class TestFactorScorer:
    @pytest.mark.asyncio
    async def test_score_computation():
        scorer = FactorScorer()
        features = pd.DataFrame({
            'sma_20': range(10),
            'rsi_14': [30, 40, 50, 60, 70] * 2
        })
        
        scores = await scorer.compute_scores(
            features,
            factors=['trend', 'momentum'],
            weights={'trend': 0.6, 'momentum': 0.4}
        )
        
        assert len(scores) == len(features)
        assert all(-1 <= x <= 1 for x in scores)
```

## 2. Integration Tests

### 2.1 End-to-End Data Flow
```python
@pytest.mark.integration
async def test_data_to_scores_flow():
    # Setup components
    provider = YahooFinanceAdapter()
    feature_computer = TechnicalFeatures()
    store = FeatureStore()
    scorer = FactorScorer()
    
    # Test full flow
    data = await provider.fetch_data(
        symbols=["RELIANCE.NS"],
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 31)
    )
    
    features = await feature_computer.compute(data['RELIANCE.NS'])
    await store.store_features('RELIANCE', features)
    
    scores = await scorer.compute_scores(
        features,
        factors=['trend', 'momentum', 'value']
    )
    
    assert not scores.empty
    assert scores.index.equals(features.index)
```

### 2.2 API Endpoint Tests
```python
from fastapi.testclient import TestClient

class TestAPI:
    def test_fetch_scores(test_client: TestClient):
        response = test_client.get("/v1/scores/RELIANCE")
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        
    def test_compute_features(test_client: TestClient):
        request = {
            "symbols": ["RELIANCE", "TCS"],
            "start_date": "2025-01-01",
            "end_date": "2025-01-31"
        }
        response = test_client.post("/v1/compute-features", json=request)
        assert response.status_code == 200
```

### 2.3 Job Scheduler Tests
```python
class TestJobScheduler:
    @pytest.mark.asyncio
    async def test_market_open_job():
        scheduler = JobScheduler(test_settings)
        await scheduler.fetch_and_compute(
            symbols=["RELIANCE", "TCS"],
            compute_scores=True
        )
        
        # Verify data was stored
        store = FeatureStore()
        features = await store.get_features('RELIANCE', datetime.now())
        assert not features.empty
```

## 3. Performance Tests

### 3.1 Load Testing
```python
@pytest.mark.performance
async def test_batch_processing():
    provider = YahooFinanceAdapter()
    symbols = ["RELIANCE.NS"] * 100  # Test with 100 copies
    
    start_time = time.time()
    data = await provider.fetch_data(
        symbols=symbols,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 31)
    )
    duration = time.time() - start_time
    
    assert duration < 30  # Should complete in 30 seconds
```

### 3.2 Concurrency Tests
```python
@pytest.mark.asyncio
async def test_concurrent_feature_computation():
    feature_computer = TechnicalFeatures()
    data = generate_test_data(100)  # 100 symbols
    
    # Test parallel computation
    tasks = []
    for symbol, df in data.items():
        task = asyncio.create_task(feature_computer.compute(df))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    assert len(results) == 100
```

## 4. Quality Control Tests

### 4.1 Data Quality Tests
```python
class TestDataQuality:
    def test_price_consistency():
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 97],
            'close': [101, 102]
        })
        
        # Check price relationships
        assert all(df['high'] >= df['low'])
        assert all(df['high'] >= df['open'])
        assert all(df['high'] >= df['close'])
        assert all(df['low'] <= df['open'])
        assert all(df['low'] <= df['close'])
```

### 4.2 Feature Quality Tests
```python
def test_feature_quality():
    features = pd.DataFrame({
        'sma_20': range(100),
        'rsi_14': range(100),
        'momentum': range(100)
    })
    
    # Check for NaN values
    assert not features.isna().any().any()
    
    # Check for infinity
    assert not np.isinf(features).any().any()
    
    # Check bounds for normalized features
    assert all(-1 <= x <= 1 for x in features['momentum'])
```

## 5. System Tests

### 5.1 Configuration Tests
```python
def test_config_loading():
    settings = Settings.from_yaml('config.yml')
    assert len(settings.symbols) > 0
    assert settings.db.host is not None
    assert settings.redis.port > 0
```

### 5.2 Error Recovery Tests
```python
@pytest.mark.asyncio
async def test_provider_fallback():
    primary = MockProvider(should_fail=True)
    fallback = MockProvider(should_fail=False)
    pipeline = IngestionPipeline([primary, fallback])
    
    result = await pipeline.ingest(['RELIANCE'])
    assert result.success
    assert result.provider == fallback
```

## 6. Backtesting Tests

### 6.1 Strategy Tests
```python
class TestBacktesting:
    def test_backtest_performance():
        engine = BacktestEngine()
        strategy = MockStrategy()
        
        results = engine.run_backtest(
            strategy,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31)
        )
        
        assert results.sharpe_ratio > 0
        assert results.max_drawdown < 0.3
```

### 6.2 Model Tests
```python
def test_model_weights():
    model = ModelManager()
    X = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100)
    })
    y = pd.Series(range(100))
    
    weights = model.train_model(X, y)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w >= 0 for w in weights.values())
```

## 7. Monitoring Tests

### 7.1 Alert Tests
```python
def test_data_freshness_alert():
    monitor = DataMonitor()
    status = monitor.check_data_freshness('RELIANCE')
    assert status.delay_minutes < 15  # Data should be fresh
```

### 7.2 Performance Metric Tests
```python
def test_score_predictive_power():
    evaluator = ScoreEvaluator()
    metrics = evaluator.compute_metrics(
        predictions=generated_scores,
        actuals=market_returns
    )
    assert metrics['ic'] > 0.1  # Information Coefficient
    assert metrics['hit_rate'] > 0.52  # Win rate
```

## Test Configuration

```yaml
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    system: System tests
addopts = --verbose --cov=market_research_agent --cov-report=term-missing
asyncio_mode = auto
```

## Test Data Management

```python
@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    return pd.DataFrame({
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

@pytest.fixture
def mock_feature_store():
    """Create an in-memory feature store for testing."""
    return MockFeatureStore()
```

## CI/CD Pipeline Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        ports:
          - 5432:5432
          
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install '.[test,dev]'
        
    - name: Run tests
      run: |
        pytest --cov=market_research_agent --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

This testing strategy ensures:
1. Every component is tested in isolation (unit tests)
2. Components work together correctly (integration tests)
3. System can handle expected load (performance tests)
4. Data quality is maintained (quality control tests)
5. Error handling works as expected (system tests)
6. Backtesting and model training are reliable
7. Monitoring and alerts function correctly

Key points for implementation:
1. Write tests first (TDD) for new features
2. Maintain high test coverage (aim for >80%)
3. Include all test categories in CI/CD pipeline
4. Use realistic test data fixtures
5. Test both success and failure scenarios
6. Monitor test performance metrics