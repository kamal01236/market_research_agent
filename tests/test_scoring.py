from market_research_agent import scoring


def test_normalize_zscore():
    vals = {"a": 1.0, "b": 2.0, "c": 3.0}
    norm = scoring.normalize_zscore(vals)
    # mean should be 0
    mean = sum(norm.values()) / len(norm)
    assert abs(mean) < 1e-9


def test_aggregate_score():
    vals = {"x": 1.0, "y": 2.0}
    norm = scoring.normalize_zscore(vals)
    weights = {"x": 0.6, "y": 0.4}
    score = scoring.aggregate_score(norm, weights)
    # score should be finite
    assert isinstance(score, float)


def test_components_shape():
    vals = {"x": 1.0, "y": 3.0}
    norm = scoring.normalize_zscore(vals)
    weights = {"x": 0.5, "y": 0.5}
    comps = scoring.components(norm, weights)
    assert set(comps.keys()) == set(vals.keys())
    for v in comps.values():
        assert "norm" in v and "weight" in v and "contribution" in v
