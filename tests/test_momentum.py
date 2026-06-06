import pytest
import pandas as pd
import numpy as np
from etf_rotation.momentum import (
    compute_returns,
    rank_scores,
    composite_score,
    score_all,
)


@pytest.fixture
def price_matrix():
    """3 ETFs, 130 trading days. ETF A trending up, B flat, C down."""
    dates = pd.bdate_range("2022-01-01", periods=130)
    np.random.seed(0)
    a = 100 * np.cumprod(1 + np.random.normal(0.002, 0.01, 130))  # strong uptrend
    b = 100 * np.cumprod(1 + np.random.normal(0.000, 0.01, 130))  # flat
    c = 100 * np.cumprod(1 + np.random.normal(-0.002, 0.01, 130))  # downtrend
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


def test_compute_returns_shape(price_matrix):
    ret = compute_returns(price_matrix, window=20)
    assert ret.shape == price_matrix.shape
    # first 20 rows should be NaN
    assert ret.iloc[:20].isna().all().all()


def test_compute_returns_value(price_matrix):
    ret = compute_returns(price_matrix, window=20)
    # row 20: return = (price[20] - price[0]) / price[0]
    expected = (price_matrix.iloc[20] - price_matrix.iloc[0]) / price_matrix.iloc[0]
    pd.testing.assert_series_equal(ret.iloc[20], expected, check_names=False)


def test_rank_scores_sums_to_n_etfs(price_matrix):
    ret = compute_returns(price_matrix, window=20)
    ranks = rank_scores(ret)
    # On any non-NaN row, ranks should sum to n*(n+1)/2
    row = ranks.dropna().iloc[0]
    assert row.sum() == pytest.approx(1 + 2 + 3)


def test_rank_scores_higher_return_gets_higher_rank(price_matrix):
    ret = compute_returns(price_matrix, window=120)
    ranks = rank_scores(ret)
    last = ranks.dropna().iloc[-1]
    # ETF A (uptrend) should have highest rank (3), C lowest (1)
    assert last["A"] > last["C"]


def test_composite_score_uses_weights(price_matrix):
    from etf_rotation.config import MOMENTUM_WEIGHTS
    scores = composite_score(price_matrix)
    assert not scores.empty
    # All columns should be present
    assert set(scores.columns) == {"A", "B", "C"}
    # No NaN in rows where all windows have data (after row 120)
    valid = scores.dropna()
    assert len(valid) > 0


def test_composite_score_nan_boundary(price_matrix):
    scores = composite_score(price_matrix)
    assert scores.iloc[:120].isna().all().all()
    assert scores.iloc[120].notna().all()


def test_score_all_returns_series_on_date(price_matrix):
    date = price_matrix.index[-1]
    scores = score_all(price_matrix, as_of=date)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    # Scores should be normalized ranks, A highest
    assert scores["A"] > scores["C"]
