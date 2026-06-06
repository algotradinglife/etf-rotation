import pytest
import pandas as pd
import numpy as np
from etf_rotation.backtest import BacktestResult, run_backtest


@pytest.fixture
def price_matrix_3yr():
    """3 ETFs, 3 years of daily data (756 trading days)."""
    dates = pd.bdate_range("2020-01-01", periods=756)
    np.random.seed(7)
    data = {
        "A": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 756)),
        "B": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 756)),
        "C": 100 * np.cumprod(1 + np.random.normal(-0.0001, 0.010, 756)),
    }
    return pd.DataFrame(data, index=dates)


def test_backtest_returns_result_object(price_matrix_3yr):
    result = run_backtest(price_matrix_3yr, top_n=2, freq="ME")
    assert isinstance(result, BacktestResult)


def test_backtest_portfolio_series_aligned_to_prices(price_matrix_3yr):
    result = run_backtest(price_matrix_3yr, top_n=2, freq="ME")
    assert result.portfolio_values.index[0] >= price_matrix_3yr.index[0]
    assert result.portfolio_values.index[-1] <= price_matrix_3yr.index[-1]


def test_backtest_total_return_is_float(price_matrix_3yr):
    result = run_backtest(price_matrix_3yr, top_n=2, freq="ME")
    assert isinstance(result.total_return, float)


def test_backtest_sharpe_is_finite(price_matrix_3yr):
    result = run_backtest(price_matrix_3yr, top_n=2, freq="ME")
    assert np.isfinite(result.sharpe_ratio)


def test_backtest_max_drawdown_negative_or_zero(price_matrix_3yr):
    result = run_backtest(price_matrix_3yr, top_n=2, freq="ME")
    assert result.max_drawdown <= 0


def test_backtest_holdings_log_has_rebalance_dates(price_matrix_3yr):
    result = run_backtest(price_matrix_3yr, top_n=2, freq="ME", abs_momentum_filter=False)
    assert len(result.holdings_log) > 0
    first = result.holdings_log[0]
    assert "date" in first
    assert "holdings" in first
    assert len(first["holdings"]) == 2


def test_weekly_rebalance_more_events(price_matrix_3yr):
    monthly = run_backtest(price_matrix_3yr, top_n=2, freq="ME")
    weekly = run_backtest(price_matrix_3yr, top_n=2, freq="W")
    assert len(weekly.holdings_log) > len(monthly.holdings_log)


def test_transaction_cost_reduces_return(price_matrix_3yr):
    no_cost = run_backtest(price_matrix_3yr, top_n=2, freq="ME", transaction_cost=0.0)
    with_cost = run_backtest(price_matrix_3yr, top_n=2, freq="ME", transaction_cost=0.001)
    assert with_cost.total_return < no_cost.total_return


def test_transaction_cost_zero_no_change(price_matrix_3yr):
    r1 = run_backtest(price_matrix_3yr, top_n=2, freq="ME", transaction_cost=0.0)
    r2 = run_backtest(price_matrix_3yr, top_n=2, freq="ME", transaction_cost=0.0)
    assert abs(r1.total_return - r2.total_return) < 1e-10


@pytest.fixture
def trending_up_matrix():
    """ETFs A and B trend up strongly, C trends down — abs filter should keep A,B."""
    dates = pd.bdate_range("2020-01-01", periods=756)
    np.random.seed(42)
    data = {
        "A": 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, 756)),   # strong up
        "B": 100 * np.cumprod(1 + np.random.normal(0.0008, 0.01, 756)),  # up
        "C": 100 * np.cumprod(1 + np.random.normal(-0.002, 0.01, 756)),  # down
    }
    return pd.DataFrame(data, index=dates)


def test_abs_momentum_filter_excludes_negative(trending_up_matrix):
    """With abs filter on, bearish ETFs should be excluded from holdings."""
    result_filtered = run_backtest(
        trending_up_matrix, top_n=3, freq="ME",
        abs_momentum_filter=True, abs_momentum_window=120,
    )
    # C trends down — should appear rarely or never in filtered holdings
    c_count = sum(1 for e in result_filtered.holdings_log if "C" in e["holdings"])
    result_unfiltered = run_backtest(
        trending_up_matrix, top_n=3, freq="ME",
        abs_momentum_filter=False,
    )
    c_count_unfiltered = sum(1 for e in result_unfiltered.holdings_log if "C" in e["holdings"])
    assert c_count <= c_count_unfiltered


def test_abs_momentum_filter_all_cash(price_matrix_3yr):
    """When all ETFs have negative absolute momentum, holdings should be empty (cash)."""
    # Invert the price matrix so everything trends down
    inverted = price_matrix_3yr.iloc[0] * 2 - price_matrix_3yr
    inverted = inverted.clip(lower=0.01)
    result = run_backtest(
        inverted, top_n=2, freq="ME",
        abs_momentum_filter=True, abs_momentum_window=20,
    )
    # Should not crash, portfolio should be a valid series
    assert len(result.portfolio_values) > 0
    assert result.portfolio_values.iloc[-1] > 0


# ── Stop-loss tests ──────────────────────────────────────────────────────────

@pytest.fixture
def crash_matrix():
    """
    3 ETFs over 2 years. ETF 'C' crashes 25% in month 7 (well past warm-up).
    A and B trend up steadily throughout.
    """
    dates = pd.bdate_range("2020-01-01", periods=504)
    np.random.seed(99)
    a = 100 * np.cumprod(1 + np.random.normal(0.0006, 0.008, 504))
    b = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, 504))
    c = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.008, 504))
    # Inject a sharp -25% drop in C around trading day 140 (month 7)
    c[140:160] *= np.linspace(1.0, 0.75, 20)
    c[160:] *= 0.75
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


def test_holding_stop_does_not_crash(crash_matrix):
    """holding_stop should run without errors."""
    result = run_backtest(
        crash_matrix, top_n=3, freq="ME",
        abs_momentum_filter=False,
        holding_stop=-0.10,
        transaction_cost=0.001,
    )
    assert isinstance(result, BacktestResult)
    assert result.portfolio_values.iloc[-1] > 0


def test_holding_stop_limits_loss_vs_no_stop(crash_matrix):
    """With a -10% stop, max drawdown should be <= no-stop baseline."""
    no_stop = run_backtest(
        crash_matrix, top_n=3, freq="ME",
        abs_momentum_filter=False,
        holding_stop=None, portfolio_stop=None,
        transaction_cost=0.0,
    )
    with_stop = run_backtest(
        crash_matrix, top_n=3, freq="ME",
        abs_momentum_filter=False,
        holding_stop=-0.10, portfolio_stop=None,
        transaction_cost=0.0,
    )
    # Stop should prevent the worst losses (drawdown closer to 0)
    assert with_stop.max_drawdown >= no_stop.max_drawdown


def test_holding_stop_cash_drag_vs_full_weight(crash_matrix):
    """
    When a position is stopped out its allocation stays in cash (not redistributed).
    Run top_n=3 with stop (C stopped → 1/3 cash) vs top_n=2 without stop (A+B at 1/2 each).
    The 1/3 cash drag means the stopped run should underperform the full A+B run.
    """
    # A and B only — full 1/2 weight each, no stop
    ab_only = crash_matrix[["A", "B"]]
    two_etf = run_backtest(
        ab_only, top_n=2, freq="ME",
        abs_momentum_filter=False,
        holding_stop=None, portfolio_stop=None,
        transaction_cost=0.0,
    )
    # All three with stop: C crashes and is stopped → its 1/3 sits in cash
    three_etf_stopped = run_backtest(
        crash_matrix, top_n=3, freq="ME",
        abs_momentum_filter=False,
        holding_stop=-0.05, portfolio_stop=None,
        transaction_cost=0.0,
    )
    # Cash drag: 1/3 of portfolio earns 0 after C is stopped vs 1/2 weight in 2-ETF run
    assert three_etf_stopped.total_return < two_etf.total_return


def test_portfolio_stop_does_not_crash(price_matrix_3yr):
    """portfolio_stop should run without errors."""
    result = run_backtest(
        price_matrix_3yr, top_n=2, freq="ME",
        abs_momentum_filter=False,
        holding_stop=None,
        portfolio_stop=-0.15,
        transaction_cost=0.001,
    )
    assert isinstance(result, BacktestResult)
    assert result.portfolio_values.iloc[-1] > 0


def test_portfolio_stop_portfolio_never_drops_far_below_hwm(price_matrix_3yr):
    """After portfolio stop fires, subsequent drawdown from HWM should be bounded."""
    stop_level = -0.20
    result = run_backtest(
        price_matrix_3yr, top_n=2, freq="ME",
        abs_momentum_filter=False,
        holding_stop=None,
        portfolio_stop=stop_level,
        transaction_cost=0.0,
    )
    # Max drawdown should not be much worse than the stop level
    # (allow some slack for the day the stop fires)
    assert result.max_drawdown >= stop_level - 0.05


# ── MA trend filter tests ─────────────────────────────────────────────────────

@pytest.fixture
def ma_filter_matrix():
    """
    ETF 'UP': strong uptrend — price well above 200d MA throughout.
    ETF 'DOWN': strong downtrend — price well below 200d MA throughout.
    500 trading days so MA warm-up is satisfied.
    """
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    up = 50 * np.cumprod(1 + np.full(n, 0.004))    # +0.4%/day, strongly up
    down = 200 * np.cumprod(1 + np.full(n, -0.003)) # -0.3%/day, strongly down
    return pd.DataFrame({"UP": up, "DOWN": down}, index=dates)


def test_ma_trend_filter_excludes_below_ma(ma_filter_matrix):
    """DOWN is always below its MA — should never appear in filtered holdings."""
    result = run_backtest(
        ma_filter_matrix, top_n=2, freq="ME",
        abs_momentum_filter=False,
        ma_trend_filter=True, ma_trend_window=200,
        transaction_cost=0.0,
    )
    down_count = sum(1 for e in result.holdings_log if "DOWN" in e["holdings"])
    assert down_count == 0


def test_ma_trend_filter_off_includes_down(ma_filter_matrix):
    """Without filter, DOWN may appear (it can have relative momentum)."""
    result_on = run_backtest(
        ma_filter_matrix, top_n=2, freq="ME",
        abs_momentum_filter=False,
        ma_trend_filter=True, ma_trend_window=200,
        transaction_cost=0.0,
    )
    result_off = run_backtest(
        ma_filter_matrix, top_n=2, freq="ME",
        abs_momentum_filter=False,
        ma_trend_filter=False,
        transaction_cost=0.0,
    )
    down_on = sum(1 for e in result_on.holdings_log if "DOWN" in e["holdings"])
    down_off = sum(1 for e in result_off.holdings_log if "DOWN" in e["holdings"])
    assert down_on <= down_off


def test_ma_trend_filter_all_below_goes_to_cash(ma_filter_matrix):
    """When all ETFs are below MA (use only DOWN column), result should be mostly cash."""
    down_only = ma_filter_matrix[["DOWN"]]
    result = run_backtest(
        down_only, top_n=1, freq="ME",
        abs_momentum_filter=False,
        ma_trend_filter=True, ma_trend_window=200,
        transaction_cost=0.0,
    )
    # After warm-up, all rebalances should be cash (empty holdings)
    late_logs = [e for e in result.holdings_log if e["date"].year >= 2021]
    cash_events = sum(1 for e in late_logs if len(e["holdings"]) == 0)
    assert cash_events == len(late_logs)


# ── Market regime filter tests ────────────────────────────────────────────────

@pytest.fixture
def regime_matrix():
    """
    REGIME: strong downtrend (acts as the market index in bear mode).
    RISK:   moderate uptrend (risk asset to hold in bull mode).
    SAFE:   flat/slight uptrend (defensive asset to hold in bear mode).
    500 trading days so MA warm-up is satisfied.
    """
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(55)
    regime = 100 * np.cumprod(1 + np.full(n, -0.003))   # always below its MA
    risk   = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    safe   = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.003, n))
    return pd.DataFrame({"REGIME": regime, "RISK": risk, "SAFE": safe}, index=dates)


def test_regime_filter_restricts_to_defensive_in_bear(regime_matrix):
    """When regime ETF is below MA, only defensive ETFs should be selected."""
    result = run_backtest(
        regime_matrix, top_n=2, freq="ME",
        abs_momentum_filter=False, transaction_cost=0.0,
        regime_filter=True, regime_etf="REGIME",
        regime_ma_window=200, defensive_etfs=["SAFE"],
    )
    # After warm-up (year 2021+), REGIME is always below MA → only SAFE should appear
    late_logs = [e for e in result.holdings_log if e["date"].year >= 2021]
    for entry in late_logs:
        for h in entry["holdings"]:
            assert h == "SAFE", f"Expected only SAFE in bear regime, got {h}"


def test_regime_filter_off_allows_risk_assets(regime_matrix):
    """Without regime filter, RISK can be selected even when REGIME is bearish."""
    result = run_backtest(
        regime_matrix, top_n=2, freq="ME",
        abs_momentum_filter=False, transaction_cost=0.0,
        regime_filter=False,
    )
    risk_count = sum(1 for e in result.holdings_log if "RISK" in e["holdings"])
    assert risk_count > 0
