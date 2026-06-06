from dataclasses import dataclass
import pandas as pd
import numpy as np
from etf_rotation.momentum import score_all
from etf_rotation.config import MOMENTUM_WINDOWS


@dataclass
class BacktestResult:
    portfolio_values: pd.Series
    holdings_log: list[dict]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    cagr: float


def _abs_momentum_positive(prices: pd.DataFrame, etf: str, date_idx: int, window: int) -> bool:
    """Return True if ETF has positive absolute return over the given window."""
    if date_idx < window:
        return False
    date = prices.index[date_idx]
    prev = prices.index[date_idx - window]
    cur_price = prices.loc[date, etf]
    prev_price = prices.loc[prev, etf]
    if pd.isna(cur_price) or pd.isna(prev_price) or prev_price <= 0:
        return False
    return cur_price > prev_price


def _above_ma(prices: pd.DataFrame, etf: str, date_idx: int, window: int) -> bool:
    """Return True if ETF's close is above its simple moving average over the window."""
    if date_idx < window - 1 or etf not in prices.columns:
        return False
    series = prices.iloc[date_idx - window + 1 : date_idx + 1][etf].dropna()
    if len(series) < window // 2:
        return False
    cur = prices.iloc[date_idx][etf]
    return bool(pd.notna(cur) and float(cur) > float(series.mean()))


def _compute_turnover(old_holdings: list[str], new_holdings: list[str]) -> float:
    """One-way portfolio turnover as a fraction [0, 1]."""
    if not old_holdings and not new_holdings:
        return 0.0
    n_old = len(old_holdings) if old_holdings else 1
    n_new = len(new_holdings) if new_holdings else 1
    old_w = {h: 1.0 / n_old for h in old_holdings}
    new_w = {h: 1.0 / n_new for h in new_holdings}
    all_etfs = set(old_w) | set(new_w)
    return 0.5 * sum(abs(new_w.get(h, 0.0) - old_w.get(h, 0.0)) for h in all_etfs)


def run_backtest(
    prices: pd.DataFrame,
    top_n: int = 3,
    freq: str = "ME",  # "ME" = month-end, "W" = week-end
    transaction_cost: float = 0.001,  # 0.1% one-way (A-share stamp duty + commission)
    abs_momentum_filter: bool = True,
    abs_momentum_window: int = 120,
    holding_stop: float | None = None,   # e.g. -0.10: exit if holding drops 10% from entry
    portfolio_stop: float | None = None, # e.g. -0.15: exit all if portfolio drops 15% from HWM
    ma_trend_filter: bool = False,       # only hold ETFs trading above their N-day MA
    ma_trend_window: int = 200,
    regime_filter: bool = False,         # top-down bear/bull gating via a single index ETF
    regime_etf: str = "510300.SH",       # the ETF used as market regime indicator
    regime_ma_window: int = 200,         # MA window for regime detection
    defensive_etfs: list[str] | None = None,  # ETFs allowed in bear regime
) -> BacktestResult:
    """
    Simulate monthly/weekly rotation into top_n ETFs by composite momentum score.
    Equal-weight portfolio, rebalanced at each period end.

    abs_momentum_filter: exclude ETFs with negative absolute return over abs_momentum_window
    transaction_cost: one-way cost applied to portfolio turnover on each rebalance/stop
    holding_stop: per-position stop relative to entry price at last rebalance
    portfolio_stop: portfolio-level stop relative to high-water mark
    """
    rebalance_dates = pd.DatetimeIndex(
        prices.groupby(pd.Grouper(freq=freq))
        .apply(lambda x: x.index[-1] if len(x) > 0 else None)
        .dropna()
        .values
    )
    min_window = max(MOMENTUM_WINDOWS)
    rebalance_dates = rebalance_dates[rebalance_dates >= prices.index[min_window]]

    portfolio = pd.Series(index=prices.index, dtype=float)
    holdings_log = []
    current_holdings: list[str] = []
    stopped_holdings: set[str] = set()   # stopped out since last rebalance
    entry_prices: dict[str, float] = {}  # close prices at last rebalance
    peak_value: float = 1.0

    for i, date in enumerate(prices.index):
        n_original = len(current_holdings)
        active = [h for h in current_holdings if h not in stopped_holdings]

        # ── Step 1: Daily P&L ────────────────────────────────────────────────
        # Stopped positions contribute 0 (cash); use original weight 1/n_original
        # so their allocation isn't redistributed to surviving holdings.
        if n_original > 0 and i > 0:
            prev_val = portfolio.iloc[i - 1]
            tradeable = [
                h for h in active
                if h in prices.columns
                and pd.notna(prices.iloc[i - 1][h]) and prices.iloc[i - 1][h] > 0
                and pd.notna(prices.loc[date, h])
            ]
            avg_ret = sum(
                (1.0 / n_original) * (prices.loc[date, h] / prices.iloc[i - 1][h] - 1)
                for h in tradeable
            ) if tradeable else 0.0
            portfolio.iloc[i] = prev_val * (1 + avg_ret)
        elif i == 0:
            portfolio.iloc[i] = 1.0
        else:
            portfolio.iloc[i] = portfolio.iloc[i - 1]

        # ── Step 2: Per-holding stop check ───────────────────────────────────
        if holding_stop is not None and n_original > 0 and i > 0:
            for h in list(active):
                if h not in entry_prices:
                    continue
                cur = prices.loc[date, h]
                if pd.notna(cur) and cur / entry_prices[h] - 1 < holding_stop:
                    stopped_holdings.add(h)
                    portfolio.iloc[i] *= (1 - (1.0 / n_original) * transaction_cost)

        # ── Step 3: Portfolio drawdown stop ──────────────────────────────────
        if portfolio_stop is not None and n_original > 0 and i > 0:
            if pd.notna(portfolio.iloc[i]) and portfolio.iloc[i] < peak_value * (1 + portfolio_stop):
                remaining = [h for h in current_holdings if h not in stopped_holdings]
                if remaining:
                    remaining_weight = len(remaining) / n_original
                    portfolio.iloc[i] *= (1 - remaining_weight * transaction_cost)
                stopped_holdings = set(current_holdings)

        # Update high-water mark
        if pd.notna(portfolio.iloc[i]):
            peak_value = max(peak_value, portfolio.iloc[i])

        # ── Step 4: Rebalance ────────────────────────────────────────────────
        if date in rebalance_dates:
            scores = score_all(prices, as_of=date)
            if not scores.empty:
                selected = list(scores.head(top_n).index)

                if abs_momentum_filter:
                    selected = [
                        h for h in selected
                        if _abs_momentum_positive(prices, h, i, abs_momentum_window)
                    ]

                if ma_trend_filter:
                    selected = [
                        h for h in selected
                        if _above_ma(prices, h, i, ma_trend_window)
                    ]

                if regime_filter and regime_etf in prices.columns:
                    bull = _above_ma(prices, regime_etf, i, regime_ma_window)
                    if not bull:
                        # Bear regime: restrict to defensive ETFs only
                        allowed = set(defensive_etfs or [])
                        candidates = [h for h in selected if h in allowed]
                        if not candidates:
                            # Score defensive ETFs directly if none survived selection
                            candidates = [
                                h for h in (defensive_etfs or [])
                                if h in prices.columns
                                and pd.notna(prices.iloc[i][h])
                                and not abs_momentum_filter or _abs_momentum_positive(prices, h, i, abs_momentum_window)
                            ]
                        selected = candidates[:top_n]

                # Turnover: compare effective current (non-stopped) vs new selection
                if transaction_cost > 0:
                    effective_current = [h for h in current_holdings if h not in stopped_holdings]
                    turnover = _compute_turnover(effective_current, selected)
                    portfolio.iloc[i] *= (1 - turnover * transaction_cost)

                current_holdings = selected
                stopped_holdings = set()
                entry_prices = {
                    h: float(prices.loc[date, h])
                    for h in selected
                    if h in prices.columns and pd.notna(prices.loc[date, h])
                }
                holdings_log.append({
                    "date": date,
                    "holdings": selected,
                    "scores": scores.head(top_n).to_dict(),
                    "cash": len(selected) == 0,
                })

    portfolio = portfolio.dropna()

    if holdings_log:
        active_start = holdings_log[0]["date"]
        active_portfolio = portfolio.loc[active_start:]
    else:
        active_portfolio = portfolio

    daily_rets = active_portfolio.pct_change().dropna()
    n_years = len(active_portfolio) / 252
    total_return = float(active_portfolio.iloc[-1] / active_portfolio.iloc[0] - 1) if len(active_portfolio) > 1 else 0.0
    cagr = float((active_portfolio.iloc[-1] / active_portfolio.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) if len(active_portfolio) > 1 else 0.0
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0.0
    rolling_max = active_portfolio.cummax()
    max_dd = float(((active_portfolio - rolling_max) / rolling_max).min()) if len(active_portfolio) > 1 else 0.0

    return BacktestResult(
        portfolio_values=portfolio,
        holdings_log=holdings_log,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        cagr=cagr,
    )
