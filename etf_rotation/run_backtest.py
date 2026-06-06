"""Run the 2012–2025 backtest on all 21 candidate ETFs and print results."""
from etf_rotation.data import DataStore
from etf_rotation.backtest import run_backtest
from etf_rotation.config import CANDIDATE_ETFS


def _row(label, result):
    cash_events = sum(1 for e in result.holdings_log if e.get("cash"))
    return (
        f"  {label:<44} "
        f"{result.cagr:>6.1%}  "
        f"{result.sharpe_ratio:>5.2f}  "
        f"{result.max_drawdown:>7.1%}  "
        f"cash={cash_events}"
    )


def main():
    store = DataStore()
    prices = store.close_matrix(CANDIDATE_ETFS)

    if prices.empty:
        print("No data found. Run: python -m etf_rotation.fetch")
        return

    print(f"Price matrix: {prices.shape[0]} days × {prices.shape[1]} ETFs")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n")

    header = f"  {'Scenario':<44} {'CAGR':>6}  {'Sharpe':>5}  {'MaxDD':>7}  Notes"
    print(header)
    print("  " + "-" * 80)

    base_kwargs = dict(transaction_cost=0.001, abs_momentum_filter=True)

    scenarios = [
        ("Monthly  baseline (no stops)",
         dict(freq="ME", holding_stop=None, portfolio_stop=None)),
        ("Monthly  holding -8%",
         dict(freq="ME", holding_stop=-0.08, portfolio_stop=None)),
        ("Monthly  holding -10%",
         dict(freq="ME", holding_stop=-0.10, portfolio_stop=None)),
        ("Monthly  holding -15%",
         dict(freq="ME", holding_stop=-0.15, portfolio_stop=None)),
        ("Monthly  portfolio -15%",
         dict(freq="ME", holding_stop=None, portfolio_stop=-0.15)),
        ("Monthly  portfolio -20%",
         dict(freq="ME", holding_stop=None, portfolio_stop=-0.20)),
        ("Monthly  holding -10% + portfolio -15%",
         dict(freq="ME", holding_stop=-0.10, portfolio_stop=-0.15)),
        ("  ─── Weekly ───",
         None),
        ("Weekly   baseline (no stops)",
         dict(freq="W", holding_stop=None, portfolio_stop=None)),
        ("Weekly   holding -8%",
         dict(freq="W", holding_stop=-0.08, portfolio_stop=None)),
        ("Weekly   holding -10%",
         dict(freq="W", holding_stop=-0.10, portfolio_stop=None)),
        ("Weekly   holding -15%",
         dict(freq="W", holding_stop=-0.15, portfolio_stop=None)),
        ("Weekly   portfolio -15%",
         dict(freq="W", holding_stop=None, portfolio_stop=-0.15)),
        ("Weekly   portfolio -20%",
         dict(freq="W", holding_stop=None, portfolio_stop=-0.20)),
        ("Weekly   holding -10% + portfolio -15%",
         dict(freq="W", holding_stop=-0.10, portfolio_stop=-0.15)),
    ]

    for label, kwargs in scenarios:
        if kwargs is None:
            print(f"\n{label}")
            continue
        result = run_backtest(prices, top_n=3, **base_kwargs, **kwargs)
        print(_row(label, result))

    # ── MA trend filter scenarios ────────────────────────────────────────────
    print("\n  ─── MA-200 trend filter ───")
    ma_scenarios = [
        ("Monthly  MA-200 baseline",
         dict(freq="ME", holding_stop=None, portfolio_stop=None, ma_trend_filter=True)),
        ("Monthly  MA-200 + portfolio -15%",
         dict(freq="ME", holding_stop=None, portfolio_stop=-0.15, ma_trend_filter=True)),
        ("Weekly   MA-200 baseline",
         dict(freq="W", holding_stop=None, portfolio_stop=None, ma_trend_filter=True)),
        ("Weekly   MA-200 + portfolio -15%",
         dict(freq="W", holding_stop=None, portfolio_stop=-0.15, ma_trend_filter=True)),
    ]
    for label, kwargs in ma_scenarios:
        result = run_backtest(prices, top_n=3, **base_kwargs, **kwargs)
        print(_row(label, result))

    # ── Market regime filter scenarios ───────────────────────────────────────
    _defensive = ["518880.SH", "511010.SH", "511020.SH"]
    _regime_kwargs = dict(
        regime_filter=True, regime_etf="510300.SH",
        regime_ma_window=200, defensive_etfs=_defensive,
    )
    print("\n  ─── Market regime filter (CSI300 MA-200 → defensive) ───")
    regime_scenarios = [
        ("Monthly  regime baseline",
         dict(freq="ME", holding_stop=None, portfolio_stop=None)),
        ("Monthly  regime + portfolio -15%",
         dict(freq="ME", holding_stop=None, portfolio_stop=-0.15)),
        ("Weekly   regime baseline",
         dict(freq="W", holding_stop=None, portfolio_stop=None)),
        ("Weekly   regime + MA-200",
         dict(freq="W", holding_stop=None, portfolio_stop=None, ma_trend_filter=True)),
        ("Weekly   regime + portfolio -15%",
         dict(freq="W", holding_stop=None, portfolio_stop=-0.15)),
    ]
    for label, kwargs in regime_scenarios:
        result = run_backtest(prices, top_n=3, **base_kwargs, **_regime_kwargs, **kwargs)
        print(_row(label, result))

    # Last 3 decisions for best weekly config
    print()
    best = run_backtest(prices, top_n=3, freq="W",
                        holding_stop=-0.10, portfolio_stop=-0.15, **base_kwargs)
    print("Last 3 weekly decisions (holding -10% + portfolio -15%):")
    for entry in best.holdings_log[-3:]:
        date_str = entry["date"].strftime("%Y-%m-%d")
        holdings = ", ".join(entry["holdings"]) if entry["holdings"] else "CASH"
        print(f"  {date_str}: {holdings}")

    print()
    best_ma = run_backtest(prices, top_n=3, freq="W",
                           ma_trend_filter=True, **base_kwargs)
    print("Last 3 weekly decisions (MA-200 baseline):")
    for entry in best_ma.holdings_log[-3:]:
        date_str = entry["date"].strftime("%Y-%m-%d")
        holdings = ", ".join(entry["holdings"]) if entry["holdings"] else "CASH"
        print(f"  {date_str}: {holdings}")

    print()
    best_regime = run_backtest(prices, top_n=3, freq="W", **base_kwargs, **_regime_kwargs)
    print("Last 3 weekly decisions (regime MA-200 baseline):")
    for entry in best_regime.holdings_log[-3:]:
        date_str = entry["date"].strftime("%Y-%m-%d")
        holdings = ", ".join(entry["holdings"]) if entry["holdings"] else "CASH"
        print(f"  {date_str}: {holdings}")

    print()
    recommended = run_backtest(
        prices, top_n=3, freq="W", **base_kwargs,
        regime_filter=True, regime_etf="510300.SH",
        regime_ma_window=150, defensive_etfs=_defensive,
        holding_stop=-0.08,
    )
    print("Last 3 weekly decisions (recommended: regime MA-150 + hold -8%):")
    for entry in recommended.holdings_log[-3:]:
        date_str = entry["date"].strftime("%Y-%m-%d")
        holdings = ", ".join(entry["holdings"]) if entry["holdings"] else "CASH"
        print(f"  {date_str}: {holdings}")

    # ── Optimised configurations (from parameter sweep) ──────────────────────
    _opt_regime = dict(
        regime_filter=True, regime_etf="510300.SH",
        regime_ma_window=150, defensive_etfs=_defensive,
    )
    print("\n  ─── Optimised (regime MA-150 + holding stop) ───")
    opt_scenarios = [
        ("Weekly   regime-150 + hold -6%  [aggressive]",
         dict(freq="W", holding_stop=-0.06, portfolio_stop=None)),
        ("Weekly   regime-150 + hold -8%  [recommended]",
         dict(freq="W", holding_stop=-0.08, portfolio_stop=None)),
        ("Weekly   regime-150 + hold -10%",
         dict(freq="W", holding_stop=-0.10, portfolio_stop=None)),
        ("Weekly   regime-150 baseline",
         dict(freq="W", holding_stop=None,  portfolio_stop=None)),
    ]
    for label, kwargs in opt_scenarios:
        result = run_backtest(prices, top_n=3, **base_kwargs, **_opt_regime, **kwargs)
        print(_row(label, result))

    # ── Annual returns: recommended vs CSI300 ────────────────────────────────
    print("\n  ─── Annual returns: recommended vs no-filter baseline vs CSI300 ───")
    rec = run_backtest(
        prices, top_n=3, freq="W", **base_kwargs,
        **_opt_regime, holding_stop=-0.08,
    )
    baseline = run_backtest(prices, top_n=3, freq="W", **base_kwargs)
    csi300 = prices["510300.SH"].dropna() if "510300.SH" in prices.columns else None

    pv_rec = rec.portfolio_values.dropna()
    pv_bas = baseline.portfolio_values.dropna()

    print(f"  {'Year':>4}  {'Recommended':>12}  {'Baseline':>10}  {'CSI300':>8}")
    print("  " + "-" * 42)
    for year in sorted(pv_rec.index.year.unique()):
        yr_rec = pv_rec[pv_rec.index.year == year]
        yr_bas = pv_bas[pv_bas.index.year == year]
        ret_rec = yr_rec.iloc[-1] / yr_rec.iloc[0] - 1 if len(yr_rec) > 1 else float("nan")
        ret_bas = yr_bas.iloc[-1] / yr_bas.iloc[0] - 1 if len(yr_bas) > 1 else float("nan")
        if csi300 is not None:
            yr_csi = csi300[csi300.index.year == year]
            ret_csi = yr_csi.iloc[-1] / yr_csi.iloc[0] - 1 if len(yr_csi) > 1 else float("nan")
            csi_str = f"{ret_csi:>8.1%}"
        else:
            csi_str = "     n/a"
        print(f"  {year:>4}  {ret_rec:>12.1%}  {ret_bas:>10.1%}  {csi_str}")


if __name__ == "__main__":
    main()
