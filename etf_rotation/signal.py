import pandas as pd
from etf_rotation.momentum import score_all
from etf_rotation.config import TOP_N

# Recommended config (matches run_backtest optimised scenario)
REGIME_ETF = "510300.SH"
REGIME_MA_WINDOW = 150
DEFENSIVE_ETFS = ["518880.SH", "511010.SH", "511020.SH"]  # 黄金, 5年国债, 10年国债


def _is_bull_regime(prices: pd.DataFrame, as_of_idx: int) -> bool:
    """True if REGIME_ETF closes above its REGIME_MA_WINDOW-day MA."""
    if REGIME_ETF not in prices.columns or as_of_idx < REGIME_MA_WINDOW - 1:
        return True
    series = prices.iloc[as_of_idx - REGIME_MA_WINDOW + 1 : as_of_idx + 1][REGIME_ETF].dropna()
    if len(series) < REGIME_MA_WINDOW // 2:
        return True
    cur = prices.iloc[as_of_idx][REGIME_ETF]
    return bool(pd.notna(cur) and float(cur) > float(series.mean()))


def generate_signal(prices: pd.DataFrame, top_n: int = TOP_N) -> tuple[pd.Series, bool]:
    """Return (ranked ETF scores, is_bull_regime) as of the latest date."""
    as_of_idx = len(prices) - 1
    as_of = prices.index[-1]
    bull = _is_bull_regime(prices, as_of_idx)

    scores = score_all(prices, as_of=as_of)

    if bull:
        selected = scores.head(top_n)
    else:
        defensive_scores = scores[scores.index.isin(DEFENSIVE_ETFS)]
        selected = defensive_scores.head(top_n) if not defensive_scores.empty else pd.Series(dtype=float)

    return selected, bull


def format_signal(signal: pd.Series, as_of: pd.Timestamp, is_bull: bool) -> str:
    regime_label = "BULL 📈" if is_bull else "BEAR 🛡️  → defensive only"
    lines = [
        f"=== ETF Rotation Signal — {as_of.strftime('%Y-%m-%d')} ===",
        f"Regime: {regime_label}  (沪深300 {'above' if is_bull else 'below'} MA-{REGIME_MA_WINDOW})",
        f"Hold the following {len(signal)} ETF(s):",
        "",
    ]
    for rank, (code, score) in enumerate(signal.items(), start=1):
        lines.append(f"  {rank}. {code}  (momentum score: {score:.2f})")
    if not signal.empty:
        lines += ["", "Equal-weight allocation recommended."]
    else:
        lines += ["", "⚠ No qualifying ETFs — hold cash."]
    return "\n".join(lines)


def print_current_signal() -> None:
    from etf_rotation.data import DataStore
    from etf_rotation.config import CANDIDATE_ETFS

    store = DataStore()
    prices = store.close_matrix(CANDIDATE_ETFS)
    if prices.empty:
        print("No data. Run: python -m etf_rotation.fetch")
        return
    signal, is_bull = generate_signal(prices)
    print(format_signal(signal, as_of=prices.index[-1], is_bull=is_bull))


if __name__ == "__main__":
    print_current_signal()
