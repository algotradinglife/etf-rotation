import pandas as pd
import numpy as np
from etf_rotation.config import MOMENTUM_WINDOWS, MOMENTUM_WEIGHTS


def compute_returns(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """N-day simple return: (price_t - price_{t-N}) / price_{t-N}."""
    return prices.pct_change(window)


def rank_scores(returns: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank on each row (1 = lowest, N = highest)."""
    return returns.rank(axis=1, ascending=True, na_option="keep")


def composite_score(prices: pd.DataFrame) -> pd.DataFrame:
    """Weighted sum of rank scores across all momentum windows."""
    total_weight = sum(MOMENTUM_WEIGHTS[w] for w in MOMENTUM_WINDOWS)
    score = None
    for window in MOMENTUM_WINDOWS:
        ret = compute_returns(prices, window)
        ranks = rank_scores(ret)
        weighted = ranks * (MOMENTUM_WEIGHTS[window] / total_weight)
        score = weighted if score is None else score + weighted
    # Zero out rows where any window still has NaN
    max_window = max(MOMENTUM_WINDOWS)
    score.iloc[:max_window] = np.nan
    return score


def score_all(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Return composite scores for all ETFs as of a specific date."""
    scores = composite_score(prices)
    if as_of not in scores.index:
        # Use closest available date <= as_of
        valid = scores.index[scores.index <= as_of]
        if valid.empty:
            return pd.Series(dtype=float)
        as_of = valid[-1]
    return scores.loc[as_of].dropna().sort_values(ascending=False)
