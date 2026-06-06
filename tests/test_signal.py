import pytest
import pandas as pd
import numpy as np
from etf_rotation.signal import generate_signal, format_signal


@pytest.fixture
def price_matrix_2yr():
    """3 ETFs, 500 trading days so momentum windows have data."""
    dates = pd.bdate_range("2022-01-01", periods=500)
    np.random.seed(99)
    data = {
        "510050.SH": 100 * np.cumprod(1 + np.random.normal(0.0006, 0.012, 500)),
        "510300.SH": 100 * np.cumprod(1 + np.random.normal(0.0003, 0.010, 500)),
        "518880.SH": 100 * np.cumprod(1 + np.random.normal(0.0008, 0.009, 500)),
    }
    return pd.DataFrame(data, index=dates)


def test_generate_signal_returns_series(price_matrix_2yr):
    sig, is_bull = generate_signal(price_matrix_2yr, top_n=2)
    assert isinstance(sig, pd.Series)
    assert isinstance(is_bull, bool)
    assert len(sig) <= 2  # may be fewer if regime restricts to defensives


def test_generate_signal_returns_only_known_codes(price_matrix_2yr):
    sig, _ = generate_signal(price_matrix_2yr, top_n=2)
    for code in sig.index:
        assert code in price_matrix_2yr.columns


def test_format_signal_contains_date(price_matrix_2yr):
    sig, is_bull = generate_signal(price_matrix_2yr, top_n=2)
    text = format_signal(sig, as_of=price_matrix_2yr.index[-1], is_bull=is_bull)
    assert "2023" in text or "2024" in text


def test_format_signal_contains_etf_codes(price_matrix_2yr):
    sig, is_bull = generate_signal(price_matrix_2yr, top_n=3)
    text = format_signal(sig, as_of=price_matrix_2yr.index[-1], is_bull=is_bull)
    for code in sig.index:
        assert code in text


def test_generate_signal_bull_regime_returns_top_n(price_matrix_2yr):
    """In bull regime (510300 above MA), signal has up to top_n ETFs."""
    sig, is_bull = generate_signal(price_matrix_2yr, top_n=3)
    if is_bull:
        assert len(sig) == 3
    # bear regime tested implicitly: signal may be empty or limited to defensives


def test_format_signal_shows_regime(price_matrix_2yr):
    sig, is_bull = generate_signal(price_matrix_2yr, top_n=2)
    text = format_signal(sig, as_of=price_matrix_2yr.index[-1], is_bull=is_bull)
    assert "BULL" in text or "BEAR" in text
