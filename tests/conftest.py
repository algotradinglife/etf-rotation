import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_price_df():
    """100 trading days of synthetic close prices for 3 ETFs."""
    dates = pd.bdate_range("2023-01-01", periods=100)
    np.random.seed(42)
    data = {}
    for code in ["510050.SH", "510300.SH", "518880.SH"]:
        prices = 1000 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))
        data[code] = prices
    df = pd.DataFrame(data, index=dates)
    df.index.name = "trade_date"
    return df
