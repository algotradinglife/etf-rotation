"""
Tests for DataStore.close_matrix() after migration to quant-data storage layer.
Data is seeded directly via ParquetStorage/BarData — no API calls needed.
"""
from datetime import datetime

import pandas as pd
import pytest

from quant_data import Exchange, Interval
from quant_data.models import BarData
from quant_data.storage import ParquetStorage

from etf_rotation.data import DataStore


def _bar(symbol: str, exchange: Exchange, dt: datetime, close: float) -> BarData:
    return BarData(
        symbol=symbol,
        exchange=exchange,
        interval=Interval.DAILY,
        datetime=dt,
        open_price=close * 0.99,
        high_price=close * 1.01,
        low_price=close * 0.98,
        close_price=close,
        volume=1_000_000.0,
        amount=close * 1_000_000.0,
    )


@pytest.fixture()
def storage(tmp_path):
    return ParquetStorage(tmp_path)


@pytest.fixture()
def store(storage):
    return DataStore(storage=storage)


DATES = [datetime(2024, 1, d) for d in (2, 3, 4, 5, 8)]


class TestCloseMatrix:
    def test_single_symbol_shape(self, store, storage):
        bars = [_bar("510300.SH", Exchange.SSE, d, 4.0 + i * 0.01) for i, d in enumerate(DATES)]
        storage.save_bar_data(bars)

        matrix = store.close_matrix(["510300.SH"])
        assert isinstance(matrix, pd.DataFrame)
        assert "510300.SH" in matrix.columns
        assert len(matrix) == len(DATES)

    def test_index_is_tz_naive_datetimeindex(self, store, storage):
        bars = [_bar("510300.SH", Exchange.SSE, d, 4.0) for d in DATES]
        storage.save_bar_data(bars)

        matrix = store.close_matrix(["510300.SH"])
        assert matrix.index.tz is None, "close_matrix must return tz-naive index for backtest compat"

    def test_close_values_correct(self, store, storage):
        closes = [2.0, 2.1, 2.2, 2.3, 2.4]
        bars = [_bar("159915.SZ", Exchange.SZSE, d, c) for d, c in zip(DATES, closes)]
        storage.save_bar_data(bars)

        matrix = store.close_matrix(["159915.SZ"])
        assert list(matrix["159915.SZ"]) == pytest.approx(closes)

    def test_multi_symbol(self, store, storage):
        for sym, ex in [("510300.SH", Exchange.SSE), ("159915.SZ", Exchange.SZSE)]:
            bars = [_bar(sym, ex, d, 1.0 + i * 0.01) for i, d in enumerate(DATES)]
            storage.save_bar_data(bars)

        matrix = store.close_matrix(["510300.SH", "159915.SZ"])
        assert set(matrix.columns) == {"510300.SH", "159915.SZ"}
        assert len(matrix) == len(DATES)

    def test_missing_symbol_returns_no_column(self, store):
        matrix = store.close_matrix(["999999.SH"])
        assert matrix.empty

    def test_partial_missing_fills_nan(self, store, storage):
        bars_a = [_bar("510300.SH", Exchange.SSE, d, 4.0) for d in DATES]
        storage.save_bar_data(bars_a)
        # 159915 not seeded — column absent, not NaN column

        matrix = store.close_matrix(["510300.SH", "159915.SZ"])
        assert "510300.SH" in matrix.columns
        assert "159915.SZ" not in matrix.columns

    def test_index_is_sorted(self, store, storage):
        # Insert in reverse order
        bars = [_bar("510300.SH", Exchange.SSE, d, 4.0) for d in reversed(DATES)]
        storage.save_bar_data(bars)

        matrix = store.close_matrix(["510300.SH"])
        assert matrix.index.is_monotonic_increasing
