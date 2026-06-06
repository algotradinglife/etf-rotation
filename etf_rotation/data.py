"""
Data access layer for etf-rotation.
Backed by quant_data.ParquetStorage — fetch logic has moved to fetch.py / DataManager.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_data import Exchange, Interval
from quant_data.storage import ParquetStorage

from etf_rotation.config import BACKTEST_START, CANDIDATE_ETFS_WITH_EXCHANGE, DATA_DIR

# symbol → Exchange lookup built from config
_EXCHANGE: dict[str, Exchange] = {sym: ex for sym, ex in CANDIDATE_ETFS_WITH_EXCHANGE}

_BACKTEST_START_DT = datetime.strptime(BACKTEST_START, "%Y%m%d")


def _infer_exchange(ts_code: str) -> Exchange:
    """Fall back for symbols not in config."""
    if ts_code in _EXCHANGE:
        return _EXCHANGE[ts_code]
    suffix = ts_code.split(".")[-1].upper() if "." in ts_code else ""
    return {"SH": Exchange.SSE, "SZ": Exchange.SZSE}.get(suffix, Exchange.UNKNOWN)


class DataStore:
    """
    Read-only view over the local Parquet store.
    Provides close_matrix() consumed by run_backtest.py and signal.py.

    Fetching (writing) is handled by fetch.py via DataManager.
    """

    def __init__(
        self,
        data_dir: Path | str = DATA_DIR,
        *,
        storage: ParquetStorage | None = None,
    ) -> None:
        self._storage = storage or ParquetStorage(data_dir, cache_size=32)

    def close_matrix(self, ts_codes: list[str]) -> pd.DataFrame:
        """
        Return a wide DataFrame of daily close prices.

        Index  : tz-naive DatetimeIndex (same as old DataStore)
        Columns: ts_codes
        Values : close price as float
        """
        frames: dict[str, pd.Series] = {}
        end = datetime.now()
        for code in ts_codes:
            exchange = _infer_exchange(code)
            df = self._storage.load_bar_data(
                code, exchange, Interval.DAILY,
                start=_BACKTEST_START_DT, end=end,
            )
            if df.empty:
                continue
            s = df.set_index("datetime")["close_price"].astype(float)
            # strip UTC tz so downstream DatetimeIndex operations stay tz-naive
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            frames[code] = s

        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).sort_index()
