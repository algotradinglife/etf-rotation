"""Incrementally fetch daily bars for all candidate ETFs via quant-data DataManager."""
from datetime import datetime

from quant_data import DataManager, Interval
from quant_data.datafeed import MinishareDatafeed
from quant_data.storage import ParquetStorage

from etf_rotation.config import (
    BACKTEST_START,
    CANDIDATE_ETFS_WITH_EXCHANGE,
    DATA_DIR,
)


def _build_manager() -> DataManager:
    return DataManager(
        datafeed=MinishareDatafeed(),
        storage=ParquetStorage(DATA_DIR, cache_size=32),
    )


def fetch_all(start_date: str = BACKTEST_START, end_date: str | None = None) -> None:
    """Incrementally fetch any missing bars for all candidate ETFs.

    end_date defaults to today so daily runs always fetch the latest bars
    without requiring a config change.
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.today() if end_date is None else datetime.strptime(end_date, "%Y%m%d")
    manager = _build_manager()
    results = manager.update_batch(CANDIDATE_ETFS_WITH_EXCHANGE, Interval.DAILY, start, end)
    for symbol, count in results.items():
        if count:
            print(f"{symbol}: +{count} new bars")
        else:
            print(f"{symbol}: already up to date")


if __name__ == "__main__":
    fetch_all()
