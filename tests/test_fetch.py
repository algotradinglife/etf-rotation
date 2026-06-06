"""Tests for fetch.py — verifies DataManager.update_batch is called correctly."""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from quant_data import Interval

from etf_rotation.config import CANDIDATE_ETFS_WITH_EXCHANGE
from etf_rotation.fetch import fetch_all


@pytest.fixture()
def mock_manager():
    m = MagicMock()
    m.update_batch.return_value = {sym: 0 for sym, _ in CANDIDATE_ETFS_WITH_EXCHANGE}
    return m


def test_fetch_all_calls_update_batch(mock_manager):
    with patch("etf_rotation.fetch._build_manager", return_value=mock_manager):
        fetch_all("20240101", "20241231")

    mock_manager.update_batch.assert_called_once()
    args, kwargs = mock_manager.update_batch.call_args
    symbols_arg, interval_arg, start_arg, end_arg = args
    assert symbols_arg == CANDIDATE_ETFS_WITH_EXCHANGE
    assert interval_arg == Interval.DAILY
    assert start_arg == datetime(2024, 1, 1)
    assert end_arg == datetime(2024, 12, 31)


def test_fetch_all_prints_new_bars(mock_manager, capsys):
    mock_manager.update_batch.return_value = {"510300.SH": 5, "159915.SZ": 0}
    with patch("etf_rotation.fetch._build_manager", return_value=mock_manager):
        fetch_all("20240101", "20241231")

    out = capsys.readouterr().out
    assert "510300.SH: +5 new bars" in out
    assert "159915.SZ: already up to date" in out


def test_fetch_all_uses_default_date_range(mock_manager):
    with patch("etf_rotation.fetch._build_manager", return_value=mock_manager):
        fetch_all()

    mock_manager.update_batch.assert_called_once()
    _, _, start_arg, end_arg = mock_manager.update_batch.call_args[0]
    assert start_arg.year == 2012
    assert end_arg.year == 2025
