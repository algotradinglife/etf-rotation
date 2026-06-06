from pathlib import Path

from quant_data import Exchange

BASE_DIR = Path(__file__).parent.parent
# quant_data stores at DATA_DIR/{exchange}/{symbol}/{interval}.parquet
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_ETFS = [
    # ── Broad Domestic ───────────────────────────────────────────────────────
    "510050.SH",  # 上证50ETF
    "510300.SH",  # 沪深300ETF
    "510500.SH",  # 中证500ETF
    "159902.SZ",  # 中小板ETF (上市 2006)
    # ── Growth / Tech ─────────────────────────────────────────────────────────
    "159915.SZ",  # 创业板ETF
    "588000.SH",  # 科创50ETF
    "159949.SZ",  # 创业板50ETF
    "159995.SZ",  # 半导体ETF
    "515050.SH",  # 5GETF
    # ── Sector / Thematic ─────────────────────────────────────────────────────
    "510880.SH",  # 上证红利ETF
    "159928.SZ",  # 消费ETF
    "159929.SZ",  # 医药ETF
    "512880.SH",  # 证券ETF
    "512660.SH",  # 国防军工ETF (上市 2013-05)
    "512400.SH",  # 有色金属ETF (上市 2013-05)
    # ── International ─────────────────────────────────────────────────────────
    "159920.SZ",  # 恒生ETF
    "513100.SH",  # 纳斯达克100ETF (上市 2013-05)
    "513500.SH",  # 标普500ETF (上市 2013-05)
    # ── Defensive ─────────────────────────────────────────────────────────────
    "518880.SH",  # 黄金ETF (上市 2013-07)
    "511010.SH",  # 5年期国债ETF (上市 2013-03)
    "511020.SH",  # 10年期国债ETF (上市 2015-04)
]

# (ts_code, Exchange) pairs — used by DataManager.update_batch and fetch.py
CANDIDATE_ETFS_WITH_EXCHANGE: list[tuple[str, Exchange]] = [
    (sym, Exchange.SSE if sym.endswith(".SH") else Exchange.SZSE)
    for sym in CANDIDATE_ETFS
]

MOMENTUM_WINDOWS = [20, 60, 120]  # trading days
MOMENTUM_WEIGHTS = {20: 0.2, 60: 0.4, 120: 0.4}  # weighted composite
TOP_N = 3  # ETFs to hold per period
BACKTEST_START = "20120101"
# BACKTEST_END is kept for explicit historical backtests that need a fixed range.
# fetch.py defaults to today's date so this value no longer drives daily refreshes.
BACKTEST_END = "20260601"
