"""Shared helpers: benchmark download, holding-period compounding, σ-matched cumulative log.

Used by ``plot_figure5_vol_adjusted_cumlog.py`` and ``export_ppt15_cum_log.py``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "final_submit_version" / "outputs" / "cache"


def _norm_ix(ix) -> pd.DatetimeIndex:
    return pd.to_datetime(ix, utc=False).tz_localize(None).normalize()


def load_backtest(h: int) -> pd.DataFrame:
    path = CACHE_DIR / f"backtest_I{h}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_pickle(path)
    df = df.copy()
    df.index = _norm_ix(df.index)
    return df.sort_index()


def download_spy_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import yfinance as yf

    pad = pd.Timedelta(days=7)
    kw = dict(
        start=(start - pad).strftime("%Y-%m-%d"),
        end=(end + pad).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            df = yf.download("SPY", **kw)
            if df.empty or len(df) < 10:
                raise RuntimeError("yfinance returned empty SPY history")
            close = df["Close"].squeeze()
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            close.index = _norm_ix(close.index)
            close = close.sort_index()
            ret = close.pct_change()
            ret.name = "SPY"
            return ret.dropna()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance SPY failed after retries: {last_err}") from last_err


def download_ff_market_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import pandas_datareader.data as web

    pad = pd.Timedelta(days=35)
    ff = web.DataReader(
        "F-F_Research_Data_Factors_daily",
        "famafrench",
        start=start - pad,
        end=end + pad,
    )[0]
    ff = ff.copy()
    ff.columns = [str(c).strip() for c in ff.columns]
    ff.index = _norm_ix(ff.index)
    ff = ff.sort_index()
    mkt = (ff["Mkt-RF"] + ff["RF"]) / 100.0
    mkt.name = "FF_mkt"
    return mkt.dropna()


def download_benchmark_daily(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, str]:
    """Returns (daily simple return series, short label for captions)."""
    try:
        return download_spy_daily(start, end), "SPY"
    except Exception as yf_exc:  # noqa: BLE001
        print(f"[warn] SPY via yfinance unavailable ({yf_exc}); using FF daily market.", file=sys.stderr)
        return download_ff_market_daily(start, end), "CRSP VW mkt (FF)"


def compound_spy_over_schedule(spy_ret_daily: pd.Series, rebal_dates: pd.DatetimeIndex) -> pd.Series:
    """Holding-period benchmark return for each interval [t_i, t_{i+1}), aligned to t_i."""
    dates = _norm_ix(rebal_dates).sort_values().unique()
    out_idx: list = []
    out_val: list[float] = []
    for i, t_start in enumerate(dates):
        t_end = dates[i + 1] if i + 1 < len(dates) else None
        if t_end is None:
            break
        chunk = spy_ret_daily[(spy_ret_daily.index > t_start) & (spy_ret_daily.index <= t_end)]
        if chunk.empty:
            continue
        r = float((1.0 + chunk).prod() - 1.0)
        out_idx.append(t_start)
        out_val.append(r)
    return pd.Series(out_val, index=pd.DatetimeIndex(out_idx, name="dlycaldt"), name="bench_hold")


def vol_scaled_cumulative_log(strategy_r: pd.Series, bench_hold_r: pd.Series) -> pd.Series:
    """Inner-join on dates; scale strategy vol to benchmark vol; return cum sum log(1+r_tilde)."""
    j = pd.DataFrame({"strat": strategy_r.astype(float), "bench": bench_hold_r.astype(float)}).dropna(
        how="any"
    )
    if len(j) < 5:
        return pd.Series(dtype=float)
    s_strat = j["strat"].std(ddof=1)
    s_b = j["bench"].std(ddof=1)
    if s_strat <= 0 or s_b <= 0 or not np.isfinite(s_strat) or not np.isfinite(s_b):
        return pd.Series(dtype=float)
    k = s_b / s_strat
    adj = j["strat"] * k
    return np.log1p(adj).cumsum()
