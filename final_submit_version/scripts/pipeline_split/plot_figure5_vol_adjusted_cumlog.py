#!/usr/bin/env python3
"""Jiang et al. (2023) Figure 5–style panel: volatility-matched cumulative log returns.

Matches the *construction* in the paper caption:

1. At each model's native rebalance dates, take equal-weight long–short returns
   (here: **gross** ``*_LS_gross*`` columns from cached ``backtest_I*.pkl``).
2. Build the same holding-period return on a **market benchmark** between consecutive
   rebalance dates. We try **SPY** (``yfinance``) first; if that fails (rate limits),
   we use **CRSP value-weighted market** from Ken French daily factors
   (``Mkt-RF + RF``), which matches many asset-pricing appendices and is close to
   “market vol” for rescaling.
3. Let σ_strat and σ_bench be full-sample standard deviations of those periodic returns.
4. Scale strategy returns by (σ_bench / σ_strat) so each series has the same vol as the benchmark
   over the test window.
5. Plot cumulative log return: sum_t log(1 + r̃_t) on a **linear** y-axis.

**Important cache limitation:** the paper plots **I5/R5, I20/R5, I60/R5** (fixed weekly
holding *R=5*). This repo's cached ``backtest_I20.pkl`` / ``backtest_I60.pkl`` use
**monthly / quarterly** rebalancing (matched *I=R*). Those curves are labelled
``I20/R20`` and ``I60/R60`` in the legend so the figure is not misread as a full
replication of the paper's horizon mix.

Extra series from cache on the weekly calendar: momentum and MA long–short (``backtest_I5``).

Needs **network** for the benchmark series (SPY and/or Fama–French daily).

Outputs (300 dpi PNG):

- ``final_submit_version/outputs/pipeline_runs/figures/figure5_vol_adjusted_cumlog.png``
- ``final_submit_version/outputs/ppt_images/figure5_vol_adjusted_cumlog.png``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT_DIR / "final_submit_version" / "outputs" / "cache"
FIG_DIR = ROOT_DIR / "final_submit_version" / "outputs" / "pipeline_runs" / "figures"
PPT_DIR = ROOT_DIR / "final_submit_version" / "outputs" / "ppt_images"


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
    # Daily factors are in percent; convert to decimals. Total market ~ Mkt-RF + RF.
    mkt = (ff["Mkt-RF"] + ff["RF"]) / 100.0
    mkt.name = "FF_mkt"
    return mkt.dropna()


def download_benchmark_daily(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, str]:
    """Returns (daily simple return series, human label for plots)."""
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
    """Inner-join on dates; scale strategy vol to benchmark vol; return cum log(1+r)."""
    j = pd.DataFrame({"strat": strategy_r.astype(float), "bench": bench_hold_r.astype(float)}).dropna(
        how="any"
    )
    if len(j) < 5:
        return pd.Series(dtype=float)
    s_strat = j["strat"].std(ddof=1)
    s_spy = j["bench"].std(ddof=1)
    if s_strat <= 0 or s_spy <= 0 or not np.isfinite(s_strat) or not np.isfinite(s_spy):
        return pd.Series(dtype=float)
    k = s_spy / s_strat
    adj = j["strat"] * k
    return np.log1p(adj).cumsum()


def main() -> None:
    import matplotlib.pyplot as plt

    series_spec: list[tuple[str, pd.Series, str]] = []

    bt5 = load_backtest(5)
    bt20 = load_backtest(20)
    bt60 = load_backtest(60)

    start = min(bt5.index.min(), bt20.index.min(), bt60.index.min())
    end = max(bt5.index.max(), bt20.index.max(), bt60.index.max())
    bench_daily, bench_label = download_benchmark_daily(start, end)

    # --- weekly (I5/R5) strategies from backtest_I5 ---
    spy5 = compound_spy_over_schedule(bench_daily, bt5.index)
    common5 = spy5.index.intersection(bt5.index)

    def add_scaled(name: str, col: str) -> None:
        r = bt5[col].reindex(common5).dropna()
        s = spy5.reindex(r.index).dropna()
        idx = r.index.intersection(s.index)
        c = vol_scaled_cumulative_log(r.reindex(idx), s.reindex(idx))
        if len(c) > 0:
            series_spec.append((name, c, col))

    add_scaled(r"CNN I5/R5 ($\mathrm{EW}$, gross)", "CNN_LS_gross_EW")
    add_scaled(r"CNN I5/R5 ($\mathrm{VW}$, gross)", "CNN_LS_gross_VW")
    add_scaled("Momentum (weekly H–L, gross)", "Mom_LS_gross")
    add_scaled("MA cross (weekly H–L, gross)", "MA_LS_gross")

    # --- CNN at native monthly / quarterly calendars (honest labels) ---
    for h, bt, tag in ((20, bt20, "I20/R20"), (60, bt60, "I60/R60")):
        spy_h = compound_spy_over_schedule(bench_daily, bt.index)
        for col, sfx in (("CNN_LS_gross_EW", "EW"), ("CNN_LS_gross_VW", "VW")):
            common = spy_h.index.intersection(bt.index)
            r = bt[col].reindex(common).astype(float)
            s = spy_h.reindex(common).astype(float)
            idx = r.dropna().index.intersection(s.dropna().index)
            c = vol_scaled_cumulative_log(r.reindex(idx), s.reindex(idx))
            if len(c) > 0:
                series_spec.append((rf"CNN {tag} ($\mathrm{{{sfx}}}$, gross)", c, f"I{h}_{sfx}"))

    # Benchmark buy&hold over same weekly intervals (no vol rescaling vs itself)
    r_spy = spy5.dropna()
    c_spy = np.log1p(r_spy).cumsum()
    series_spec.append((rf"{bench_label} (weekly hold)", c_spy, "SPY"))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PPT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6.2), dpi=120)
    colors = {
        "CNN_LS_gross_EW": "#1f77b4",
        "CNN_LS_gross_VW": "#aec7e8",
        "Mom_LS_gross": "#2ca02c",
        "MA_LS_gross": "#9467bd",
        "I20_EW": "#d62728",
        "I20_VW": "#ff9896",
        "I60_EW": "#000000",
        "I60_VW": "#bbbbbb",
        "SPY": "#ffbb00",
    }
    for name, curve, key in series_spec:
        c = colors.get(key, "#7f7f7f")
        lw = 2.2 if key in {"CNN_LS_gross_EW", "SPY"} else 1.6
        plt.plot(curve.index, curve.values, label=name, color=c, linewidth=lw)

    plt.title(
        f"Volatility-adjusted cumulative log returns (EW L/S; σ matched to {bench_label})",
        fontsize=12,
    )
    plt.xlabel("Date")
    plt.ylabel(r"Cumulative $\sum \log(1+\tilde r_t)$ after vol rescaling")
    plt.axhline(0.0, color="gray", linewidth=0.7, linestyle="-", alpha=0.4)
    plt.legend(loc="upper left", fontsize=8, framealpha=0.92)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out1 = FIG_DIR / "figure5_vol_adjusted_cumlog.png"
    out2 = PPT_DIR / "figure5_vol_adjusted_cumlog.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {out1}")
    print(f"wrote {out2}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
