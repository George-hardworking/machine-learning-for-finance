"""Fama--French six-factor spanning regression for the H--L CNN spread.

Reads cached backtest tables produced by the main notebook
(``outputs/cache/backtest_I{5,20,60}.pkl``), downloads the daily
Fama--French 5 + Momentum factors with ``pandas_datareader``, compounds
the factors over each rebalance interval to match the holding period,
and regresses each horizon's net long--short spread ``CNN_LS_net`` on
the six factors.

Writes one CSV per horizon at::

    final_submit_version/outputs/pipeline_runs/tables/ff_regression_I{L}.csv

If the cache is missing or the FF download fails, the script exits with
a non-zero status and a clear stderr message rather than half-writing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT_DIR / "final_submit_version" / "outputs" / "cache"
TABLE_DIR = ROOT_DIR / "final_submit_version" / "outputs" / "pipeline_runs" / "tables"

PERIODS_PER_YEAR = {5: 52, 20: 12, 60: 4}
FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _strip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize_index(ix) -> pd.DatetimeIndex:
    return pd.to_datetime(ix, utc=False).tz_localize(None).normalize()


def load_period_ret(horizon: int) -> pd.DataFrame:
    path = CACHE_DIR / f"backtest_I{horizon}.pkl"
    if not path.exists():
        sys.stderr.write(f"[error] missing backtest cache: {path}\n")
        sys.exit(2)
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        sys.stderr.write(f"[error] unexpected pickle type for I={horizon}: {type(obj)}\n")
        sys.exit(2)
    if "CNN_LS_net" not in obj.columns:
        sys.stderr.write(
            f"[error] CNN_LS_net column missing in backtest_I{horizon}.pkl; "
            f"found {list(obj.columns)}\n"
        )
        sys.exit(2)
    obj = obj.copy()
    obj.index = _normalize_index(obj.index)
    obj = obj.sort_index()
    return obj


def download_ff6_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import pandas_datareader.data as web
    except ImportError:
        sys.stderr.write("[error] pandas_datareader is not installed; pip install pandas-datareader\n")
        sys.exit(3)
    pad = pd.Timedelta(days=35)
    try:
        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start - pad, end=end + pad
        )[0]
        mom = web.DataReader(
            "F-F_Momentum_Factor_daily", "famafrench", start=start - pad, end=end + pad
        )[0]
    except Exception as exc:  # noqa: BLE001 - we want to bail on any download error
        sys.stderr.write(f"[error] could not download Fama--French factors: {exc}\n")
        sys.exit(4)
    ff5 = _strip(ff5)
    mom = _strip(mom)
    if "Mom" not in mom.columns and len(mom.columns) == 1:
        mom.columns = ["Mom"]
    elif "WML" in mom.columns and "Mom" not in mom.columns:
        mom = mom.rename(columns={"WML": "Mom"})
    ff6 = ff5.join(mom, how="inner") / 100.0
    ff6.index = _normalize_index(ff6.index)
    return ff6.sort_index()


def compound_to_periods(ff6_daily: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    rows = []
    pad_days = max(14, horizon * 3)
    for i, t_start in enumerate(dates):
        t_end = dates[i + 1] if i + 1 < len(dates) else t_start + pd.Timedelta(days=pad_days)
        cut = ff6_daily[(ff6_daily.index >= t_start) & (ff6_daily.index < t_end)]
        if cut.empty:
            continue
        compounded = (1 + cut).prod() - 1
        compounded.name = t_start
        rows.append(compounded)
    if not rows:
        sys.stderr.write("[error] compounded FF6 panel is empty after alignment\n")
        sys.exit(5)
    out = pd.DataFrame({s.name: s for s in rows}).T
    out.index = _normalize_index(out.index)
    return _strip(out)


def regress_one(horizon: int) -> pd.DataFrame:
    import statsmodels.api as sm

    period_ret = load_period_ret(horizon)
    ff6_daily = download_ff6_daily(period_ret.index.min(), period_ret.index.max())
    ff6_period = compound_to_periods(ff6_daily, period_ret.index, horizon)

    join = period_ret[["CNN_LS_net"]].join(ff6_period, how="inner").dropna(how="any")
    missing = [c for c in FACTOR_NAMES + ["RF"] if c not in join.columns]
    if missing:
        sys.stderr.write(f"[error] FF6 join missing columns: {missing}\n")
        sys.exit(6)
    if len(join) < max(15, len(FACTOR_NAMES) + 5):
        sys.stderr.write(f"[error] too few aligned rows for I={horizon}: n={len(join)}\n")
        sys.exit(7)

    X = sm.add_constant(join[FACTOR_NAMES])
    y = join["CNN_LS_net"] - join["RF"]
    res = sm.OLS(y, X).fit()

    ppy = PERIODS_PER_YEAR[horizon]
    rows = []
    rows.append(("alpha_ann_pct", res.params["const"] * ppy * 100.0))
    rows.append(("alpha_t", res.tvalues["const"]))
    for f in FACTOR_NAMES:
        rows.append((f"beta_{f}", res.params[f]))
        rows.append((f"t_{f}", res.tvalues[f]))
    rows.append(("R2", res.rsquared))
    rows.append(("R2_adj", res.rsquared_adj))
    rows.append(("n_obs", float(int(res.nobs))))
    return pd.DataFrame(rows, columns=["statistic", f"I{horizon}"]).set_index("statistic")


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for L in (5, 20, 60):
        print(f"[ff] running I={L} ...")
        df = regress_one(L)
        out_path = TABLE_DIR / f"ff_regression_I{L}.csv"
        df.to_csv(out_path)
        print(f"  wrote {out_path}")
        summary.append(df)
    merged = pd.concat(summary, axis=1)
    merged_path = TABLE_DIR / "ff_regression_combined.csv"
    merged.to_csv(merged_path)
    print(f"[ff] wrote combined table at {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
