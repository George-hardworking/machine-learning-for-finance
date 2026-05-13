#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd


PREDICT_HORIZON_TO_TRADE_FREQ = {5: "week", 20: "month", 60: "quarter"}
PERIODS_PER_YEAR = {5: 52, 20: 12, 60: 4}

CACHE_DIR = Path("/root/machine-learning-for-finance/Final/outputs/cache")
OUT_DIR = Path("/root/machine-learning-for-finance/Final/outputs/pipeline_runs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def period_end_timestamps_from_calendar(all_dates: pd.Series, trade_freq: str):
    uniq = pd.Series(pd.to_datetime(all_dates).unique()).sort_values()
    ddf = pd.DataFrame({"d": uniq})
    if trade_freq == "week":
        ddf["p"] = ddf["d"].dt.to_period("W-FRI")
    elif trade_freq == "month":
        ddf["p"] = ddf["d"].dt.to_period("M")
    elif trade_freq == "quarter":
        ddf["p"] = ddf["d"].dt.to_period("Q")
    else:
        raise ValueError(f"invalid trade_freq={trade_freq}")
    ends = ddf.groupby("p", sort=True)["d"].max()
    return set(pd.Timestamp(x).normalize() for x in ends.values)


def build_valid_indices(df_test: pd.DataFrame, L: int, F: int):
    trade_freq = PREDICT_HORIZON_TO_TRADE_FREQ[F]
    period_end_set = period_end_timestamps_from_calendar(df_test["dlycaldt"], trade_freq)
    valid_indices = []
    grp = df_test.groupby("permno", group_keys=False)
    for _, g in grp:
        idxs = g.index.values
        if len(idxs) < 2 * L - 1 + F:
            continue
        for t_idx in idxs[2 * L - 1 : len(idxs) - F]:
            ts = pd.Timestamp(df_test.at[t_idx, "dlycaldt"]).normalize()
            if ts in period_end_set:
                valid_indices.append(int(t_idx))
    return valid_indices


def enrich_rebalance_panel_with_daily_baselines(df_sparse: pd.DataFrame, df_daily: pd.DataFrame, F_horizon: int):
    d = df_daily.sort_values(["permno", "dlycaldt"]).copy()
    d["mom_score"] = d.groupby("permno")["dlyclose_adj"].pct_change(F_horizon)
    ma_fast = d.groupby("permno")["dlyclose_adj"].transform(
        lambda x: x.rolling(max(2, F_horizon // 2), min_periods=1).mean()
    )
    ma_slow = d.groupby("permno")["dlyclose_adj"].transform(
        lambda x: x.rolling(max(20, F_horizon), min_periods=1).mean()
    )
    d["ma_score"] = (ma_fast / ma_slow - 1).fillna(0)
    sub = d[["permno", "dlycaldt", "mom_score", "ma_score"]]
    out = df_sparse.merge(sub, on=["permno", "dlycaldt"], how="left")
    out["mom_score"] = out["mom_score"].fillna(0)
    out["ma_score"] = out["ma_score"].fillna(0)
    return out


def calc_decile_metrics(df_test_valid: pd.DataFrame, F_horizon: int, score_col: str):
    df = df_test_valid.copy()
    df["decile"] = df.groupby("dlycaldt")[score_col].rank(pct=True, ascending=True)
    df["decile"] = (df["decile"] * 10).apply(np.ceil).astype(int)
    decile_stats = (
        df.groupby("decile")[f"R_fut_{F_horizon}"]
        .agg(mean_ret="mean", std_ret="std")
        .reset_index()
        .sort_values("decile")
    )
    ppy = PERIODS_PER_YEAR[F_horizon]
    decile_stats["ann_ret"] = decile_stats["mean_ret"] * ppy
    decile_stats["ann_vol"] = decile_stats["std_ret"] * np.sqrt(ppy)
    return decile_stats


def main():
    cleaned_path = CACHE_DIR / "cleaned_data_all.pkl"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"missing cache: {cleaned_path}")

    df = pd.read_pickle(cleaned_path)
    # Match notebook split policy: tv=1993-2000, test=2001-2019
    df = df[(df["dlycaldt"] >= "1993-01-01") & (df["dlycaldt"] <= "2019-12-31")].copy()
    df_test = df[(df["dlycaldt"] >= "2001-01-01") & (df["dlycaldt"] <= "2019-12-31")].copy().reset_index(drop=True)

    for h in (5, 20, 60):
        preds_path = CACHE_DIR / f"preds_I{h}.npy"
        if not preds_path.exists():
            print(f"[skip] missing preds: {preds_path}")
            continue

        valid_indices = build_valid_indices(df_test, L=h, F=h)
        preds = np.load(preds_path)
        n = min(len(valid_indices), len(preds))
        if n == 0:
            print(f"[skip] no valid rows for I{h}")
            continue
        if len(valid_indices) != len(preds):
            print(f"[warn] I{h} length mismatch: valid={len(valid_indices)} preds={len(preds)}; using n={n}")

        panel = df_test.iloc[valid_indices[:n]].copy()
        panel["pred_score"] = preds[:n]
        panel = enrich_rebalance_panel_with_daily_baselines(panel, df_test, h)

        cnn = calc_decile_metrics(panel, h, "pred_score")
        mom = calc_decile_metrics(panel, h, "mom_score")
        ma = calc_decile_metrics(panel, h, "ma_score")

        cnn.to_csv(OUT_DIR / f"fig6_I{h}_cnn.csv", index=False)
        mom.to_csv(OUT_DIR / f"fig6_I{h}_mom.csv", index=False)
        ma.to_csv(OUT_DIR / f"fig6_I{h}_ma.csv", index=False)
        print(f"[ok] wrote fig6 I{h} CSVs")


if __name__ == "__main__":
    main()
