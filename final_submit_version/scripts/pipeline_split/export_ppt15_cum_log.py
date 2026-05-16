#!/usr/bin/env python3
"""Regenerate PPT 15 PNGs: σ-matched cumulative log returns (Jiang Figure 5 style).

Uses cached ``backtest_I{5,20,60}.pkl`` plus SPY or Fama--French daily market
(see ``vol_matched_cumlog.py``). One figure per horizon × (net | gross): CNN EW vs Momentum.

Title lines are kept only as comments in source (no matplotlib title shown), per paper style.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vol_matched_cumlog import (
    compound_spy_over_schedule,
    download_benchmark_daily,
    load_backtest,
    vol_scaled_cumulative_log,
)

SUBMIT_ROOT = Path(__file__).resolve().parents[2]
PPT_OUT = SUBMIT_ROOT / "outputs" / "ppt_images"


def _curve(
    bt: pd.DataFrame,
    bench_daily: pd.Series,
    strat_col: str,
) -> pd.Series:
    bench_h = compound_spy_over_schedule(bench_daily, bt.index)
    common = bench_h.index.intersection(bt.index)
    r = bt[strat_col].reindex(common).astype(float)
    s = bench_h.reindex(common).astype(float)
    idx = r.dropna().index.intersection(s.dropna().index)
    return vol_scaled_cumulative_log(r.reindex(idx), s.reindex(idx))


def main() -> None:
    PPT_OUT.mkdir(parents=True, exist_ok=True)

    bt5, bt20, bt60 = load_backtest(5), load_backtest(20), load_backtest(60)
    start = min(bt5.index.min(), bt20.index.min(), bt60.index.min())
    end = max(bt5.index.max(), bt20.index.max(), bt60.index.max())
    bench_daily, _bench_label = download_benchmark_daily(start, end)

    for h, bt in ((5, bt5), (20, bt20), (60, bt60)):
        title_suffix = f"L={h}"

        plt.figure(figsize=(10, 5))
        cnn_n = _curve(bt, bench_daily, "CNN_LS_net_EW")
        mom_n = _curve(bt, bench_daily, "Mom_LS_net")
        plt.plot(cnn_n.index, cnn_n.values, label="CNN Portfolio EW (Net, σ-matched)", color="#c0392b", linewidth=2)
        plt.plot(mom_n.index, mom_n.values, label="Momentum Baseline (Net, σ-matched)", color="#7f8c8d", linestyle="--")
        # plt.title(f"Vol-matched cumulative log — Net of costs ({title_suffix})", fontsize=12)
        plt.ylabel(r"Cumulative $\sum \log(1+\tilde r_t)$ (σ matched to benchmark)")
        plt.xlabel("Date")
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        net_name = f"ppt15_cumulative_returns_L{h}.png"
        plt.savefig(PPT_OUT / net_name, dpi=300)
        plt.close()
        print(f"[ok] {PPT_OUT / net_name}")

        plt.figure(figsize=(10, 5))
        cnn_g = _curve(bt, bench_daily, "CNN_LS_gross_EW")
        mom_g = _curve(bt, bench_daily, "Mom_LS_gross")
        plt.plot(cnn_g.index, cnn_g.values, label="CNN Portfolio EW (Gross, σ-matched)", color="#c0392b", linewidth=2)
        plt.plot(mom_g.index, mom_g.values, label="Momentum Baseline (Gross, σ-matched)", color="#7f8c8d", linestyle="--")
        # plt.title(f"Vol-matched cumulative log — Gross ({title_suffix})", fontsize=12)
        plt.ylabel(r"Cumulative $\sum \log(1+\tilde r_t)$ (σ matched to benchmark)")
        plt.xlabel("Date")
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        gross_name = f"ppt15_cumulative_returns_gross_L{h}.png"
        plt.savefig(PPT_OUT / gross_name, dpi=300)
        plt.close()
        print(f"[ok] {PPT_OUT / gross_name}")


if __name__ == "__main__":
    main()
