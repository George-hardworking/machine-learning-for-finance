#!/usr/bin/env python3
"""Regenerate PPT 15 cumulative *log* return PNGs from cached backtest pickles only."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "cache"
PPT_OUT = ROOT / "outputs" / "ppt_images"


def cum_log(s: pd.Series) -> pd.Series:
    return np.log1p(s.astype(float)).cumsum()


def main() -> None:
    PPT_OUT.mkdir(parents=True, exist_ok=True)
    for h in (5, 20, 60):
        p = CACHE / f"backtest_I{h}.pkl"
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        target_ret = pd.read_pickle(p)
        if not isinstance(target_ret, pd.DataFrame):
            print(f"[skip] bad pickle (not DataFrame): {p}")
            continue
        title_suffix = f"L={h}"

        plt.figure(figsize=(10, 5))
        plt.plot(target_ret.index, cum_log(target_ret["CNN_LS_net_EW"]), label="CNN Portfolio EW (Net)", color="#c0392b", linewidth=2)
        plt.plot(target_ret.index, cum_log(target_ret["Mom_LS_net"]), label="Momentum Baseline (Net)", color="#7f8c8d", linestyle="--")
        plt.title(f"Cumulative Log Returns — Net of Costs ({title_suffix})", fontsize=12)
        plt.ylabel("Cumulative log return (sum of log(1+r))")
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        net_name = f"ppt15_cumulative_returns_L{h}.png"
        plt.savefig(PPT_OUT / net_name, dpi=300)
        plt.close()
        print(f"[ok] {PPT_OUT / net_name}")

        plt.figure(figsize=(10, 5))
        plt.plot(target_ret.index, cum_log(target_ret["CNN_LS_gross_EW"]), label="CNN Portfolio EW (Gross)", color="#c0392b", linewidth=2)
        plt.plot(target_ret.index, cum_log(target_ret["Mom_LS_gross"]), label="Momentum Baseline (Gross)", color="#7f8c8d", linestyle="--")
        plt.title(f"Cumulative Log Returns — Gross of Costs ({title_suffix})", fontsize=12)
        plt.ylabel("Cumulative log return (sum of log(1+r))")
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
