#!/usr/bin/env python3
"""
Redraw Figure 6 (decile return / volatility) from cached CSV files.

Changes versus the original notebook plot:
  - x-axis: no extra blank margins on left/right (tight to decile 1-10)
  - y-axis: auto-range tight to the data (no fixed bounds)
  - Return subplot  (left):  y-tick step 0.10 (was 0.05)
  - Volatility subplot (right): y-tick step 0.05 (was 0.10)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # repo root
TABLE_DIR = ROOT / "final_submit_version" / "outputs" / "pipeline_runs" / "tables"
FIG_OUT_DIR = ROOT / "paper_writing_latex" / "figures"

HORIZONS = [5, 20, 60]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def load(horizon: int, signal: str) -> pd.DataFrame:
    p = TABLE_DIR / f"fig6_I{horizon}_{signal}.csv"
    return pd.read_csv(p)


def tight_ylim(ax, *series_list):
    """Set y-limits exactly to the data min/max across all series, then apply
    a MultipleLocator so tick lines fall at clean intervals — but the axis
    boundary itself never extends beyond the data range."""
    all_vals = np.concatenate([s.values for s in series_list])
    ymin, ymax = all_vals.min(), all_vals.max()
    ax.set_ylim(ymin, ymax)


def redraw(horizon: int) -> None:
    cnn = load(horizon, "cnn")
    mom = load(horizon, "mom")
    ma  = load(horizon, "ma")

    decile_x = np.arange(1, 11)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # ---- left: annualised return ----
    ax1.plot(decile_x, cnn["ann_ret"], "o-",  color="#d62728", lw=2,   ms=6, label=f"CNN (I{horizon}/R{horizon})", clip_on=False)
    ax1.plot(decile_x, mom["ann_ret"], "s--", color="#1f77b4", lw=1.5, ms=5, label="Baseline Momentum",           clip_on=False)
    ax1.plot(decile_x, ma["ann_ret"],  "^-.", color="#2ca02c", lw=1.5, ms=5, label="Baseline MA Cross",           clip_on=False)

    ax1.set_title("Average Realized Return by Decile", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Signal Decile", fontsize=11)
    ax1.set_ylabel("Annualized Return", fontsize=11)
    ax1.set_xticks(decile_x)
    ax1.set_xlim(1, 10)                                               # x-axis exactly 1→10
    tight_ylim(ax1, cnn["ann_ret"], mom["ann_ret"], ma["ann_ret"])    # y exactly data range
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.10))         # step = 0.10
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.axhline(y=0, color="black", lw=0.8, alpha=0.5)
    ax1.legend(fontsize=10)

    # ---- right: annualised volatility ----
    ax2.plot(decile_x, cnn["ann_vol"], "o-",  color="#d62728", lw=2,   ms=6, label=f"CNN (I{horizon}/R{horizon})", clip_on=False)
    ax2.plot(decile_x, mom["ann_vol"], "s--", color="#1f77b4", lw=1.5, ms=5, label="Baseline Momentum",            clip_on=False)
    ax2.plot(decile_x, ma["ann_vol"],  "^-.", color="#2ca02c", lw=1.5, ms=5, label="Baseline MA Cross",            clip_on=False)

    ax2.set_title("Return Volatility by Decile", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Signal Decile", fontsize=11)
    ax2.set_ylabel("Annualized Volatility", fontsize=11)
    ax2.set_xticks(decile_x)
    ax2.set_xlim(1, 10)                                               # x-axis exactly 1→10
    tight_ylim(ax2, cnn["ann_vol"], mom["ann_vol"], ma["ann_vol"])    # y exactly data range
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))         # step = 0.05
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(fontsize=10)

    fig.suptitle(
        f"Figure 6. Prediction Accuracy by Decile (I{horizon}/R{horizon})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    out = FIG_OUT_DIR / f"fig6_I{horizon}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig6] saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    for h in HORIZONS:
        redraw(h)
    print("Done.")
