#!/usr/bin/env python3
"""Jiang et al. (2023) Figure 5–style panel: volatility-matched cumulative log returns.

See ``vol_matched_cumlog.py`` for construction steps. Cached I20/I60 use monthly/quarterly
rebalance (I=R), not the paper's I*/R5 weekly overlay.

Outputs (300 dpi PNG):

- ``figure5_vol_adjusted_cumlog.png`` — gross of transaction costs
- ``figure5_vol_adjusted_cumlog_net.png`` — net of proportional costs (same σ-matching)

Also copies to ``final_submit_version/outputs/ppt_images/``.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from vol_matched_cumlog import (
    compound_spy_over_schedule,
    download_benchmark_daily,
    load_backtest,
    vol_scaled_cumulative_log,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "final_submit_version" / "outputs" / "pipeline_runs" / "figures"
PPT_DIR = REPO_ROOT / "final_submit_version" / "outputs" / "ppt_images"


def _build_series_spec(*, gross: bool) -> list[tuple[str, object, str]]:
    """Return list of (legend math name, cumulative log series, color key)."""
    series_spec: list[tuple[str, object, str]] = []
    cost_tag = "gross" if gross else "net"
    bt5 = load_backtest(5)
    bt20 = load_backtest(20)
    bt60 = load_backtest(60)

    start = min(bt5.index.min(), bt20.index.min(), bt60.index.min())
    end = max(bt5.index.max(), bt20.index.max(), bt60.index.max())
    bench_daily, bench_label = download_benchmark_daily(start, end)

    spy5 = compound_spy_over_schedule(bench_daily, bt5.index)
    common5 = spy5.index.intersection(bt5.index)

    def add_scaled(name: str, col: str, color_key: str) -> None:
        r = bt5[col].reindex(common5).dropna()
        s = spy5.reindex(r.index).dropna()
        idx = r.index.intersection(s.index)
        c = vol_scaled_cumulative_log(r.reindex(idx), s.reindex(idx))
        if len(c) > 0:
            series_spec.append((name, c, color_key))

    cnn_ew = f"CNN_LS_{cost_tag}_EW"
    cnn_vw = f"CNN_LS_{cost_tag}_VW"
    add_scaled(rf"CNN I5/R5 ($\mathrm{{EW}}$, {cost_tag})", cnn_ew, "CNN_LS_gross_EW")
    add_scaled(rf"CNN I5/R5 ($\mathrm{{VW}}$, {cost_tag})", cnn_vw, "CNN_LS_gross_VW")
    add_scaled(rf"Momentum (weekly H–L, {cost_tag})", f"Mom_LS_{cost_tag}", "Mom_LS_gross")
    add_scaled(rf"MA cross (weekly H–L, {cost_tag})", f"MA_LS_{cost_tag}", "MA_LS_gross")

    for h, bt, horizon_label in ((20, bt20, "I20/R20"), (60, bt60, "I60/R60")):
        spy_h = compound_spy_over_schedule(bench_daily, bt.index)
        for sfx in ("EW", "VW"):
            col = f"CNN_LS_{cost_tag}_" + sfx
            common = spy_h.index.intersection(bt.index)
            r = bt[col].reindex(common).astype(float)
            s = spy_h.reindex(common).astype(float)
            idx = r.dropna().index.intersection(s.dropna().index)
            c = vol_scaled_cumulative_log(r.reindex(idx), s.reindex(idx))
            if len(c) > 0:
                color_key = f"I{h}_{sfx}"
                series_spec.append(
                    (rf"CNN {horizon_label} ($\mathrm{{{sfx}}}$, {cost_tag})", c, color_key)
                )

    r_spy = spy5.dropna()
    c_spy = np.log1p(r_spy).cumsum()
    series_spec.append((rf"{bench_label} (weekly hold)", c_spy, "SPY"))
    return series_spec


def _plot_and_save(series_spec: list[tuple[str, object, str]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

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

    plt.axhline(0.0, color="gray", linewidth=0.7, linestyle="-", alpha=0.4)
    plt.legend(loc="upper left", fontsize=8, framealpha=0.92)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main() -> None:
    gross_spec = _build_series_spec(gross=True)
    net_spec = _build_series_spec(gross=False)

    out_gross = FIG_DIR / "figure5_vol_adjusted_cumlog.png"
    out_net = FIG_DIR / "figure5_vol_adjusted_cumlog_net.png"
    _plot_and_save(gross_spec, out_gross)
    _plot_and_save(net_spec, out_net)

    PPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_gross, PPT_DIR / "figure5_vol_adjusted_cumlog.png")
    shutil.copy2(out_net, PPT_DIR / "figure5_vol_adjusted_cumlog_net.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
