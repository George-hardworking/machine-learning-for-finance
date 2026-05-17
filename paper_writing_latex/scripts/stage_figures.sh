#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/../final_submit_version/outputs"
DST="$HERE/figures"
mkdir -p "$DST"
# Regenerate Fig.7 for LaTeX (after updating fig7_simple_* CSVs under tables):
#   conda activate 5020_env
#   python final_submit_version/scripts/pipeline_split/redraw_fig7.py
cp "$SRC/pipeline_runs/figures/fig6_I5.png"          "$DST/fig6_I5.png"
cp "$SRC/pipeline_runs/figures/fig6_I20.png"         "$DST/fig6_I20.png"
cp "$SRC/pipeline_runs/figures/fig6_I60.png"         "$DST/fig6_I60.png"
# Fig.7 thesis-style raster (x=1–10 tight, left yticks step 0.10, right 0.05,
# y-limits hug data): regenerate with redraw_fig7.py — do NOT overwrite below
# with notebook previews fig7_simple_*.png (that reverts margins / tick spacing).
cp "$SRC/ppt_images/ppt15_cumulative_returns_L5.png"        "$DST/cum_ret_net_L5.png"
cp "$SRC/ppt_images/ppt15_cumulative_returns_L20.png"       "$DST/cum_ret_net_L20.png"
cp "$SRC/ppt_images/ppt15_cumulative_returns_L60.png"       "$DST/cum_ret_net_L60.png"
cp "$SRC/ppt_images/ppt15_cumulative_returns_gross_L5.png"  "$DST/cum_ret_gross_L5.png"
cp "$SRC/ppt_images/ppt15_cumulative_returns_gross_L20.png" "$DST/cum_ret_gross_L20.png"
cp "$SRC/ppt_images/ppt15_cumulative_returns_gross_L60.png" "$DST/cum_ret_gross_L60.png"
cp "$SRC/pipeline_runs/figures/figure5_vol_adjusted_cumlog.png" "$DST/figure5_vol_adjusted_cumlog.png"
cp "$SRC/pipeline_runs/figures/figure5_vol_adjusted_cumlog_net.png" "$DST/figure5_vol_adjusted_cumlog_net.png"
echo "Staged $(ls "$DST" | wc -l) figure files into $DST"
