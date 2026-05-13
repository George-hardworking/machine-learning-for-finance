"""Generate a three-panel OHLC sample figure (I=5/20/60) for the paper.

Reuses the image-drawing logic and configuration from
``final_submit_version/notebooks/final_improve.ipynb`` so that the figure
embedded in the manuscript is bit-identical to what the model sees at
training time.

Usage (run from any cwd; defaults are relative to the repo)::

    python final_submit_version/scripts/pipeline_split/export_paper_assets.py

Outputs:
    paper_writing_latex/figures/ohlc_sample.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[3]
CACHE_PATH = ROOT_DIR / "final_submit_version" / "outputs" / "cache" / "cleaned_data_all.pkl"
OUT_PATH = ROOT_DIR / "paper_writing_latex" / "figures" / "ohlc_sample.png"

MODEL_CFG = {
    5: {"W": 15, "H": 32, "H_ohlc": 25, "H_vol": 6},
    20: {"W": 60, "H": 64, "H_ohlc": 51, "H_vol": 12},
    60: {"W": 180, "H": 96, "H_ohlc": 76, "H_vol": 19},
}

SEED = 42
ANCHOR_DATE = pd.Timestamp("2010-12-31")


def draw_sample_image(perm_df: pd.DataFrame, t_idx: int, L: int) -> tuple[np.ndarray, int]:
    """Render one OHLC + volume image using the notebook's PIL drawing rules.

    Mirrors ``StockImageDataset.__getitem__`` so the figure shown in the
    paper exactly matches the tensors fed to the CNN at training time.
    """

    cfg = MODEL_CFG[L]
    P_adj = perm_df[["dlyopen_adj", "dlyhigh_adj", "dlylow_adj", "dlyclose_adj"]].values
    V = perm_df["dlyvol"].values
    label = int(perm_df[f"Label_{L}"].values[t_idx])

    P_full = P_adj[t_idx - 2 * L + 1 : t_idx]
    V_win = V[t_idx - L : t_idx]
    P_win = P_full[-L:]

    sma_vals = np.zeros(L)
    for tau in range(L):
        sma_vals[tau] = P_full[tau : tau + L, 3].mean()

    P_min = min(P_win.min(), sma_vals.min())
    P_max = max(P_win.max(), sma_vals.max())
    P_range = P_max - P_min + 1e-8

    ohlc_h = cfg["H_ohlc"]
    vol_h = cfg["H_vol"]
    total_w = cfg["W"]

    def ret_to_y(ret: float) -> int:
        pixels_per_unit = (ohlc_h - 1.0) / P_range
        return int(np.around((ret - P_min) * pixels_per_unit))

    ohlc_img = Image.new("L", (total_w, ohlc_h), 0)
    draw_ohlc = ImageDraw.Draw(ohlc_img)
    pixels_ohlc = ohlc_img.load()

    for tau in range(L - 1):
        x0 = tau * 3 + 1
        x1 = (tau + 1) * 3 + 1
        y0 = ret_to_y(sma_vals[tau])
        y1 = ret_to_y(sma_vals[tau + 1])
        draw_ohlc.line((x0, y0, x1, y1), width=1, fill=255)

    for tau in range(L):
        x_c = tau * 3 + 1
        y_high = ret_to_y(P_win[tau, 1])
        y_low = ret_to_y(P_win[tau, 2])
        y_open = ret_to_y(P_win[tau, 0])
        y_close = ret_to_y(P_win[tau, 3])
        for j in range(min(y_low, y_high), max(y_low, y_high) + 1):
            pixels_ohlc[x_c, j] = 255
        for i in range(x_c - 1, x_c + 1):
            pixels_ohlc[i, y_open] = 255
        pixels_ohlc[x_c + 1, y_close] = 255

    ohlc_img = ohlc_img.transpose(Image.FLIP_TOP_BOTTOM)

    vol_img = Image.new("L", (total_w, vol_h), 0)
    pixels_vol = vol_img.load()
    V_max = np.nanmax(V_win) if V_win.size else 0.0
    if (not np.isnan(V_max)) and V_max != 0:
        pixels_per_volume = vol_h / abs(V_max)
        for tau in range(L):
            if np.isnan(V_win[tau]):
                continue
            v_h_curr = int(np.around(abs(V_win[tau]) * pixels_per_volume))
            v_h_curr = max(1, min(vol_h, v_h_curr))
            x_c = tau * 3 + 1
            for j in range(vol_h - v_h_curr, vol_h):
                pixels_vol[x_c, j] = 255

    full = Image.new("L", (total_w, cfg["H"]), 0)
    full.paste(ohlc_img, (0, 0))
    full.paste(vol_img, (0, ohlc_h + 1))
    return np.asarray(full, dtype=np.uint8), label


def pick_anchor_index(perm_df: pd.DataFrame, L: int) -> int:
    """Pick an in-range index near ``ANCHOR_DATE`` with enough history for L."""
    dates = pd.to_datetime(perm_df["dlycaldt"]).values
    target = np.datetime64(ANCHOR_DATE)
    idx = int(np.searchsorted(dates, target))
    idx = max(2 * L, min(len(perm_df) - L - 1, idx))
    return idx


def main() -> int:
    if not CACHE_PATH.exists():
        sys.stderr.write(f"[error] cleaned cache not found at {CACHE_PATH}\n")
        return 1

    print(f"Loading {CACHE_PATH} (this may take a minute) ...")
    df = pd.read_pickle(CACHE_PATH)
    needed = [
        "permno",
        "dlycaldt",
        "dlyopen_adj",
        "dlyhigh_adj",
        "dlylow_adj",
        "dlyclose_adj",
        "dlyvol",
        "Label_5",
        "Label_20",
        "Label_60",
    ]
    df = df[needed].dropna()
    df["dlycaldt"] = pd.to_datetime(df["dlycaldt"])
    df = df.sort_values(["permno", "dlycaldt"]).reset_index(drop=True)

    counts = df.groupby("permno").size()
    eligible = counts[counts >= 2 * max(MODEL_CFG) + max(MODEL_CFG) + 10].index.to_numpy()
    if eligible.size == 0:
        sys.stderr.write("[error] no permno has enough history for I=60 image\n")
        return 2

    rng = np.random.default_rng(SEED)
    chosen = int(rng.choice(eligible))
    perm_df = df[df["permno"] == chosen].reset_index(drop=True)
    print(f"Selected permno={chosen} with {len(perm_df)} daily rows")

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), dpi=300)
    for ax, L in zip(axes, [5, 20, 60]):
        idx = pick_anchor_index(perm_df, L)
        if idx - 2 * L + 1 < 0 or idx + L >= len(perm_df):
            sys.stderr.write(f"[error] insufficient history for I={L} at idx={idx}\n")
            return 3
        img, lab = draw_sample_image(perm_df, idx, L)
        ax.imshow(img, cmap="gray", origin="upper", interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"I={L}\nlabel={'up' if lab == 1 else 'down'}", fontsize=10)

    fig.suptitle(
        f"OHLC + volume image bundle (permno {chosen}, anchor {ANCHOR_DATE.date()})",
        fontsize=11,
        y=1.02,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
