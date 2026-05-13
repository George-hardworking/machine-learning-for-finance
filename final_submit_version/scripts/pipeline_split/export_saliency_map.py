"""Standalone saliency-map export for the I=60 CNN.

Reuses ``MODEL_CFG`` and the I60 architecture from the main notebook,
loads one trained ensemble member, draws OHLC images for a few stocks
from the test window (post-2001), keeps only those for which the model
is very confident "up", and renders an overlay of input gradient
saliency.

Output: ``paper_writing_latex/figures/saliency_I60.png``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[3]
CACHE_PATH = ROOT_DIR / "final_submit_version" / "outputs" / "cache" / "cleaned_data_all.pkl"
MODEL_DIR = ROOT_DIR / "final_submit_version" / "outputs" / "models"
CHECKPOINT = MODEL_DIR / "I60_checkpoint42.pth.tar"
MEAN_STD = MODEL_DIR / "mean_std_I60_R60_train.npz"
OUT_PATH = ROOT_DIR / "paper_writing_latex" / "figures" / "saliency_I60.png"

L = 60
F = 60
CFG = {"W": 180, "H": 96, "H_ohlc": 76, "H_vol": 19}
TEST_START = pd.Timestamp("2002-01-01")
TEST_END = pd.Timestamp("2019-12-31")
NUM_SAMPLES = 3
NUM_PERMNOS_TO_SCAN = 60
CONFIDENCE_THRESHOLD = 0.85
SEED = 42


class I60Model(nn.Module):
    """4-block CNN matching the I=60 head used at training time."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(3, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(184320, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.fc(x)


def draw_image(perm_df: pd.DataFrame, t_idx: int) -> tuple[np.ndarray, int]:
    P_adj = perm_df[["dlyopen_adj", "dlyhigh_adj", "dlylow_adj", "dlyclose_adj"]].values
    V = perm_df["dlyvol"].values
    label = int(perm_df[f"Label_{F}"].values[t_idx])

    P_full = P_adj[t_idx - 2 * L + 1 : t_idx]
    V_win = V[t_idx - L : t_idx]
    P_win = P_full[-L:]

    sma = np.array([P_full[tau : tau + L, 3].mean() for tau in range(L)])
    P_min = min(P_win.min(), sma.min())
    P_max = max(P_win.max(), sma.max())
    P_range = P_max - P_min + 1e-8

    ohlc_h = CFG["H_ohlc"]
    vol_h = CFG["H_vol"]
    total_w = CFG["W"]

    def ret_to_y(ret: float) -> int:
        return int(np.around((ret - P_min) * (ohlc_h - 1.0) / P_range))

    ohlc_img = Image.new("L", (total_w, ohlc_h), 0)
    draw_o = ImageDraw.Draw(ohlc_img)
    px_o = ohlc_img.load()
    for tau in range(L - 1):
        draw_o.line(
            (tau * 3 + 1, ret_to_y(sma[tau]), (tau + 1) * 3 + 1, ret_to_y(sma[tau + 1])),
            width=1,
            fill=255,
        )
    for tau in range(L):
        x_c = tau * 3 + 1
        y_h = ret_to_y(P_win[tau, 1])
        y_l = ret_to_y(P_win[tau, 2])
        y_o = ret_to_y(P_win[tau, 0])
        y_close = ret_to_y(P_win[tau, 3])
        for j in range(min(y_l, y_h), max(y_l, y_h) + 1):
            px_o[x_c, j] = 255
        for i in range(x_c - 1, x_c + 1):
            px_o[i, y_o] = 255
        px_o[x_c + 1, y_close] = 255
    ohlc_img = ohlc_img.transpose(Image.FLIP_TOP_BOTTOM)

    vol_img = Image.new("L", (total_w, vol_h), 0)
    px_v = vol_img.load()
    V_max = np.nanmax(V_win) if V_win.size else 0.0
    if (not np.isnan(V_max)) and V_max != 0:
        pixels_per_volume = vol_h / abs(V_max)
        for tau in range(L):
            if np.isnan(V_win[tau]):
                continue
            v_h_curr = max(1, min(vol_h, int(np.around(abs(V_win[tau]) * pixels_per_volume))))
            x_c = tau * 3 + 1
            for j in range(vol_h - v_h_curr, vol_h):
                px_v[x_c, j] = 255

    full = Image.new("L", (total_w, CFG["H"]), 0)
    full.paste(ohlc_img, (0, 0))
    full.paste(vol_img, (0, ohlc_h + 1))
    return np.asarray(full, dtype=np.uint8), label


def normalize(img_u8: np.ndarray, mu: float, std: float) -> np.ndarray:
    img = img_u8.astype(np.float32) / 255.0
    return (img - mu) / (std + 1e-8)


def load_model() -> I60Model:
    if not CHECKPOINT.exists():
        sys.stderr.write(f"[error] missing checkpoint {CHECKPOINT}\n")
        sys.exit(2)
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    cleaned = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model = I60Model()
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        sys.stderr.write(f"[warn] missing={missing} unexpected={unexpected}\n")
    model.eval()
    return model


def main() -> int:
    if not CACHE_PATH.exists():
        sys.stderr.write(f"[error] missing cleaned cache {CACHE_PATH}\n")
        return 1
    if not MEAN_STD.exists():
        sys.stderr.write(f"[error] missing norm stats {MEAN_STD}\n")
        return 1

    arr = np.load(MEAN_STD)
    mu_train = float(arr["mean"])
    std_train = float(arr["std"])
    print(f"train norm: mu={mu_train:.4f} std={std_train:.4f}")

    print(f"loading {CACHE_PATH} ...")
    df = pd.read_pickle(CACHE_PATH)
    needed = [
        "permno",
        "dlycaldt",
        "dlyopen_adj",
        "dlyhigh_adj",
        "dlylow_adj",
        "dlyclose_adj",
        "dlyvol",
        f"Label_{F}",
    ]
    df = df[needed].dropna()
    df["dlycaldt"] = pd.to_datetime(df["dlycaldt"])
    df = df.sort_values(["permno", "dlycaldt"]).reset_index(drop=True)

    test_mask = (df["dlycaldt"] >= TEST_START) & (df["dlycaldt"] <= TEST_END)
    counts_in_test = df[test_mask].groupby("permno").size()
    eligible = counts_in_test[counts_in_test >= 2 * L + F + 10].index.to_numpy()
    if eligible.size == 0:
        sys.stderr.write("[error] no permno has enough history in test window\n")
        return 3

    rng = np.random.default_rng(SEED)
    if eligible.size > NUM_PERMNOS_TO_SCAN:
        eligible = rng.choice(eligible, size=NUM_PERMNOS_TO_SCAN, replace=False)

    print("loading I60 model ...")
    model = load_model()

    picks: list[tuple[np.ndarray, int, float, int]] = []
    for permno in eligible:
        perm_df = df[df["permno"] == permno].reset_index(drop=True)
        idx_in_test = perm_df.index[perm_df["dlycaldt"].between(TEST_START, TEST_END)].to_numpy()
        idx_in_test = idx_in_test[(idx_in_test >= 2 * L) & (idx_in_test < len(perm_df) - F)]
        if idx_in_test.size == 0:
            continue
        sample_idxs = rng.choice(idx_in_test, size=min(20, idx_in_test.size), replace=False)
        for t_idx in sample_idxs:
            img_u8, lab = draw_image(perm_df, int(t_idx))
            if lab != 1:
                continue
            img_norm = normalize(img_u8, mu_train, std_train)
            if float(np.std(img_norm)) < 0.25:
                continue
            x = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                prob_up = torch.softmax(logits, dim=1)[0, 1].item()
            if prob_up < CONFIDENCE_THRESHOLD:
                continue
            picks.append((img_u8, lab, prob_up, int(permno)))
            if len(picks) >= NUM_SAMPLES:
                break
        if len(picks) >= NUM_SAMPLES:
            break

    if not picks:
        sys.stderr.write("[error] no high-confidence positive samples found; lower threshold\n")
        return 4
    print(f"selected {len(picks)} saliency examples")

    fig, axes = plt.subplots(len(picks), 2, figsize=(11.5, 2.9 * len(picks)), dpi=300)
    if len(picks) == 1:
        axes = [axes]
    for i, (img_u8, lab, prob_up, permno) in enumerate(picks):
        img_norm = normalize(img_u8, mu_train, std_train)
        x = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x.requires_grad_()
        logits = model(x)
        logits[0, 1].backward()
        sal = x.grad.detach().abs().squeeze().cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        vibrant = 1.0 - img_u8.astype(np.float32) / 255.0

        ax_left = axes[i][0]
        ax_right = axes[i][1]
        ax_left.imshow(vibrant, cmap="gist_yarg", origin="upper", aspect="equal", interpolation="nearest")
        ax_left.set_title(f"Input image (permno {permno}, P(up)={prob_up:.1%})", fontsize=10)
        ax_left.axis("off")

        ax_right.imshow(vibrant, cmap="gist_yarg", origin="upper", aspect="equal", interpolation="nearest")
        im = ax_right.imshow(sal, cmap="Reds", origin="upper", alpha=0.55, aspect="equal", interpolation="nearest")
        ax_right.set_title(f"Saliency overlay (label={lab})", fontsize=10)
        ax_right.axis("off")
        fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)

    fig.suptitle("CNN input-gradient saliency on I=60 OHLC images", fontsize=12, y=1.01)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
