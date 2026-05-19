# CNN-Based Price Trend Prediction (Presentation Version)

## Project Overview

This directory contains the intermediate (presentation) version of the final project: **CNN-based price trend prediction from OHLC charts**, inspired by Jiang, Kelly, and Xiu (2023). Convolutional neural networks are trained on OHLC-volume chart images to predict future price trends at 5-day, 20-day, and 60-day horizons using CRSP daily U.S. equity data (1992–2024).

This is an exploratory version used for developing the methodology and generating presentation visuals. The final, reproducible version is in [`final_submit_version/`](../final_submit_version/).

## Repository Structure

```text
final_presentation_version/
├── raw/
│   └── OHLC_92_24.csv               # CRSP daily OHLC data (1992–2024)
├── scripts/
│   └── final_improve.ipynb           # Main notebook (CNN training & evaluation)
├── outputs/
│   ├── models/                       # Trained CNN model checkpoints
│   ├── cache/                        # Preprocessed data and feature cache
│   ├── cache_smoke/                  # Lightweight cache for quick tests
│   ├── models_smoke/                 # Lightweight models for quick tests
│   └── ppt_images/                   # Figures and charts for presentations
└── docs/
    ├── read_paper_note.md            # Detailed research notes
    ├── variable_description.txt      # CRSP daily data field descriptions
    └── The Journal of Finance...pdf  # Original Jiang, Kelly, Xiu (2023) paper
```

## Main Analysis File

```text
scripts/final_improve.ipynb
```

The notebook covers:

1. CRSP daily OHLC data loading and filtering (1992–2024)
2. OHLC chart image generation with moving averages and volume bars
3. Three horizon-specific CNN architectures (5-day, 20-day, 60-day)
4. In-sample training (1993–2000) with 70/30 train-validation split
5. Out-of-sample testing (2001–2019)
6. Portfolio construction and performance evaluation
7. Visualization of cumulative returns and model predictions

## CNN Architectures

| Horizon | Input Size | Conv Blocks | Channels |
|---|---|---|---|
| 5-day | 32 × 15 | 2 | 64 → 128 |
| 20-day | 64 × 60 | 3 | 64 → 128 → 256 |
| 60-day | 96 × 180 | 4 | 64 → 128 → 256 → 512 |

All architectures use 5×3 convolutions, 2×1 max-pooling, Leaky ReLU activation, and softmax binary classification (up/down).

## Data

- **Source**: CRSP daily U.S. stock data (1992–2024)
- **Core fields**: PERMNO, close, high, low, open, volume, returns
- **Filters**: Price ≥ $1, positive volume, non-missing observations
- **Labels**: Binary (future return > 0) for horizons of 5, 20, and 60 trading days

## Environment

Python 3.10+ with key packages:

- `pandas`, `numpy`
- `pytorch` (CUDA recommended for GPU training)
- `matplotlib`, `seaborn`
- `PIL` / `pillow` (chart image rendering)
- `scikit-learn`

## Outputs

- `outputs/models/` — Trained CNN checkpoints for each horizon
- `outputs/cache/` — Preprocessed features and dataset splits
- `outputs/ppt_images/` — Presentation-quality figures (cumulative returns, portfolio performance)

## Notes

- This is an exploratory version; paths and configurations may differ from the final submission
- For the fully reproducible, modular pipeline version, see [`final_submit_version/`](../final_submit_version/)
- Chart image rendering uses a custom OHLC renderer with moving average and volume sub-panels

## License

This project is for course submission and academic use.
