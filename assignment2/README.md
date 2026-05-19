# Assignment 2: Deep Learning for Limit Order Books

## Project Overview

This assignment replicates the **DeepLOB** architecture (Zhang, Zohren, and Roberts, 2019) for predicting mid-price movements in high-frequency limit order book (LOB) data. A convolutional neural network processes the top 10 bid-ask levels of the FI-2010 benchmark dataset to classify future price directions (up / flat / down) at multiple prediction horizons.

## Repository Structure

```text
assignment2/
├── raw/
│   ├── Train_Dst_NoAuction_ZScore_CF_1~9.txt   # Training LOB data (9 files)
│   └── Test_Dst_NoAuction_ZScore_CF_1~9.txt    # Testing LOB data (9 files)
├── scripts/
│   ├── assignment2_kaibiao.ipynb       # Main submission notebook
│   ├── run_train_pytorch.ipynb         # Training script
│   ├── best_val_model_assignment2.pt   # Best validation model weights
│   └── best_val_model_assignment2_full.pt  # Full model checkpoint
└── docs/
    ├── README.md                       # Assignment instructions
    └── DeepLOB_Deep Convolutional...pdf # Original DeepLOB paper
```

## Main Analysis File

```text
scripts/assignment2_kaibiao.ipynb
```

The notebook covers:

1. LOB data loading and preprocessing (z-score normalization)
2. DeepLOB CNN architecture implementation in PyTorch
3. Model training with multiple configurations
4. Mid-price movement classification (3-class: up / flat / down)
5. Evaluation: classification accuracy, confusion matrices
6. Bonus: Multi-horizon prediction (k = 1, 2, 3, 5, 10 events)
7. Bonus: Asset-specific vs. universal model comparison

## Data

- **Source**: FI-2010 benchmark dataset (5 NASDAQ Nordic stocks)
- **Features**: Top 10 bid-ask levels (price + volume), rows 1–40
- **Labels**: Relative mid-price changes at horizons k = 1, 2, 3, 5, 10
- **Threshold**: ±0.2% for up/flat/down classification
- **Training**: First 7 days (Train_CF_7)
- **Testing**: Last 3 days (Test_CF_7, 8, 9)

## Environment

Python 3.10+ with key packages:

- `pandas`, `numpy`
- `pytorch` (with CUDA recommended)
- `matplotlib`, `seaborn`
- `scikit-learn`

## Outputs

Running the notebook produces:

- `best_val_model_assignment2.pt` — Best model weights by validation loss
- `best_val_model_assignment2_full.pt` — Full model with optimizer state
- Inline classification metrics and confusion matrices
- Horizon-specific predictability analysis

## Notes on Reproducibility

- Random seeds are set for PyTorch, NumPy, and Python random
- GPU training is recommended but CPU fallback is supported
- Data is pre-normalized using z-scores as per the FI-2010 benchmark
- The notebook should be run from a clean kernel

## License

This assignment is for course submission and academic use.
