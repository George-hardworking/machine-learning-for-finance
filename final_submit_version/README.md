# CNN-Based Price Trend Prediction (Final Submission)

## Project Overview

This directory contains the **standalone, submission-ready reproduction project** for CNN-based price trend prediction from OHLC charts. It replicates and extends the methodology of Jiang, Kelly, and Xiu (2023), replacing hand-crafted technical indicators with horizon-specific convolutional neural networks trained directly on OHLC-volume chart images.

The project is designed for full reproducibility: modular pipeline stages, project-relative paths, GPU-first execution with automatic CPU fallback, and CSV-based output tables for downstream analysis.

## Repository Structure

```text
final_submit_version/
├── notebooks/
│   ├── final_improve.ipynb          # Main notebook (closest to paper logic)
│   └── archive/                     # Legacy notebook versions (kept for history)
├── scripts/
│   ├── run_papermill.sh             # Papermill batch execution script
│   └── pipeline_split/              # Modular stage-based runners
│       ├── config.py                # Global paths, GPU/CPU settings
│       ├── run_core_stage.py        # Core pipeline (main results)
│       ├── run_fig6_stage.py        # Figure 6: Cross-sectional decile analysis
│       ├── run_fig7_stage.py        # Figure 7: Volatility-adjusted returns
│       ├── run_fig8_stage.py        # Figure 8: Saliency maps
│       ├── run_table1_stage.py      # Table I: Performance metrics
│       └── run_all_stages.sh        # Execute all post-analysis stages
├── outputs/
│   ├── cache/                       # Model weights and preprocessed data
│   ├── models/                      # Horizon-specific CNN models
│   └── pipeline_runs/               # Stage outputs
│       ├── logs/                    # Stage execution logs
│       ├── notebooks/               # Parameterized notebook outputs
│       ├── figures/                 # Publication-ready plots
│       └── tables/                  # Performance tables (CSV)
└── docs/
    └── variable_description.txt     # CRSP daily data field descriptions
```

## Quick Start

Run individual stages from this directory:

```bash
# Core pipeline
python scripts/pipeline_split/run_core_stage.py

# Post-analysis stages (requires core outputs)
python scripts/pipeline_split/run_fig6_stage.py    # Decile analysis
python scripts/pipeline_split/run_fig7_stage.py    # Volatility-adjusted returns
python scripts/pipeline_split/run_fig8_stage.py    # Saliency maps
python scripts/pipeline_split/run_table1_stage.py  # Performance table
```

Run all post-analysis stages at once:

```bash
bash scripts/pipeline_split/run_all_stages.sh
```

## Pipeline Configuration

GPU-first design with automatic CPU fallback (see `scripts/pipeline_split/config.py`):

- **GPU**: CUDA with FP16 mixed precision, batch size 1024
- **CPU**: 2 threads, 2 dataloader workers
- **Torch compile**: Disabled for stability
- **Paths**: Project-relative for local and cloud portability

## Data

- **Source**: CRSP daily U.S. equity data (1992–2024)
- **Training**: 1993–2000 (70/30 train-validation split)
- **Testing**: 2001–2019 (out-of-sample)
- **Labels**: Binary (future return > 0) for horizons of 5, 20, and 60 trading days
- **Filters**: Price ≥ $1, positive volume, non-missing observations

## Outputs

| Stage | Output |
|---|---|
| `run_core_stage.py` | Trained CNN models, predictions, core metrics |
| `run_fig6_stage.py` | Decile portfolio returns (cross-sectional analysis) |
| `run_fig7_stage.py` | Volatility-adjusted cumulative long-short returns |
| `run_fig8_stage.py` | Input-gradient saliency maps |
| `run_table1_stage.py` | Portfolio performance metrics (Sharpe, max drawdown, etc.) |

All tables are exported as CSV for use in the paper.

## Notes on Reproducibility

- Random seeds are set across PyTorch, NumPy, and Python random
- Paths are project-relative (no hard-coded absolute paths)
- `outputs/` is intended for local storage and excluded from version control
- Stages are independent: individual figures/tables can be regenerated without re-running the full pipeline
- The notebook `notebooks/final_improve.ipynb` contains heavy outputs disabled by default for portability

## License

This project is for course submission and academic use.

