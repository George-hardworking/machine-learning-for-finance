# Machine Learning for Finance

## Project Overview

This repository contains coursework and final project materials for **Machine Learning for Finance**, covering classical linear models, deep learning for limit order books, and CNN-based price trend prediction with full reproducibility.

The repository is organized into five subdirectories:

| Directory | Topic | Description |
|---|---|---|
| `assignment1/` | Linear Regression | Stock return prediction using OLS, Ridge, and LASSO on the GKX dataset |
| `assignment2/` | Deep Learning for LOB | DeepLOB replication: CNN-based mid-price movement classification on FI-2010 limit order book data |
| `final_presentation_version/` | CNN Price Trends (Intermediate) | CNN-based price trend prediction from OHLC charts — exploratory and presentation version |
| `final_submit_version/` | CNN Price Trends (Final) | Standalone, reproducible final project with modular pipeline for NeurIPS 2026 submission |
| `paper_writing_latex/` | Academic Paper | NeurIPS 2026 LaTeX paper: reproduction study of Jiang, Kelly, and Xiu (2023) |

## Repository Structure

```text
.
├── assignment1/                      # Assignment 1: Linear Regression
│   ├── raw/gkx_20201231.csv          # GKX dataset (94 firm characteristics)
│   ├── scripts/assignment1_kaibiao.ipynb
│   ├── outputs/                      # Prediction CSVs
│   └── docs/                         # Assignment instructions & data description
├── assignment2/                      # Assignment 2: DeepLOB
│   ├── raw/                          # FI-2010 LOB data (Train/Test splits)
│   ├── scripts/
│   │   ├── assignment2_kaibiao.ipynb
│   │   └── run_train_pytorch.ipynb
│   ├── docs/                         # Assignment instructions & DeepLOB paper
│   └── *.pt                          # Trained model weights
├── final_presentation_version/       # Final project (presentation version)
│   ├── raw/OHLC_92_24.csv            # CRSP daily OHLC (1992–2024)
│   ├── scripts/final_improve.ipynb
│   ├── outputs/models/               # CNN model checkpoints
│   ├── outputs/cache/                # Preprocessed data cache
│   ├── outputs/ppt_images/           # Presentation figures
│   └── docs/                         # Research notes & variable descriptions
├── final_submit_version/             # Final project (submission-ready, reproducible)
│   ├── notebooks/final_improve.ipynb
│   ├── scripts/pipeline_split/       # Modular stage runners (core, fig6–8, table1)
│   ├── scripts/run_papermill.sh
│   ├── outputs/                      # Pipeline runs, models, cache
│   └── docs/                         # Variable descriptions
├── paper_writing_latex/              # NeurIPS 2026 LaTeX paper
│   ├── neurips_2026.tex             # Main paper
│   ├── neurips_2026.pdf             # Compiled PDF
│   ├── references.bib               # Bibliography
│   ├── figures/                     # Paper figures
│   └── scripts/stage_figures.sh     # Figure staging script
└── README.md
```

## Environment

Python 3.10+ is recommended. Key packages include:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `pytorch` (with CUDA support for GPU training)
- `jupyter`

Individual subdirectories may have additional dependencies; refer to each subdirectory's README for details.

## License

This repository is for course project submission and academic use.

