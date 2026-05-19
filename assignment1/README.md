# Assignment 1: Linear Regression for Stock Return Prediction

## Project Overview

This assignment applies linear regression methods — **OLS**, **Ridge**, and **LASSO** — to predict U.S. stock returns using the GKX dataset of 94 lagged firm characteristics. The analysis follows a rigorous time-ordered cross-validation framework and evaluates models on both statistical metrics (MSE, R²) and economic value (long-short portfolio performance).

## Repository Structure

```text
assignment1/
├── raw/
│   └── gkx_20201231.csv            # GKX dataset (94 firm characteristics, monthly)
├── scripts/
│   └── assignment1_kaibiao.ipynb    # Main submission notebook
├── outputs/
│   ├── prediction_full_lasso.csv    # LASSO model out-of-sample predictions
│   └── prediction_full_ridge.csv    # Ridge model out-of-sample predictions
└── docs/
    ├── README.md                   # Assignment instructions
    └── data_description.md         # GKX variable documentation
```

## Main Analysis File

```text
scripts/assignment1_kaibiao.ipynb
```

The notebook covers:

1. Data loading and cleaning
2. Feature preprocessing (standardization, missing value handling)
3. Time-ordered cross-validation with recursive expanding windows
4. OLS baseline regression
5. Ridge regression with hyperparameter tuning
6. LASSO regression with hyperparameter tuning
7. Model evaluation: MSE, R², and out-of-sample prediction plots
8. Bonus: Economic value evaluation via long-short decile portfolios
9. Bonus: Macroeconomic indicators and Group LASSO feature selection

## Data

- **Source**: GKX (Gu, Kelly, and Xiu) dataset — 94 monthly lagged firm characteristics
- **Frequency**: Monthly
- **Target**: Stock returns
- **Preprocessing**: Drops identifier and market-cap columns (SHROUT, mve0, prc, permno, DATE, sic2)

## Environment

Python 3.10+ with key packages:

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `statsmodels`

## Outputs

Running the notebook produces:

- `outputs/prediction_full_lasso.csv` — LASSO model predictions
- `outputs/prediction_full_ridge.csv` — Ridge model predictions
- Inline performance metrics (MSE, R², Sharpe ratio, Information ratio)
- Portfolio cumulative return plots

## Notes on Reproducibility

- Time-ordered cross-validation ensures no forward-looking bias
- Random seeds are set for reproducibility
- The notebook should be run from a clean kernel

## License

This assignment is for course submission and academic use.
