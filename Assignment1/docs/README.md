Dear all,

In this assignment, you will apply linear regression methods (OLS, LASSO, Ridge) to predict U.S. stock returns using firm characteristics from the GKX dataset.

Key Requirements:

1. Basic Requirements:
   - Run linear models using all firm characteristics to predict stock returns
   - Use recursive time-ordered cross-validation
   - Evaluate model performance using MSE and R²
   - Drop the following columns before modeling: cols_del = ['SHROUT', 'mve0', 'prc', 'permno', 'DATE', 'sic2']

2. Bonus Tasks:
   - Evaluate Economic Value: Analyze the economic value of your models through long-short spread portfolios
   - Advanced Feature Selection: Incorporate macroeconomic indicators and use Group LASSO to identify key characteristics

Data:
The dataset (GKX_20201231.csv) is available on the Teaching - Dropbox folder. Please do NOT distribute this data without my authorization.

⚠️ IMPORTANT — Before You Start:
Please make sure you carefully read the following materials BEFORE working with the data:
1. readme2021.txt — This file explains the dataset structure, variable definitions, and important notes about the data.
2. My lecture slides (Lecture 2) — which contain detailed instructions and clarifications specific to this homework assignment.


Submitting format:

A comprehensive Jupyter Notebook (.ipynb) containing executable code, visualizations, and detailed explanations for the project.



---

## My own Workflow

1) Data Loading & Initial Checks
- Load GKX dataset, inspect shape, date range, dtypes, missingness; plot basic distributions/anomalies for returns/features.

2) Cleaning & Feature Prep
- Drop cols_del = ['SHROUT', 'mve0', 'prc', 'permno', 'DATE', 'sic2']; handle missing values (train-only fit for imputers), standardize features using training fit only; ensure time order by DATE (even if dropped post-ordering).

3) Time-Ordered Cross-Validation
- Use recursive/rolling or expanding windows; per fold: split train/validation chronologically, fit OLS/Ridge/LASSO, collect MSE and R²; aggregate metrics across folds.

4) Model Selection & Testing
- Tune regularization strength for Ridge/LASSO via CV grid; keep OLS as baseline. Optionally reserve final tail window as test set for one-shot evaluation using chosen hyperparameters.

5) Economic Value (Bonus)
- From model forecasts, build long-short portfolios (e.g., top vs. bottom decile each period, equal- or value-weight); compute periodic and cumulative returns, Sharpe/IR; compare models.

6) Macro + Group LASSO (Bonus)
- Merge lagged macro indicators aligned to stock dates to avoid look-ahead; apply Group LASSO to select characteristic groups; compare sparsity and performance.

7) Reporting & Reproducibility
- Present tables/plots for metrics and portfolio performance; summarize key drivers (coefficients/feature importance); document window setup, hyperparameters, and data filters to ensure reproducibility.