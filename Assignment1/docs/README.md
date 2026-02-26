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

Best regards