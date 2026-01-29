# Notebooks Overview

This folder contains Jupyter notebooks for the GMF Investments portfolio optimization and forecasting project.

## Notebooks
- `eda_preprocessing.ipynb`: Exploratory data analysis and preprocessing of YFinance data, including data cleaning, missing value handling, normalization, calculation of daily returns, rolling volatility, and stationarity tests.
- `tsla_forecasting.ipynb`: Time series forecasting of Tesla (TSLA) stock prices using ARIMA/SARIMA and LSTM models, with model evaluation metrics (MAE, RMSE, MAPE) and comparison of performance.
- `future_forecast.ipynb`: Generation of future Tesla stock price forecasts based on the best-performing model, including visualization with confidence intervals, trend analysis, and identification of market opportunities and risks.
- `portfolio_optimization.ipynb`: Construction of an optimal portfolio using Modern Portfolio Theory (MPT) informed by forecasted TSLA returns and historical SPY and BND data, including Efficient Frontier visualization, covariance matrix heatmap, and portfolio recommendation.
- `strategy_backtesting.ipynb`: Backtesting of the optimized portfolio strategy against a benchmark (60% SPY / 40% BND), including cumulative returns comparison, performance metrics (total return, annualized return, Sharpe ratio, maximum drawdown), and conclusions on strategy viability.
