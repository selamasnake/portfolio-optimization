# Portfolio Optimization 
## Project Overview

This project demonstrates a data-driven approach to portfolio management by integrating **time series forecasting**, **Modern Portfolio Theory (MPT)**, and **backtesting**. The goal is to guide investment decisions by predicting market trends, optimizing asset allocation, and evaluating portfolio performance against benchmarks.

The project focuses on three key financial assets:

| Asset | Ticker | Description | Risk Profile |
|-------|--------|------------|-------------|
| Tesla | TSLA   | High-growth stock in the consumer discretionary sector (Automobile Manufacturing) | High risk, high potential return |
| Vanguard Total Bond Market ETF | BND | Tracks U.S. investment-grade bonds | Low risk, stability and income |
| S&P 500 ETF | SPY | Tracks the S&P 500 Index | Moderate risk, broad market exposure |

---

## Key Features

1. **Time Series Forecasting**
   - Build ARIMA/SARIMA and LSTM models to predict Tesla stock prices.
   - Evaluate model performance using MAE, RMSE, and MAPE.
   - Generate future price forecasts with confidence intervals.

2. **Portfolio Optimization**
   - Construct portfolios using MPT to maximize risk-adjusted returns.
   - Calculate expected returns, covariance matrices, and Efficient Frontier.
   - Identify Maximum Sharpe Ratio and Minimum Volatility portfolios.

3. **Strategy Backtesting**
   - Simulate portfolio performance over a historical period.
   - Compare optimized portfolios against benchmarks (e.g., 60/40 SPY/BND).
   - Compute key performance metrics: total return, annualized return, volatility, Sharpe ratio, maximum drawdown.
   - Visualize cumulative returns and portfolio efficiency.

4. **Modular Design**
   - Source modules for EDA, forecasting, optimization, and backtesting.
   - Jupyter notebooks for step-by-step analysis and reproducibility.
   - Scripts and utilities to facilitate pipeline automation and experimentation.


---

## Notebooks Overview

- `eda_preprocessing.ipynb` – Data cleaning, exploratory analysis, and risk metric calculations.
- `tsla_forecasting.ipynb` – Time series forecasting of Tesla stock using ARIMA/SARIMA and LSTM.
- `future_forecast.ipynb` – Generate future price forecasts and confidence intervals.
- `portfolio_optimization.ipynb` – Construct optimal portfolios using Modern Portfolio Theory.
- `strategy_backtesting.ipynb` – Simulate portfolio performance and benchmark comparison.

---

## Source Modules Overview

- `eda.py` – Functions for downloading, cleaning, and performing exploratory analysis on YFinance datasets.
- `forecast.py` – Implements ARIMA/SARIMA and LSTM forecasting models with evaluation and prediction utilities.
- `backtesting.py` – Simulate portfolio performance, compute metrics, and compare portfolios against benchmarks.

### Usage 
1. Clone the repository:
```bash
git clone https://github.com/selamasnake/portfolio-optimization.git
cd portfolio-optimization
```
2. Create a virtual environment and activate it
3. Install dependencies:
```bash 
pip install -r requirements.txt
```

Run Jupyter notebooks sequentially to reproduce the full analysis:

1. Data preprocessing and EDA
2. Forecasting and model evaluation
3. Portfolio optimization
4. Strategy backtesting
