import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

class EDA:


    #  Data Extraction

    @staticmethod
    def fetch_data(tickers, start, end):
        """
        Fetch historical daily data from Yahoo Finance and return a dict of DataFrames keyed by ticker.
        """
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False
        )
        return raw

    @staticmethod
    def split_tickers(raw_data, tickers):
        """
        Split multi-index DataFrame from yfinance into a dict of DataFrames keyed by ticker.
        """
        data_dict = {t: raw_data[t].copy().reset_index() for t in tickers}
        return data_dict
        
    # Returns Calculation

    @staticmethod
    def calculate_daily_returns(df, price_col="Adj Close"):
        df = df.copy()
        df["Daily Return"] = df[price_col].pct_change()
        df.dropna(inplace=True)
        return df


    # Outlier Detection

    @staticmethod
    def detect_outliers(df, return_col="Daily Return", threshold=3):
        std = df[return_col].std()
        return df[np.abs(df[return_col]) > threshold * std]

    @staticmethod
    def split_outliers(df, return_col="Daily Return", threshold=3):
        std = df[return_col].std()
        high = df[df[return_col] > threshold * std]
        low  = df[df[return_col] < -threshold * std]
        return high, low


    # Augmented Dickey-Fuller Test

    @staticmethod
    def adf_test(df, column="Adj Close", asset_name="Asset"):
        result = adfuller(df[column].dropna())
        print(f"\nADF Test for {asset_name} ({column}):")
        print(f"  Test Statistic: {result[0]:.4f}")
        print(f"  p-value: {result[1]:.4f}")
        print(f"  Critical Values: {result[4]}")
        if result[1] < 0.05:
            print("  => Reject H0: Series is stationary")
        else:
            print("  => Fail to reject H0: Series is non-stationary")
        return result


    # Risk Metrics

    @staticmethod
    def risk_metrics(df, return_col="Daily Return", risk_free_rate=0.01):
        var_95 = np.percentile(df[return_col].dropna(), 5)
        mean_daily = df[return_col].mean()
        std_daily  = df[return_col].std()
        sharpe_ratio = (mean_daily*252 - risk_free_rate) / (std_daily * np.sqrt(252))
        return var_95, sharpe_ratio


    # Extreme Returns Summary

    @staticmethod
    def extreme_returns_summary(data_dict, threshold=3):
        summary = []
        for asset, df in data_dict.items():
            high, low = EDA.split_outliers(df, threshold=threshold)
            summary.append({
                "Asset": asset,
                "High-Return Days": high.shape[0],
                "Low-Return Days": low.shape[0]
            })
        return pd.DataFrame(summary)
