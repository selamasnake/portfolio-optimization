import pandas as pd
import numpy as np

class Backtesting:

    # def performance_metrics(daily_returns: pd.Series):
    #     """Compute key performance metrics for a single portfolio."""
    #     total_return = daily_returns.add(1).prod() - 1
    #     ann_return = daily_returns.mean() * 252
    #     ann_vol = daily_returns.std() * np.sqrt(252)
    #     sharpe_ratio = ann_return / ann_vol
    #     cum_returns = (1 + daily_returns).cumprod()
    #     max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    #     return total_return, ann_return, ann_vol, sharpe_ratio, max_drawdown

    # def compare_portfolios(portfolios: dict):
    #     """
    #     Compute performance metrics for multiple portfolios.
        
    #     Parameters:
    #         portfolios (dict): keys are portfolio names, values are pd.Series of daily returns.
        
    #     Returns:
    #         pd.DataFrame: metrics for all portfolios.
    #     """
    #     metrics_list = []
    #     names = []

    #     for name, returns in portfolios.items():
    #         metrics_list.append(performance_metrics(returns))
    #         names.append(name)

    #     metrics_df = pd.DataFrame(
    #         metrics_list,
    #         index=names,
    #         columns=["Total Return", "Annualized Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"]
    #     )
    #     return metrics_df
    
    @staticmethod
    def performance_metrics(daily_returns: pd.Series):
        """Compute key performance metrics for a single portfolio."""
        total_return = daily_returns.add(1).prod() - 1
        ann_return = daily_returns.mean() * 252
        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = ann_return / ann_vol
        cum_returns = (1 + daily_returns).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        return total_return, ann_return, ann_vol, sharpe_ratio, max_drawdown

    @staticmethod
    def compare_portfolios(portfolios: dict):
        """
        Compute performance metrics for multiple portfolios.
        """
        metrics_list = []
        names = []

        for name, returns in portfolios.items():
            metrics_list.append(Backtesting.performance_metrics(returns))
            names.append(name)

        metrics_df = pd.DataFrame(
            metrics_list,
            index=names,
            columns=["Total Return", "Annualized Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"]
        )
        return metrics_df