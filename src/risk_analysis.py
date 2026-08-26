"""
Risk Analysis — Returns, volatility, covariance, and Sharpe ratios.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import RISK_FREE_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE


def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute month-over-month percentage returns."""
    return monthly_prices.pct_change(fill_method=None).dropna()


def compute_covariance_matrix(monthly_returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the sample covariance matrix of monthly returns
    and symmetrise it (for numerical stability with CVXPY).
    """
    cov = monthly_returns.cov().values
    return (cov + cov.T) / 2


def fetch_risk_free_rate(
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
) -> float:
    """
    Fetch the average monthly risk-free rate from the 10-Year
    US Treasury yield (^TNX) over the given period.

    Returns
    -------
    float
        Monthly risk-free rate as a decimal (e.g. 0.0022).
    """
    tnx = yf.download(
        RISK_FREE_TICKER,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    tnx_yield = tnx["Close"].squeeze().dropna() / 100
    annual_rf = tnx_yield.mean()
    monthly_rf = (1 + annual_rf) ** (1 / 12) - 1
    return float(monthly_rf)


def compute_risk_return_table(
    monthly_returns: pd.DataFrame,
    monthly_rf: float | None = None,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Build a summary table of return/risk metrics per ticker.

    Columns: Ticker, Mean Monthly Return, Monthly Volatility,
             Excess Monthly Return, Sharpe Ratio
    """
    if monthly_rf is None:
        monthly_rf = fetch_risk_free_rate(start, end)

    mean_ret = monthly_returns.mean()
    std_ret = monthly_returns.std()

    table = pd.DataFrame({
        "Ticker": mean_ret.index,
        "Mean Monthly Return": mean_ret.values,
        "Monthly Volatility": std_ret.values,
    })
    table["Excess Monthly Return"] = table["Mean Monthly Return"] - monthly_rf
    table["Sharpe Ratio"] = (
        table["Excess Monthly Return"] / table["Monthly Volatility"]
    )
    return table
