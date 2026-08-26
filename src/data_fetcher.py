"""
Data Fetcher — Retrieves historical stock prices via yfinance.
"""

import pandas as pd
import yfinance as yf

from src.config import ALL_TICKERS, DEFAULT_START_DATE, DEFAULT_END_DATE


def fetch_stock_data(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Download adjusted-close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str] | None
        Ticker symbols to download.  Defaults to ALL_TICKERS.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Daily adjusted-close prices indexed by date, one column per ticker.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    close_data = data["Adj Close"]
    # Flatten multi-level columns if present
    if hasattr(close_data.columns, "get_level_values"):
        close_data.columns = close_data.columns.get_level_values(0)

    return close_data


def get_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end."""
    return daily_prices.resample("ME").last()
