"""
Covariance matrix estimation with Ledoit-Wolf shrinkage.

Rolling window estimation for backtesting.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    annualize: bool = True,
) -> np.ndarray:
    """
    Estimate covariance matrix from returns.

    Args:
        returns: DataFrame of daily returns (T x N), columns = tickers
        method: "ledoit_wolf" or "sample"
        annualize: multiply by 252 for annualized covariance

    Returns:
        NxN covariance matrix
    """
    returns_clean = returns.dropna()

    if len(returns_clean) < 60:
        raise ValueError(f"Not enough observations: {len(returns_clean)} (need ≥60)")

    if method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns_clean.values)
        cov = lw.covariance_
    elif method == "sample":
        cov = returns_clean.cov().values
    else:
        raise ValueError(f"Unknown method: {method}")

    if annualize:
        cov = cov * 252

    return cov


def rolling_covariance(
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback: int = 252,
    method: str = "ledoit_wolf",
) -> np.ndarray:
    """
    Estimate covariance using rolling window ending at as_of_date.

    Args:
        prices: DataFrame of daily close prices (columns = tickers)
        as_of_date: End date for the lookback window
        lookback: Number of trading days to use
        method: Estimation method

    Returns:
        Annualized NxN covariance matrix
    """
    # Filter to data available as of date
    available = prices[prices.index <= as_of_date]

    if len(available) < lookback + 1:
        raise ValueError(
            f"Not enough price data before {as_of_date}: "
            f"{len(available)} days (need {lookback + 1})"
        )

    # Take last lookback+1 days (to get lookback returns)
    window = available.iloc[-(lookback + 1):]
    returns = window.pct_change().dropna()

    return estimate_covariance(returns, method=method, annualize=True)
