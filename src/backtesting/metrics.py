"""
Portfolio performance metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats


def annualized_return(returns: pd.Series) -> float:
    """Annualized return from daily returns."""
    total = (1 + returns).prod()
    n_years = len(returns) / 252
    if n_years <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility from daily returns."""
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.02) -> float:
    """
    Annualized Sharpe ratio.

    Args:
        returns: Daily returns
        rf_annual: Annual risk-free rate (default 2%)
    """
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - rf_annual) / ann_vol


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.02) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    ann_ret = annualized_return(returns)
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    downside = returns[returns < rf_daily] - rf_daily
    if len(downside) == 0:
        return 0.0
    downside_vol = downside.std() * np.sqrt(252)
    if downside_vol == 0:
        return 0.0
    return (ann_ret - rf_annual) / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (as negative fraction)."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def calmar_ratio(returns: pd.Series, rf_annual: float = 0.02) -> float:
    """Calmar ratio = annualized return / |max drawdown|."""
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return (ann_ret - rf_annual) / abs(mdd)


def total_return(returns: pd.Series) -> float:
    """Total cumulative return."""
    return (1 + returns).prod() - 1


def compute_turnover(weights_history: list) -> pd.Series:
    """
    Compute quarterly turnover from weights history.

    Args:
        weights_history: List of (date, weights_dict) tuples

    Returns:
        Series of turnover values per rebalance
    """
    turnovers = []
    dates = []

    for i in range(1, len(weights_history)):
        date, w_new = weights_history[i]
        _, w_old = weights_history[i - 1]

        # Align tickers
        all_tickers = set(w_new.keys()) | set(w_old.keys())
        turnover = sum(
            abs(w_new.get(t, 0) - w_old.get(t, 0)) for t in all_tickers
        )
        turnovers.append(turnover)
        dates.append(date)

    return pd.Series(turnovers, index=dates, name="turnover")


def compute_all_metrics(returns: pd.Series, rf_annual: float = 0.02) -> dict:
    """Compute all metrics for a return series."""
    return {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, rf_annual),
        "sortino_ratio": sortino_ratio(returns, rf_annual),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, rf_annual),
        "total_return": total_return(returns),
        "n_days": len(returns),
    }


def compare_strategies(
    strategy_returns: dict,
    rf_annual: float = 0.02,
) -> pd.DataFrame:
    """
    Compare multiple strategies.

    Args:
        strategy_returns: Dict of {strategy_name: daily_returns_series}
        rf_annual: Annual risk-free rate

    Returns:
        DataFrame with metrics per strategy
    """
    rows = {}
    for name, rets in strategy_returns.items():
        rows[name] = compute_all_metrics(rets, rf_annual)

    df = pd.DataFrame(rows).T
    df.index.name = "strategy"
    return df


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    rf_annual: float = 0.02,
) -> tuple:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Returns:
        (lower, upper) bounds
    """
    rng = np.random.default_rng(42)
    n = len(returns)
    sharpes = []

    for _ in range(n_bootstrap):
        sample = returns.iloc[rng.integers(0, n, size=n)]
        sharpes.append(sharpe_ratio(sample, rf_annual))

    alpha = (1 - confidence) / 2
    lower = np.percentile(sharpes, alpha * 100)
    upper = np.percentile(sharpes, (1 - alpha) * 100)
    return lower, upper


def test_sharpe_difference(
    returns_a: pd.Series,
    returns_b: pd.Series,
    rf_annual: float = 0.02,
) -> dict:
    """
    Test if Sharpe ratio of A is significantly different from B.

    Uses paired t-test on excess returns.
    """
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    # Align dates
    common = returns_a.index.intersection(returns_b.index)
    excess_a = returns_a.loc[common] - rf_daily
    excess_b = returns_b.loc[common] - rf_daily
    diff = excess_a - excess_b

    t_stat, p_value = stats.ttest_1samp(diff, 0)

    return {
        "sharpe_a": sharpe_ratio(returns_a, rf_annual),
        "sharpe_b": sharpe_ratio(returns_b, rf_annual),
        "sharpe_diff": sharpe_ratio(returns_a, rf_annual) - sharpe_ratio(returns_b, rf_annual),
        "t_stat": t_stat,
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
    }
