"""
Backtesting engine for portfolio strategies.

Handles:
- Rebalance schedule (monthly w/o dead months)
- View aggregation (latest view per ticker at rebalance date)
- Portfolio tracking with transaction costs
- Multiple strategy comparison
"""

import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def _stable_seed(rebal_date, placebo_seed: int) -> int:
    """Process-independent seed from a rebalance date (Python's hash() is
    randomized per process via PYTHONHASHSEED, which made the v1 placebo
    non-reproducible run-to-run). Uses a stable digest instead."""
    key = f"{pd.Timestamp(rebal_date).strftime('%Y-%m-%d')}|{placebo_seed}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % (2**32)

from src.optimization.covariance import rolling_covariance
from src.optimization.black_litterman import (
    ViewData,
    compute_prior,
    build_view_matrices,
    black_litterman_posterior,
)
from src.optimization.optimizer import optimize_portfolio
from src.reasoning.calibration import calibrate_uncertainty

logger = logging.getLogger(__name__)

# Dead months — no filings, skip rebalancing
DEAD_MONTHS = {1, 6, 9, 12}
ACTIVE_MONTHS = {2, 3, 4, 5, 7, 8, 10, 11}


def generate_rebalance_dates(
    start_year: int = 2006,
    end_year: int = 2025,
    frequency: str = "monthly_active",
) -> list:
    """
    Generate rebalance dates.

    Args:
        start_year: First year
        end_year: Last year
        frequency: "monthly_active" (skip dead months) or "quarterly"

    Returns:
        List of pd.Timestamp (last business day of month)
    """
    dates = []

    if frequency == "monthly_active":
        for y in range(start_year, end_year + 1):
            for m in sorted(ACTIVE_MONTHS):
                d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                # Adjust to last business day
                if d.weekday() >= 5:  # Saturday or Sunday
                    d = d - pd.offsets.BDay(1)
                dates.append(d)
    elif frequency == "biweekly_active":
        for y in range(start_year, end_year + 1):
            for m in sorted(ACTIVE_MONTHS):
                # Mid-month: 15th (adjusted to business day)
                d_mid = pd.Timestamp(y, m, 15)
                if d_mid.weekday() >= 5:
                    d_mid = d_mid - pd.offsets.BDay(1)
                dates.append(d_mid)
                # End-of-month
                d_end = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                if d_end.weekday() >= 5:
                    d_end = d_end - pd.offsets.BDay(1)
                dates.append(d_end)
    elif frequency == "quarterly":
        for y in range(start_year, end_year + 1):
            for m in [3, 6, 9, 12]:
                d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                if d.weekday() >= 5:
                    d = d - pd.offsets.BDay(1)
                dates.append(d)
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    return sorted(dates)


def get_next_trading_day(date: pd.Timestamp, prices: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Find the next trading day after a given date."""
    future = prices.index[prices.index > date]
    if len(future) == 0:
        return None
    return future[0]


def load_views_summary(views_csv: Path) -> pd.DataFrame:
    """Load and prepare views summary."""
    df = pd.read_csv(views_csv)
    df["view_date"] = pd.to_datetime(df["view_date"])
    df = df.sort_values(["ticker", "view_date"])
    return df


def get_active_views(
    views_df: pd.DataFrame,
    tickers: list,
    rebalance_date: pd.Timestamp,
    max_staleness_days: int = 120,
) -> list:
    """
    Get the latest view per ticker as of rebalance_date.

    Args:
        views_df: Views summary DataFrame
        tickers: List of tickers to get views for
        rebalance_date: Current rebalance date
        max_staleness_days: Max age of view before it's considered stale

    Returns:
        List of ViewData objects
    """
    available = views_df[views_df.view_date <= rebalance_date]
    active_views = []

    for ticker in tickers:
        ticker_views = available[available.ticker == ticker]
        if ticker_views.empty:
            continue

        latest = ticker_views.iloc[-1]
        age_days = (rebalance_date - latest.view_date).days

        if age_days > max_staleness_days:
            continue

        # Calibrate uncertainty
        confidence = latest.confidence
        sentiment = latest.sentiment if pd.notna(latest.sentiment) else None

        # Map confidence string to enum for calibration
        from src.reasoning.schemas import Confidence
        conf_map = {"high": Confidence.HIGH, "medium": Confidence.MEDIUM, "low": Confidence.LOW}
        conf_enum = conf_map.get(confidence, Confidence.MEDIUM)

        sigma = calibrate_uncertainty(
            confidence=conf_enum,
            sentiment=sentiment,
        )

        view = ViewData(
            ticker=ticker,
            view_bps=latest.view_bps,
            confidence=confidence,
            sigma=sigma,
            sentiment=sentiment,
        )
        active_views.append(view)

    return active_views


def run_backtest(
    prices: pd.DataFrame,
    views_df: pd.DataFrame,
    tickers: list,
    rebalance_dates: list,
    strategy: str = "bl_llm",
    delta: float = 2.5,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    max_weight: float = 0.10,
    lookback: int = 252,
    transaction_cost_bps: float = 10.0,
    max_staleness_days: int = 120,
    placebo_seed: int = 0,
) -> dict:
    """
    Run a full backtest.

    Args:
        prices: Daily close prices (index=DatetimeIndex, columns=tickers)
        views_df: Views summary DataFrame
        tickers: List of tickers to trade
        rebalance_dates: List of rebalance dates
        strategy: "bl_llm", "equal_weight", "mvo", "bl_prior", "random_views"
        delta: Risk aversion for BL prior
        tau: BL scaling factor
        risk_aversion: λ for optimization
        max_weight: Max weight per asset
        lookback: Covariance lookback in trading days
        transaction_cost_bps: Round-trip transaction cost
        max_staleness_days: Max view age
        placebo_seed: Seed offset for stochastic strategies (random_views, shuffled_views).
            0 reproduces the original single-realization behavior; vary it for ensembles.

    Returns:
        Dict with daily_returns, weights_history, metrics
    """
    n = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    tc_rate = transaction_cost_bps / 10000.0

    # Compute daily returns for all tickers
    daily_returns = prices[tickers].pct_change().dropna()

    # Track portfolio
    portfolio_returns = []
    portfolio_dates = []
    weights_history = []
    current_weights = np.ones(n) / n  # start equal weight

    rebalance_count = 0

    for i, rebal_date in enumerate(rebalance_dates):
        # Skip if not enough price history
        available_prices = prices[prices.index <= rebal_date]
        if len(available_prices) < lookback + 10:
            logger.debug(f"Skipping {rebal_date}: not enough price history")
            continue

        # Find trade date (next trading day after rebalance)
        trade_date = get_next_trading_day(rebal_date, prices)
        if trade_date is None:
            continue

        # Find next rebalance date (or end of data)
        if i + 1 < len(rebalance_dates):
            next_trade = get_next_trading_day(rebalance_dates[i + 1], prices)
            if next_trade is None:
                next_trade = prices.index[-1]
        else:
            next_trade = prices.index[-1]

        # Compute target weights based on strategy
        try:
            target_weights = _compute_strategy_weights(
                strategy=strategy,
                prices=prices[tickers],
                views_df=views_df,
                tickers=tickers,
                ticker_to_idx=ticker_to_idx,
                rebal_date=rebal_date,
                n=n,
                delta=delta,
                tau=tau,
                risk_aversion=risk_aversion,
                max_weight=max_weight,
                lookback=lookback,
                max_staleness_days=max_staleness_days,
                placebo_seed=placebo_seed,
            )
        except Exception as e:
            # Not enough data — keep current position (no rebalance)
            logger.debug(f"Keeping current weights at {rebal_date}: {e}")
            target_weights = current_weights.copy()

        # Transaction costs
        turnover = np.sum(np.abs(target_weights - current_weights))
        tc = turnover * tc_rate

        # Save weights
        weights_dict = {tickers[j]: target_weights[j] for j in range(n)}
        weights_history.append((rebal_date, weights_dict))

        # Track daily returns from trade_date to next_trade
        period_mask = (daily_returns.index >= trade_date) & (daily_returns.index < next_trade)
        period_returns = daily_returns.loc[period_mask]

        if len(period_returns) == 0:
            continue

        # First day includes transaction cost
        w = target_weights.copy()
        for day_idx, (date, day_ret) in enumerate(period_returns.iterrows()):
            port_ret = w @ day_ret.values

            if day_idx == 0:
                port_ret -= tc  # deduct transaction cost on trade day

            portfolio_returns.append(port_ret)
            portfolio_dates.append(date)

            # Update weights for drift
            w = w * (1 + day_ret.values)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum

        current_weights = w
        rebalance_count += 1

    # Build return series
    ret_series = pd.Series(portfolio_returns, index=portfolio_dates, name=strategy)

    logger.info(
        f"[{strategy}] Backtest complete: {rebalance_count} rebalances, "
        f"{len(ret_series)} days, total return {(1 + ret_series).prod() - 1:.2%}"
    )

    return {
        "daily_returns": ret_series,
        "weights_history": weights_history,
        "rebalance_count": rebalance_count,
    }


def _compute_strategy_weights(
    strategy: str,
    prices: pd.DataFrame,
    views_df: pd.DataFrame,
    tickers: list,
    ticker_to_idx: dict,
    rebal_date: pd.Timestamp,
    n: int,
    delta: float,
    tau: float,
    risk_aversion: float,
    max_weight: float,
    lookback: int,
    max_staleness_days: int,
    placebo_seed: int = 0,
) -> np.ndarray:
    """Compute target weights for a given strategy."""

    if strategy == "equal_weight":
        return np.ones(n) / n

    # Concentrated selection tests (does cross-sectional ranking have value?).
    # top_n_views: hold the N stocks with the highest view_bps, equal-weighted.
    # random_n:    hold N random stocks (the null baseline for selection skill).
    TOP_N = 10
    if strategy in ("top_n_views", "bottom_n_views", "random_n"):
        w = np.zeros(n)
        if strategy in ("top_n_views", "bottom_n_views"):
            views = get_active_views(views_df, tickers, rebal_date, max_staleness_days)
            if not views:
                return np.ones(n) / n
            hi = strategy == "top_n_views"
            ranked = sorted(views, key=lambda v: v.view_bps, reverse=hi)[:TOP_N]
            idx = [ticker_to_idx[v.ticker] for v in ranked]
        else:  # random_n — same count, random tickers
            rng = np.random.default_rng(_stable_seed(rebal_date, placebo_seed))
            idx = list(rng.choice(n, size=min(TOP_N, n), replace=False))
        w[idx] = 1.0 / len(idx)
        return w

    # All other strategies need covariance
    cov = rolling_covariance(prices, rebal_date, lookback=lookback)
    w_prior = np.ones(n) / n
    prior = compute_prior(cov, w_prior, delta=delta)

    if strategy == "bl_prior":
        return optimize_portfolio(prior, cov, risk_aversion, max_weight)

    if strategy == "mvo":
        # Use historical mean returns
        available = prices[prices.index <= rebal_date]
        hist_returns = available.iloc[-(lookback + 1):].pct_change().dropna()
        mu_hist = hist_returns.mean().values * 252  # annualize
        return optimize_portfolio(mu_hist, cov, risk_aversion, max_weight)

    if strategy == "random_views":
        # placebo_seed draws independent realizations for an ensemble.
        # Seeding is now process-independent (see _stable_seed); the v1 code used
        # Python's hash(), which is randomized per process, so v1 placebo numbers
        # were not reproducible. v2 numbers regenerate under the stable seed.
        rng = np.random.default_rng(_stable_seed(rebal_date, placebo_seed))
        random_q = rng.normal(0, 84.5 / 10000, size=n)
        P = np.eye(n)
        omega = np.full(n, 0.03**2)  # medium confidence
        posterior = black_litterman_posterior(cov, prior, P, random_q, omega, tau)
        return optimize_portfolio(posterior, cov, risk_aversion, max_weight)

    if strategy == "shuffled_views":
        # Content-vs-mechanism test (full shuffle): keep the SAME set of view
        # values active on this date, but reassign them to a RANDOM subset of the
        # full universe of the same size. This destroys BOTH which companies get a
        # view AND the value-to-company pairing, while preserving the view-value
        # distribution and the number of active views. If Sharpe collapses toward
        # the placebo, the advantage comes from company-specific content.
        from src.optimization.optimizer import tilt_weights

        views = get_active_views(views_df, tickers, rebal_date, max_staleness_days)
        views = [v for v in views if v.confidence == "medium" and v.view_bps > 0]

        if not views:
            return np.ones(n) / n

        k = len(views)
        rng = np.random.default_rng(_stable_seed(rebal_date, placebo_seed))
        # Pick k distinct tickers from the FULL universe at random, then attach the
        # k view values (view_bps, sigma) to those randomly chosen tickers.
        random_idx = rng.choice(n, size=k, replace=False)
        for v, idx in zip(views, random_idx):
            v.ticker = tickers[idx]

        P, Q, omega = build_view_matrices(views, ticker_to_idx, n)
        posterior = black_litterman_posterior(cov, prior, P, Q, omega, tau)
        return tilt_weights(
            posterior,
            base_weights=np.ones(n) / n,
            tilt_strength=1.0,
            max_weight=max_weight,
            min_weight=0.005,
        )

    if strategy == "bl_llm":
        views = get_active_views(views_df, tickers, rebal_date, max_staleness_days)

        if not views:
            return optimize_portfolio(prior, cov, risk_aversion, max_weight)

        P, Q, omega = build_view_matrices(views, ticker_to_idx, n)
        posterior = black_litterman_posterior(cov, prior, P, Q, omega, tau)
        return optimize_portfolio(posterior, cov, risk_aversion, max_weight)

    if strategy == "bl_llm_tilt":
        from src.optimization.optimizer import tilt_weights

        views = get_active_views(views_df, tickers, rebal_date, max_staleness_days)
        # Filter: medium confidence only (best cross-validated config)
        views = [v for v in views if v.confidence == "medium"]

        if not views:
            return np.ones(n) / n

        P, Q, omega = build_view_matrices(views, ticker_to_idx, n)
        posterior = black_litterman_posterior(cov, prior, P, Q, omega, tau)

        return tilt_weights(
            posterior,
            base_weights=np.ones(n) / n,
            tilt_strength=1.0,
            max_weight=max_weight,
            min_weight=0.005,
        )

    if strategy == "bl_llm_tilt_pos":
        from src.optimization.optimizer import tilt_weights

        views = get_active_views(views_df, tickers, rebal_date, max_staleness_days)
        # Filter: medium confidence + positive views only
        views = [v for v in views if v.confidence == "medium" and v.view_bps > 0]

        if not views:
            return np.ones(n) / n

        P, Q, omega = build_view_matrices(views, ticker_to_idx, n)
        posterior = black_litterman_posterior(cov, prior, P, Q, omega, tau)

        return tilt_weights(
            posterior,
            base_weights=np.ones(n) / n,
            tilt_strength=1.0,
            max_weight=max_weight,
            min_weight=0.005,
        )

    raise ValueError(f"Unknown strategy: {strategy}")
