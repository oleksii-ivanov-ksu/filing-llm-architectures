"""
Black-Litterman model.

Combines prior equilibrium returns with LLM-generated views
to produce posterior expected returns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ViewData:
    """A single investment view for Black-Litterman."""
    ticker: str
    view_bps: float          # Expected excess return in basis points
    confidence: str          # "high", "medium", "low"
    sigma: float             # Calibrated uncertainty (std dev)
    sentiment: Optional[str] = None
    estimate_dispersion: Optional[float] = None


def compute_prior(
    cov: np.ndarray,
    w_prior: np.ndarray,
    delta: float = 2.5,
) -> np.ndarray:
    """
    Compute implied equilibrium returns (prior).

    π = δ × Σ × w_prior

    Args:
        cov: NxN annualized covariance matrix
        w_prior: Nx1 prior weights (e.g., 1/N)
        delta: Risk aversion coefficient

    Returns:
        Nx1 implied equilibrium returns
    """
    return delta * cov @ w_prior


def build_view_matrices(
    views: List[ViewData],
    ticker_to_idx: dict,
    n_assets: int,
) -> tuple:
    """
    Build P, Q, Omega matrices from views.

    Args:
        views: List of ViewData objects
        ticker_to_idx: Mapping ticker -> column index
        n_assets: Total number of assets

    Returns:
        (P, Q, omega) where:
            P: KxN pick matrix
            Q: Kx1 view returns
            omega: Kx1 view uncertainties (variances)
    """
    k = len(views)
    P = np.zeros((k, n_assets))
    Q = np.zeros(k)
    omega = np.zeros(k)

    for i, view in enumerate(views):
        idx = ticker_to_idx[view.ticker]
        P[i, idx] = 1.0
        Q[i] = view.view_bps / 10000.0   # bps → decimal
        omega[i] = view.sigma ** 2        # variance

    return P, Q, omega


def black_litterman_posterior(
    cov: np.ndarray,
    prior: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Compute Black-Litterman posterior expected returns.

    E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]

    Args:
        cov: NxN annualized covariance matrix
        prior: Nx1 implied equilibrium returns (π)
        P: KxN pick matrix
        Q: Kx1 view returns
        omega: Kx1 view uncertainties (variances, diagonal of Ω)
        tau: Scaling factor for prior uncertainty

    Returns:
        Nx1 posterior expected returns
    """
    n = len(prior)
    k = len(Q)

    if k == 0:
        return prior.copy()

    # τΣ inverse
    tau_cov_inv = np.linalg.inv(tau * cov)

    # Ω inverse (diagonal)
    Omega_inv = np.diag(1.0 / omega)

    # Posterior precision = (τΣ)⁻¹ + P'Ω⁻¹P
    posterior_precision = tau_cov_inv + P.T @ Omega_inv @ P

    # Posterior covariance
    posterior_cov = np.linalg.inv(posterior_precision)

    # Posterior mean
    posterior_mean = posterior_cov @ (tau_cov_inv @ prior + P.T @ Omega_inv @ Q)

    return posterior_mean


def compute_bl_weights(
    prices: "pd.DataFrame",
    views: List[ViewData],
    tickers: list,
    as_of_date: "pd.Timestamp",
    delta: float = 2.5,
    tau: float = 0.05,
    lookback: int = 252,
) -> np.ndarray:
    """
    Full BL pipeline: prices + views → posterior returns.

    Args:
        prices: Daily close prices (index=date, columns=tickers)
        views: List of ViewData for active views
        tickers: List of all tickers (column order)
        as_of_date: Rebalance date
        delta: Risk aversion
        tau: Prior uncertainty scaling
        lookback: Covariance lookback window

    Returns:
        Nx1 posterior expected returns
    """
    from .covariance import rolling_covariance

    n = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # 1. Covariance
    cov = rolling_covariance(prices[tickers], as_of_date, lookback=lookback)

    # 2. Prior (equal weights)
    w_prior = np.ones(n) / n
    prior = compute_prior(cov, w_prior, delta=delta)

    # 3. Filter views to valid tickers
    valid_views = [v for v in views if v.ticker in ticker_to_idx]

    if not valid_views:
        return prior

    # 4. Build view matrices
    P, Q, omega = build_view_matrices(valid_views, ticker_to_idx, n)

    # 5. Posterior
    posterior = black_litterman_posterior(cov, prior, P, Q, omega, tau=tau)

    return posterior
