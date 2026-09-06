"""
Portfolio optimization: MV optimization and weight tilting.
"""

import numpy as np
from scipy.optimize import minimize


def tilt_weights(
    expected_returns: np.ndarray,
    base_weights: np.ndarray = None,
    tilt_strength: float = 1.0,
    max_weight: float = 0.10,
    min_weight: float = 0.005,
) -> np.ndarray:
    """
    Tilt base weights proportionally to expected returns.

    w_i = base_i × (1 + tilt_strength × z_i)
    where z_i = (μ_i - mean(μ)) / std(μ)

    Preserves diversification while incorporating views.

    Args:
        expected_returns: Nx1 expected returns from BL posterior
        base_weights: Starting weights (default: 1/N)
        tilt_strength: How aggressively to tilt (0 = no tilt, 1 = moderate)
        max_weight: Maximum weight cap
        min_weight: Minimum weight floor

    Returns:
        Nx1 tilted weights (sum = 1)
    """
    n = len(expected_returns)

    if base_weights is None:
        base_weights = np.ones(n) / n

    # Z-score of expected returns
    mu_mean = expected_returns.mean()
    mu_std = expected_returns.std()

    if mu_std < 1e-10:
        return base_weights.copy()

    z_scores = (expected_returns - mu_mean) / mu_std

    # Tilt
    tilted = base_weights * (1.0 + tilt_strength * z_scores)

    # Floor negatives
    tilted = np.maximum(tilted, min_weight)

    # Cap
    tilted = np.minimum(tilted, max_weight)

    # Normalize to sum = 1
    tilted /= tilted.sum()

    return tilted


def optimize_portfolio(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 2.5,
    max_weight: float = 0.10,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Mean-variance optimization with constraints.

    max  w'μ - λ/2 × w'Σw
    s.t. w ≥ min_weight, w ≤ max_weight, Σw = 1

    Args:
        expected_returns: Nx1 expected returns
        cov: NxN covariance matrix
        risk_aversion: λ parameter
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset (0 = long-only)

    Returns:
        Nx1 optimal weights
    """
    n = len(expected_returns)

    def neg_utility(w):
        ret = w @ expected_returns
        risk = w @ cov @ w
        return -(ret - risk_aversion / 2 * risk)

    def neg_utility_jac(w):
        return -(expected_returns - risk_aversion * cov @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_utility,
        w0,
        jac=neg_utility_jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        # Fallback to equal weights
        return np.ones(n) / n

    weights = result.x
    # Clean up tiny negatives from numerical noise
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()

    return weights
