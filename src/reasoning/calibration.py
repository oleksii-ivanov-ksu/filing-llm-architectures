"""
Confidence to Uncertainty mapping for Black-Litterman.

Converts LLM confidence levels to standard deviation (sigma)
used in the Black-Litterman omega matrix.
"""

from typing import Optional
from .schemas import Confidence


# Base sigma values for each confidence level
CONFIDENCE_SIGMA_MAP = {
    Confidence.HIGH: 0.01,    # 1% std - highly reliable view
    Confidence.MEDIUM: 0.03,  # 3% std - moderate uncertainty
    Confidence.LOW: 0.05,     # 5% std - high uncertainty
}


def confidence_to_sigma(confidence: Confidence) -> float:
    """
    Convert confidence level to base standard deviation.

    Args:
        confidence: Confidence level (high, medium, low)

    Returns:
        Base sigma (standard deviation)
    """
    return CONFIDENCE_SIGMA_MAP[confidence]


def confidence_to_omega(confidence: Confidence) -> float:
    """
    Convert confidence level to omega (variance) for Black-Litterman.

    Omega = sigma^2

    Args:
        confidence: Confidence level

    Returns:
        Omega (variance)
    """
    sigma = confidence_to_sigma(confidence)
    return sigma ** 2


def calibrate_uncertainty(
    confidence: Confidence,
    analyst_dispersion: Optional[float] = None,
    sentiment: Optional[str] = None,
) -> float:
    """
    Calibrate uncertainty based on confidence and additional signals.

    Adjustments:
    - High analyst dispersion (>30%) increases uncertainty by 50%
    - Extreme sentiment (confident/concerned) decreases uncertainty by 20%

    Args:
        confidence: Base confidence level
        analyst_dispersion: (epsHigh - epsLow) / |epsAvg| from estimates
        sentiment: Sentiment from filing extraction

    Returns:
        Calibrated sigma (standard deviation)
    """
    base_sigma = confidence_to_sigma(confidence)

    # Increase uncertainty if analysts disagree
    if analyst_dispersion is not None and analyst_dispersion > 0.3:
        base_sigma *= 1.5

    # Decrease uncertainty if sentiment is extreme (more conviction)
    extreme_sentiments = {"confident", "concerned"}
    if sentiment and sentiment.lower() in extreme_sentiments:
        base_sigma *= 0.8

    return base_sigma


def sigma_to_confidence(sigma: float) -> Confidence:
    """
    Reverse mapping: sigma to confidence level.

    Useful for interpreting calibrated values.

    Args:
        sigma: Standard deviation

    Returns:
        Closest confidence level
    """
    if sigma <= 0.015:
        return Confidence.HIGH
    elif sigma <= 0.04:
        return Confidence.MEDIUM
    else:
        return Confidence.LOW
