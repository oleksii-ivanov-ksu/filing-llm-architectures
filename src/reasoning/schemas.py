"""
Pydantic models for LLM Reasoner views.

Defines:
- ReasonerInput: All data needed to generate a view
- View: The generated view (bps, confidence, reasoning)
- ViewOutput: Complete output including inputs and metadata
"""

from datetime import date
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    """Confidence level for investment view."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskFactorSummary(BaseModel):
    """Summary of a risk factor from extracted data."""
    category: str
    severity: str
    description: str


class GuidanceSummary(BaseModel):
    """Summary of management guidance."""
    direction: Optional[str] = None
    revenue_outlook: Optional[str] = None
    earnings_outlook: Optional[str] = None
    statement: Optional[str] = None


class EarningsSurprise(BaseModel):
    """Earnings surprise data (point-in-time safe)."""
    period: str = Field(..., description="Fiscal period (e.g., 'Q4 2024')")
    filing_date: str = Field(..., description="When earnings were filed (point-in-time)")

    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None

    revenue_actual: Optional[float] = None  # In billions
    revenue_estimate: Optional[float] = None  # In billions
    revenue_surprise_pct: Optional[float] = None

    num_analysts: Optional[int] = None
    estimate_dispersion: Optional[float] = Field(
        None,
        description="(epsHigh - epsLow) / |epsAvg| - proxy for uncertainty"
    )


class ReasonerInput(BaseModel):
    """
    All inputs needed to generate an investment view.

    Combines:
    - Extracted facts from SEC filings (Phase 2)
    - Analyst estimates and earnings surprise
    - Historical price data
    """
    ticker: str
    view_date: date  # filing_date + 1 day
    fiscal_period: str  # "FY2023", "Q1_2024", etc.
    sector: str

    # Trigger filing info
    trigger_filing_type: str  # "10-K" or "10-Q"
    trigger_filing_date: str  # YYYY-MM-DD

    # Latest 10-K data (may be the trigger or previous)
    latest_10k_date: Optional[str] = None
    annual_risk_factors: List[RiskFactorSummary] = Field(default_factory=list)
    annual_risk_summary: Optional[str] = None
    annual_key_events: List[str] = Field(default_factory=list)

    # Latest 10-Q data (may be the trigger or previous)
    latest_10q_date: Optional[str] = None
    quarterly_sentiment: Optional[str] = None
    quarterly_sentiment_rationale: Optional[str] = None
    quarterly_guidance: Optional[GuidanceSummary] = None
    quarterly_key_events: List[str] = Field(default_factory=list)

    # Earnings surprise (ONLY where filingDate < view_date)
    earnings_surprise: Optional[EarningsSurprise] = None

    # Historical price context
    price_return_1m: Optional[float] = None  # 1-month return (%)
    price_return_3m: Optional[float] = None  # 3-month return (%)
    price_volatility_60d: Optional[float] = None  # Annualized volatility (%)


class View(BaseModel):
    """
    Generated investment view.

    view_bps: Expected excess return in basis points (-300 to +300)
    confidence: Reliability of the view
    reasoning: Brief explanation (2-3 sentences)
    """
    view_bps: int = Field(..., ge=-300, le=300)
    confidence: Confidence
    reasoning: str = Field(..., min_length=20)

    @field_validator('view_bps')
    @classmethod
    def clamp_view_bps(cls, v: int) -> int:
        """Ensure view_bps is within bounds."""
        return max(-300, min(300, v))


class FilingTrigger(BaseModel):
    """Info about the filing that triggered this view."""
    filing_type: str  # "10-K" or "10-Q"
    filing_date: str  # YYYY-MM-DD


class ViewInputsSummary(BaseModel):
    """
    Summary of ALL inputs used for view generation.

    Mirrors the data passed to the LLM prompt for reproducibility
    and analysis of what drove each view.
    """
    # Sector
    sector: Optional[str] = None

    # Filing dates
    latest_10k_date: Optional[str] = None
    latest_10q_date: Optional[str] = None

    # Risk summary from 10-K
    risk_summary: Optional[str] = None
    num_risk_factors: Optional[int] = None

    # Sentiment from 10-Q
    sentiment: Optional[str] = None
    sentiment_rationale: Optional[str] = None

    # Guidance from 10-Q
    guidance_direction: Optional[str] = None
    guidance_revenue_outlook: Optional[str] = None
    guidance_earnings_outlook: Optional[str] = None

    # Earnings surprise
    earnings_period: Optional[str] = None
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_actual_b: Optional[float] = None  # billions
    revenue_estimate_b: Optional[float] = None  # billions
    revenue_surprise_pct: Optional[float] = None
    num_analysts: Optional[int] = None
    estimate_dispersion_pct: Optional[float] = None  # percentage

    # Price metrics
    price_return_1m_pct: Optional[float] = None
    price_return_3m_pct: Optional[float] = None
    price_volatility_60d_pct: Optional[float] = None


class ViewOutput(BaseModel):
    """
    Complete output for a generated view.

    Saved to: data/views/{TICKER}/view_{VIEW_DATE}_{FISCAL_PERIOD}.json
    """
    ticker: str
    view_date: str  # YYYY-MM-DD (filing_date + 1 day)
    fiscal_period: str  # "FY2023", "Q1_2024", etc.

    triggered_by: FilingTrigger
    inputs: ViewInputsSummary
    view: View

    def to_csv_row(self) -> dict:
        """Convert to row for views_summary.csv."""
        return {
            "ticker": self.ticker,
            "view_date": self.view_date,
            "fiscal_period": self.fiscal_period,
            "filing_type": self.triggered_by.filing_type,
            "view_bps": self.view.view_bps,
            "confidence": self.view.confidence.value,
            "sentiment": self.inputs.sentiment,
            "eps_surprise": self.inputs.eps_surprise_pct,
        }

    def get_filename(self) -> str:
        """Generate filename for this view."""
        return f"view_{self.view_date}_{self.fiscal_period}.json"
