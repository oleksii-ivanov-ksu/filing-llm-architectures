"""
Pydantic schemas for SEC filing extraction.

NOTE: Financial metrics (revenue, margins, EPS) are NOT extracted here.
      They come from FMP API (income_statement.json, balance_sheet.json, cash_flow.json).

      This module extracts QUALITATIVE information only:
      - Risk factors
      - Management guidance
      - Key events
      - Sentiment/tone
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# Schema v2: verifiable facts must carry a verbatim source quote that the
# GroundingValidator checks against the filing text ("not grounded -> does not exist").
SCHEMA_VERSION = "2.0"


class RiskCategory(str, Enum):
    """Categories of risk factors."""
    COMPETITION = "competition"
    REGULATORY = "regulatory"
    SUPPLY_CHAIN = "supply_chain"
    CYBERSECURITY = "cybersecurity"
    MACROECONOMIC = "macroeconomic"
    MARKET = "market"  # Market conditions, volatility, pricing
    CREDIT = "credit"  # Credit risk, defaults, counterparty
    LIQUIDITY = "liquidity"  # Liquidity, funding, capital
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    INVESTMENT = "investment"
    LEGAL = "legal"
    ENVIRONMENTAL = "environmental"
    GEOPOLITICAL = "geopolitical"
    TECHNOLOGY = "technology"
    REPUTATION = "reputation"
    STRATEGIC = "strategic"
    LABOR = "labor"  # Workforce, unions, hiring, retention
    OTHER = "other"


class Severity(str, Enum):
    """Severity levels for risks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(str, Enum):
    """
    Management sentiment/tone (5-point scale for sensitivity).

    Scale designed for realistic corporate communications:
    - Management rarely uses truly negative language in SEC filings
    - This scale captures nuances between "confident" and "concerned"
    """
    CONFIDENT = "confident"      # Strong positive: "exceeded expectations", "record performance"
    OPTIMISTIC = "optimistic"    # Moderate positive: "expect growth", "well-positioned"
    NEUTRAL = "neutral"          # Factual, balanced: no strong sentiment either way
    CAUTIOUS = "cautious"        # Hedging: "uncertain environment", "monitoring closely"
    CONCERNED = "concerned"      # Acknowledging issues: "challenging", "headwinds", "restructuring"


def _coerce_enum(value, enum_cls, default):
    """Map an LLM-provided value to a valid enum member; unknown -> default.
    Models often invent labels outside our taxonomy (e.g. 'governance', 'esg');
    coercing avoids failing the whole extraction over one out-of-vocab label."""
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    valid = {e.value: e for e in enum_cls}
    return valid.get(v, default)


class RiskFactor(BaseModel):
    """A single risk factor from SEC filing."""
    category: RiskCategory = Field(description="Risk category")
    severity: Severity = Field(description="Severity level based on language used")
    description: str = Field(description="Brief description (1-2 sentences)")
    is_new: bool = Field(default=False, description="True if this is a NEW risk not in prior filings")

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v): return _coerce_enum(v, RiskCategory, RiskCategory.OTHER)

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v): return _coerce_enum(v, Severity, Severity.MEDIUM)
    source_quote: str = Field(
        description="ONE short verbatim fragment (max ~160 characters) copied EXACTLY from the filing "
                    "that supports this risk factor. Copy a single clause/sentence, do not paraphrase, "
                    "keep it short. If no supporting text exists, do not output this risk factor."
    )


class Guidance(BaseModel):
    """Management guidance/outlook from the filing."""
    direction: Optional[Sentiment] = Field(default=None, description="Overall direction: positive/negative/neutral")

    @field_validator("direction", mode="before")
    @classmethod
    def _dir(cls, v): return _coerce_enum(v, Sentiment, None) if v else None
    statement: Optional[str] = Field(default=None, description="Key guidance statement (direct quote or paraphrase)")
    revenue_outlook: Optional[str] = Field(default=None, description="Revenue guidance if mentioned - ONLY if specific numbers given")
    earnings_outlook: Optional[str] = Field(default=None, description="Earnings guidance if mentioned - ONLY if specific numbers given")
    source_quote: Optional[str] = Field(
        default=None,
        description="ONE short verbatim fragment (max ~160 chars) copied EXACTLY from the filing that supports the guidance direction, if one exists. Keep it short, do not paraphrase."
    )


class KeyEvent(BaseModel):
    """Significant event or change mentioned in filing."""
    event_type: str = Field(description="Type: acquisition, divestiture, restructuring, litigation, product_launch, etc.")
    description: str = Field(description="Brief description of the event")
    impact: Optional[str] = Field(default=None, description="Stated or implied impact")
    source_quote: str = Field(
        description="ONE short verbatim fragment (max ~160 characters) copied EXACTLY from the filing "
                    "that reports this event. Copy a single clause/sentence, do not paraphrase, keep it short. "
                    "If no supporting text exists, do not output this event."
    )


class ExtractedFacts(BaseModel):
    """
    Qualitative facts extracted from SEC filing.

    NOTE: Financial numbers come from FMP API, not from this extraction.
    This focuses on QUALITATIVE information that LLM excels at:
    - Risk analysis
    - Management tone/sentiment
    - Forward-looking statements
    - Key events
    """
    # Metadata
    ticker: str = Field(description="Stock ticker symbol")
    filing_date: str = Field(description="Filing date (YYYY-MM-DD)")
    filing_type: str = Field(description="Filing type: 10-K or 10-Q")
    fiscal_period: str = Field(description="Fiscal period: Q1/Q2/Q3/Q4 YYYY or FY YYYY")

    # Risk factors (from Item 1A)
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Key risk factors identified (top 5-10)"
    )
    risk_summary: Optional[str] = Field(
        default=None,
        description="One sentence summary of overall risk profile"
    )

    # Management guidance (from MD&A)
    guidance: Optional[Guidance] = Field(
        default=None,
        description="Forward-looking statements from management"
    )

    # Overall sentiment (from MD&A tone)
    sentiment: Sentiment = Field(
        default=Sentiment.NEUTRAL,
        description="Overall sentiment/tone of MD&A section"
    )
    sentiment_rationale: Optional[str] = Field(
        default=None,
        description="Brief explanation of sentiment assessment"
    )

    @field_validator("sentiment", mode="before")
    @classmethod
    def _sent(cls, v): return _coerce_enum(v, Sentiment, Sentiment.NEUTRAL)

    # Key events
    key_events: List[KeyEvent] = Field(
        default_factory=list,
        description="Major events: M&A, restructuring, litigation, etc."
    )

    # Competitive position
    competitive_position: Optional[str] = Field(
        default=None,
        description="Management's view on competitive position (1-2 sentences)"
    )

    # Extraction metadata
    extraction_confidence: Optional[str] = Field(
        default=None,
        description="LLM's confidence in extraction: high/medium/low"
    )

    # Schema v2: grounding summary (populated by GroundingValidator, not the LLM)
    grounding: Optional[Dict] = Field(
        default=None,
        description="Grounding validation summary: total_facts, grounded, dropped, threshold"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AMZN",
                "filing_date": "2024-02-01",
                "filing_type": "10-K",
                "fiscal_period": "FY 2023",
                "risk_factors": [
                    {
                        "category": "competition",
                        "severity": "high",
                        "description": "Intense competition in cloud services from Microsoft Azure and Google Cloud",
                        "is_new": False,
                        "source_quote": "The market segments in which we compete are rapidly evolving and intensely competitive, and we face a broad array of competitors, including companies with greater resources in cloud computing."
                    },
                    {
                        "category": "regulatory",
                        "severity": "medium",
                        "description": "Increased antitrust scrutiny in US and EU markets",
                        "is_new": True,
                        "source_quote": "We are subject to increasing governmental scrutiny, including investigations by competition authorities in the United States and the European Union."
                    }
                ],
                "risk_summary": "Primary risks are competitive pressure in cloud and regulatory scrutiny",
                "guidance": {
                    "direction": "optimistic",
                    "statement": "Management expects continued growth driven by AWS and advertising",
                    "revenue_outlook": "Q1 2024 revenue between $138B-$143.5B",
                    "earnings_outlook": "Operating income between $8B-$12B"
                },
                "sentiment": "optimistic",
                "sentiment_rationale": "Uses phrases like 'strong momentum' and 'well-positioned for growth'",
                "key_events": [
                    {
                        "event_type": "restructuring",
                        "description": "Completed workforce reduction of 27,000 employees",
                        "impact": "Expected annual savings of $2B",
                        "source_quote": "In the first quarter of 2023, we completed a workforce reduction of approximately 27,000 roles as part of our broader cost reduction initiatives."
                    }
                ],
                "competitive_position": "Maintains market leadership in e-commerce and cloud computing",
                "extraction_confidence": "high"
            }
        }
