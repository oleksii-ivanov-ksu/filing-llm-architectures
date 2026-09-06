"""
SEC Filing Extraction Module.

Extracts qualitative information from SEC filings using LLM:
- Risk factors
- Management guidance
- Sentiment/tone
- Key events

Financial metrics (revenue, margins) come from FMP API, not this module.
"""

from .schemas import ExtractedFacts, RiskFactor, Guidance, KeyEvent
from .parser import SECFilingParser
from .chunker import TokenChunker, estimate_cost
from .extractor import FilingExtractor, CostTracker

__all__ = [
    "ExtractedFacts",
    "RiskFactor",
    "Guidance",
    "KeyEvent",
    "SECFilingParser",
    "TokenChunker",
    "estimate_cost",
    "FilingExtractor",
    "CostTracker",
]
