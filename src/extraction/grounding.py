"""
Grounding validation for extracted facts (schema v2).

Third validation level of the pipeline (syntax -> schema -> grounding):
every verifiable fact must carry a verbatim source_quote, and this module
checks that the quote actually occurs in the filing text. Rule:
"not grounded -> does not exist" - ungrounded facts are dropped upstream.

Matching is done on normalized text because parsed HTML differs from what
the model saw in chunks: non-breaking spaces, typographic quotes/dashes,
soft hyphens, ligatures, line breaks.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

# Quotes shorter than this (after normalization) must match exactly:
# fuzzy matching on short strings produces false positives.
MIN_FUZZY_QUOTE_LEN = 30

DEFAULT_THRESHOLD = 90.0

# Characters normalized to plain equivalents before matching
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "­": None,  # soft hyphen
    "​": None,  # zero-width space
})

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text for robust quote matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_MAP)
    text = text.lower()
    text = _WS_RE.sub(" ", text)
    return text.strip()


@dataclass
class GroundingResult:
    grounded: bool
    score: float          # 100 = exact match; otherwise best fuzzy score
    method: str           # "exact" | "fuzzy" | "none" | "empty"


class GroundingValidator:
    """Verifies that a fact's source_quote literally occurs in the filing text."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._doc_cache: Optional[str] = None
        self._doc_norm: str = ""

    def set_document(self, document_text: str) -> None:
        """Normalize the document once; check_quote() then reuses it."""
        self._doc_cache = document_text
        self._doc_norm = normalize(document_text)

    def check_quote(self, quote: str) -> GroundingResult:
        """Check a single quote against the document set via set_document()."""
        if self._doc_cache is None:
            raise RuntimeError("set_document() must be called before check_quote()")

        quote_norm = normalize(quote)
        if not quote_norm:
            return GroundingResult(grounded=False, score=0.0, method="empty")

        if quote_norm in self._doc_norm:
            return GroundingResult(grounded=True, score=100.0, method="exact")

        if len(quote_norm) < MIN_FUZZY_QUOTE_LEN:
            return GroundingResult(grounded=False, score=0.0, method="none")

        score = fuzz.partial_ratio(quote_norm, self._doc_norm)
        if score >= self.threshold:
            return GroundingResult(grounded=True, score=float(score), method="fuzzy")
        return GroundingResult(grounded=False, score=float(score), method="none")
