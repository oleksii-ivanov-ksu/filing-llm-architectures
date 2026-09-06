"""
One-stage baseline generator: filing text (+ quantitative summary) -> view,
in a single LLM call. No validated intermediate fact layer.

Grounding is still applied to the model's `supporting_quotes` so the one-stage
and two-stage pipelines are compared on the same non-fabrication metric.
"""
import json
from typing import List, Optional, Dict

import openai
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.extraction.parser import SECFilingParser
from src.extraction.chunker import TokenChunker
from src.extraction.grounding import GroundingValidator, DEFAULT_THRESHOLD
from src.extraction.extractor import (
    _parse_json_lenient, TransientLLMError, LLMTruncated,
)


class SingleStageView(BaseModel):
    """Investment view produced directly from a filing in one call."""
    view_bps: int = Field(..., description="Expected quarterly excess return, basis points, clamped [-300, 300]")
    confidence: str = Field(..., description="high | medium | low")
    reasoning: str = Field(..., min_length=10, description="2-3 sentences naming the drivers")
    supporting_quotes: List[str] = Field(
        default_factory=list,
        description="1-3 SHORT verbatim fragments (<=160 chars) copied EXACTLY from the filing that justify the view",
    )

    @field_validator("view_bps", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            return max(-300, min(300, int(round(float(v)))))
        except Exception:
            return 0

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        v = str(v).strip().lower()
        return v if v in ("high", "medium", "low") else "medium"


SYSTEM_PROMPT = """You are a financial analyst. You read a company's SEC filing and \
some quantitative context, and you produce ONE forward-looking investment view: the \
expected excess return of the stock next quarter, in basis points.

Rules:
- Output STRICT JSON matching the schema. No prose outside JSON.
- view_bps: integer in [-300, 300]. Positive = expected outperformance.
- confidence: "high", "medium", or "low".
- reasoning: 2-3 sentences naming the specific drivers behind the number.
- supporting_quotes: 1-3 SHORT verbatim fragments (<=160 chars each) copied EXACTLY \
from the filing text that justify the view. Copy word for word; do not paraphrase. \
If you cannot support the view with real text, lower confidence rather than invent quotes.
- Do NOT fabricate. Every quote must be findable by exact search in the filing."""

USER_PROMPT = """Produce an investment view for {ticker} ({sector}) as of {view_date}.

Quantitative context (point-in-time):
- Triggered by: {trigger_filing_type} filed {trigger_filing_date}
- EPS surprise: {eps_surprise}
- Revenue surprise: {revenue_surprise}
- Analyst dispersion: {dispersion}
- 1M / 3M price return: {ret1m} / {ret3m}
- 60d volatility: {vol}

Filing text (Risk Factors / MD&A / Liquidity / Business):
{filing_text}

Return JSON: {{"view_bps": int, "confidence": "high|medium|low", "reasoning": str, "supporting_quotes": [str, ...]}}
"""


def _fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "n/a"


class SingleStageGenerator:
    """Single LLM call: (filing text + quant summary) -> SingleStageView, grounded."""

    def __init__(self, api_key, model, base_url=None, temperature=0.3,
                 max_tokens=1500, grounding_threshold=DEFAULT_THRESHOLD,
                 max_input_tokens=80000):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url \
            else openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunker = TokenChunker(model=model)
        self.max_input_tokens = max_input_tokens
        self.grounding_threshold = grounding_threshold
        self.input_tokens = 0
        self.output_tokens = 0

    def _filing_text(self, filing_path) -> str:
        html = open(filing_path, "r", encoding="utf-8", errors="ignore").read()
        sections = SECFilingParser(html, filename=str(filing_path)).extract_all_sections()
        if not sections:
            return ""
        chunks = self.chunker.chunk_sections(sections, max_total_tokens=self.max_input_tokens)
        return "\n\n".join(chunks)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, TransientLLMError)),
        wait=wait_exponential(multiplier=2, min=10, max=120), stop=stop_after_attempt(7),
    )
    def _call(self, filing_text, quant) -> Dict:
        user = USER_PROMPT.format(filing_text=filing_text, **quant)
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
        )
        if resp.usage:
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens
        if not resp.choices:
            raise TransientLLMError("empty response")
        choice = resp.choices[0]
        text = choice.message.content
        if not text:
            raise TransientLLMError("empty content")
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMTruncated(f"truncated at max_tokens={self.max_tokens}")
        return _parse_json_lenient(text)

    def generate(self, ticker, filing_path, filing_type, filing_date, quant: dict):
        """Returns (SingleStageView, grounding_dict) or (None, None) on failure."""
        text = self._filing_text(filing_path)
        if not text:
            return None, None
        q = {
            "ticker": ticker, "sector": quant.get("sector", "n/a"),
            "view_date": quant.get("view_date", "n/a"),
            "trigger_filing_type": filing_type, "trigger_filing_date": filing_date,
            "eps_surprise": _fmt(quant.get("eps_surprise_pct"), "%"),
            "revenue_surprise": _fmt(quant.get("revenue_surprise_pct"), "%"),
            "dispersion": _fmt(quant.get("dispersion")),
            "ret1m": _fmt(quant.get("price_return_1m"), "%"),
            "ret3m": _fmt(quant.get("price_return_3m"), "%"),
            "vol": _fmt(quant.get("price_volatility_60d"), "%"),
        }
        raw = self._call(text, q)
        view = SingleStageView(**raw)

        # Ground the supporting quotes against the filing text (same metric as two-stage).
        validator = GroundingValidator(threshold=self.grounding_threshold)
        validator.set_document(text)
        total = len(view.supporting_quotes)
        grounded = sum(1 for qt in view.supporting_quotes if validator.check_quote(qt).grounded)
        ginfo = {"total_quotes": total, "grounded": grounded,
                 "dropped": total - grounded, "threshold": self.grounding_threshold}
        return view, ginfo
