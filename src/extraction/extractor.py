"""
LLM-Extractor: Extract qualitative information from SEC filings.

Features:
- Rate limiting (respects OpenAI limits)
- Exponential backoff retry
- Cost tracking
- Resume support (skip existing outputs)
- Incident logging for failed/problematic extractions
"""
import json
import re
import logging
import time
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from datetime import datetime
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .schemas import ExtractedFacts, SCHEMA_VERSION
from .parser import SECFilingParser
from .chunker import TokenChunker, estimate_cost
from .grounding import GroundingValidator, DEFAULT_THRESHOLD


class TransientLLMError(Exception):
    """Retryable LLM error (empty response, transient API hiccup)."""


class LLMTruncated(Exception):
    """Non-retryable: the model's output hit max_tokens and is truncated.
    Retrying the identical request would truncate again."""


def _parse_json_lenient(text: str) -> Dict:
    """Parse a JSON object from an LLM response that may be wrapped in markdown
    fences or preceded/followed by prose (common with Claude and some others).
    Falls back to the substring between the first '{' and the last '}'."""
    text = text.strip()
    # 1) direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) strip a leading ```json / ``` fence and trailing ```
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # 3) take the outermost {...} span
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])  # may raise JSONDecodeError -> logged upstream
    raise json.JSONDecodeError("No JSON object found in response", text, 0)
from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT

logger = logging.getLogger(__name__)


class IncidentLogger:
    """
    Log extraction incidents to a JSONL file for later analysis.

    Incidents include:
    - Missing sections (risk_factors, mda, etc.)
    - Parsing errors
    - LLM errors
    - Validation errors

    File is flushed immediately after each write to survive Ctrl+C.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize incident logger.

        Args:
            output_dir: Directory to save incidents file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create incidents file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.incidents_file = self.output_dir / f"extraction_incidents_{timestamp}.jsonl"
        self.file_handle = open(self.incidents_file, 'a', encoding='utf-8')

        # Register cleanup on exit (handles Ctrl+C)
        atexit.register(self.close)

        logger.info(f"Incident logger initialized: {self.incidents_file}")

    def log(
        self,
        ticker: str,
        filing_date: str,
        filing_type: str,
        incident_type: str,
        message: str,
        details: Optional[Dict] = None
    ):
        """
        Log an incident.

        Args:
            ticker: Stock ticker
            filing_date: Filing date
            filing_type: 10-K or 10-Q
            incident_type: Type of incident (missing_section, parse_error, llm_error, validation_error)
            message: Human-readable message
            details: Additional details dict
        """
        incident = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "filing_date": filing_date,
            "filing_type": filing_type,
            "incident_type": incident_type,
            "message": message,
            "details": details or {}
        }

        # Write and flush immediately
        self.file_handle.write(json.dumps(incident) + "\n")
        self.file_handle.flush()

    def close(self):
        """Close file handle."""
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()

    def __del__(self):
        self.close()


class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 20):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute (default 20 for tier 1 accounts)
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit. Thread-safe."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class CostTracker:
    """Thread-safe API cost tracker."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.model = "gpt-4o-mini"
        self._lock = threading.Lock()

    def add(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

    @property
    def total_cost(self) -> float:
        return estimate_cost(
            self.total_input_tokens,
            self.total_output_tokens,
            self.model
        )

    def summary(self) -> str:
        return (
            f"Requests: {self.total_requests}, "
            f"Input: {self.total_input_tokens:,} tokens, "
            f"Output: {self.total_output_tokens:,} tokens, "
            f"Cost: ${self.total_cost:.4f}"
        )


class FilingExtractor:
    """
    Extract qualitative facts from SEC filings using LLM.

    Focuses on:
    - Risk factors
    - Management guidance
    - Sentiment/tone
    - Key events
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        requests_per_minute: int = 40,  # ~1.5 sec between requests
        incidents_dir: Optional[Path] = None,  # Directory for incident logs
        max_workers: int = 5,  # Parallel workers for extraction
        enable_grounding: bool = False,  # Schema v2: verify source_quote of each fact
        grounding_threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Initialize the extractor.

        Args:
            api_key: OpenAI API key (or OpenRouter key)
            model: Model to use for extraction (from config.yaml)
            base_url: API base URL (None=OpenAI, "https://openrouter.ai/api/v1"=OpenRouter)
            temperature: Temperature for generation (low for factual extraction)
            max_tokens: Maximum tokens for response
            requests_per_minute: Rate limit (40/min for 200K TPM @ ~5K tokens/req)
            incidents_dir: Directory for incident logs (default: data/extracted)
            max_workers: Number of parallel workers (default 5)
        """
        # Support OpenAI and OpenRouter (OpenAI-compatible API)
        if base_url:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"X-Title": "FilingAwareLLMViewsforPortfolioOptimization"}
            )
        else:
            self.client = openai.OpenAI(api_key=api_key)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers

        self.chunker = TokenChunker(model=model)
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.cost_tracker = CostTracker()

        # Schema v2: grounding validation of source quotes
        self.enable_grounding = enable_grounding
        self.grounding_threshold = grounding_threshold

        # Initialize incident logger
        incidents_path = incidents_dir or Path("data/extracted")
        self.incident_logger = IncidentLogger(incidents_path)
        self.cost_tracker.model = model

        logger.info(f"Initialized FilingExtractor with model={model}, base_url={base_url or 'OpenAI'}, rate={requests_per_minute} req/min, workers={max_workers}")

    def _clean_null_strings(self, obj):
        """
        Recursively convert string representations of null to actual None.
        LLM sometimes returns "null", "NULL", "None" as strings instead of JSON null.
        """
        if isinstance(obj, dict):
            return {k: self._clean_null_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_null_strings(item) for item in obj]
        elif isinstance(obj, str) and obj.lower() in ('null', 'none', 'n/a', 'na', ''):
            return None
        return obj

    @retry(
        # Retry ONLY transient errors (network, rate limit, empty response).
        # NOT truncation or malformed JSON — retrying identical requests just burns
        # money and time (the v1 code retried JSONDecodeError 7x because it subclasses
        # ValueError; that caused the expensive retry storms on verbose models).
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, TransientLLMError)),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(7)
    )
    def _call_llm(self, prompt: str) -> Dict:
        """Call LLM. Transient failures retry; truncation / bad JSON fail fast."""
        self.rate_limiter.wait()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )

        # Track usage (OpenRouter may not return usage)
        if response.usage:
            self.cost_tracker.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        else:
            logger.debug("No usage data in response (OpenRouter may not provide this)")

        # Transient: empty response — worth retrying
        if not response.choices:
            raise TransientLLMError("Empty response from LLM - no choices returned")

        choice = response.choices[0]
        response_text = choice.message.content
        if not response_text:
            raise TransientLLMError("Empty response content from LLM")

        # Non-retryable: output hit the token cap and is truncated -> invalid JSON.
        # Retrying the same request truncates again, so fail fast (logged upstream).
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMTruncated(
                f"Output truncated at max_tokens={self.max_tokens} (finish_reason=length)"
            )

        # Robust parse: some models (e.g. Claude) wrap JSON in markdown fences or add
        # a preamble despite json_object mode. Strip that before json.loads.
        # JSONDecodeError here is NOT retried (not a TransientLLMError).
        return _parse_json_lenient(response_text)

    def _apply_grounding(
        self,
        facts: ExtractedFacts,
        document_text: str,
        ticker: str,
        filing_type: str,
        filing_date: str,
    ) -> ExtractedFacts:
        """
        Schema v2 grounding: keep only facts whose source_quote is found in the
        filing text. Dropped facts are logged as `ungrounded_fact` incidents.
        Attaches a `grounding` summary to the facts object (serialized later).
        """
        validator = GroundingValidator(threshold=self.grounding_threshold)
        validator.set_document(document_text)

        total = 0
        dropped = 0

        def keep(items, quote_required: bool):
            nonlocal total, dropped
            survivors = []
            for item in items:
                quote = getattr(item, "source_quote", None)
                if not quote:
                    # Required-quote facts with no quote are dropped; optional ones kept.
                    if quote_required:
                        total += 1
                        dropped += 1
                        self.incident_logger.log(
                            ticker=ticker, filing_date=filing_date, filing_type=filing_type,
                            incident_type="ungrounded_fact",
                            message="Fact has no source_quote",
                            details={"fact_type": type(item).__name__, "score": 0.0, "reason": "missing_quote"},
                        )
                    else:
                        survivors.append(item)
                    continue
                total += 1
                result = validator.check_quote(quote)
                if result.grounded:
                    survivors.append(item)
                else:
                    dropped += 1
                    self.incident_logger.log(
                        ticker=ticker, filing_date=filing_date, filing_type=filing_type,
                        incident_type="ungrounded_fact",
                        message="source_quote not found in filing text",
                        details={
                            "fact_type": type(item).__name__,
                            "quote": quote[:300],
                            "score": round(result.score, 1),
                            "method": result.method,
                        },
                    )
            return survivors

        facts.risk_factors = keep(facts.risk_factors, quote_required=True)
        facts.key_events = keep(facts.key_events, quote_required=True)
        # Guidance quote is optional; verify only if present, never drop guidance itself.
        if facts.guidance and facts.guidance.source_quote:
            total += 1
            if not validator.check_quote(facts.guidance.source_quote).grounded:
                dropped += 1
                facts.guidance.source_quote = None  # strip unverifiable quote, keep guidance

        facts.grounding = {
            "schema_version": SCHEMA_VERSION,
            "total_facts": total,
            "grounded": total - dropped,
            "dropped": dropped,
            "threshold": self.grounding_threshold,
        }
        return facts

    def extract_from_file(
        self,
        filing_path: Path,
        ticker: str,
        filing_type: str,
        filing_date: str
    ) -> Optional[ExtractedFacts]:
        """
        Extract facts from a single SEC filing HTML file.

        Args:
            filing_path: Path to HTML file
            ticker: Stock ticker
            filing_type: "10-K" or "10-Q"
            filing_date: Filing date (YYYY-MM-DD)

        Returns:
            ExtractedFacts object or None if extraction fails
        """
        try:
            # Read HTML file
            with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            # Parse the filing
            parser = SECFilingParser(html_content, filename=filing_path.name)

            # Extract key sections
            sections = parser.extract_all_sections()

            if not sections:
                logger.warning(f"No sections extracted from {filing_path}")
                self.incident_logger.log(
                    ticker=ticker,
                    filing_date=filing_date,
                    filing_type=filing_type,
                    incident_type="no_sections",
                    message="No sections could be extracted from filing",
                    details={"file_path": str(filing_path)}
                )
                return None

            # Log missing important sections (context-aware)
            # - 10-K: mda required; risk_factors required only after 2005 (SEC rule)
            # - 10-Q: mda required; risk_factors optional (often "no material changes")
            expected_sections = ['mda']  # Always required

            # Risk factors required for 10-K filings after 2005
            if filing_type == '10-K':
                try:
                    year = int(filing_date[:4])
                    if year >= 2006:
                        expected_sections.append('risk_factors')
                except (ValueError, IndexError):
                    pass

            missing_sections = [s for s in expected_sections if s not in sections]
            if missing_sections:
                self.incident_logger.log(
                    ticker=ticker,
                    filing_date=filing_date,
                    filing_type=filing_type,
                    incident_type="missing_sections",
                    message=f"Missing sections: {', '.join(missing_sections)}",
                    details={
                        "missing": missing_sections,
                        "found": list(sections.keys()),
                        "file_path": str(filing_path)
                    }
                )

            # Determine fiscal period from filing
            fiscal_period = self._determine_fiscal_period(filing_type, filing_date)

            # Extract facts
            facts = self._extract_facts(
                sections=sections,
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                fiscal_period=fiscal_period
            )

            return facts

        except Exception as e:
            logger.error(f"Error extracting from {filing_path}: {e}")
            self.incident_logger.log(
                ticker=ticker,
                filing_date=filing_date,
                filing_type=filing_type,
                incident_type="extraction_error",
                message=str(e),
                details={"file_path": str(filing_path), "error_type": type(e).__name__}
            )
            return None

    def _determine_fiscal_period(self, filing_type: str, filing_date: str) -> str:
        """Determine fiscal period from filing type and date."""
        try:
            dt = datetime.strptime(filing_date, "%Y-%m-%d")

            if filing_type == "10-K":
                # Annual report - fiscal year is usually previous year
                return f"FY {dt.year - 1}" if dt.month <= 3 else f"FY {dt.year}"
            else:
                # Quarterly - determine quarter from filing month
                month = dt.month
                year = dt.year
                if month <= 3:
                    return f"Q4 {year - 1}"
                elif month <= 6:
                    return f"Q1 {year}"
                elif month <= 9:
                    return f"Q2 {year}"
                else:
                    return f"Q3 {year}"
        except:
            return "Unknown"

    def _extract_facts(
        self,
        sections: Dict[str, str],
        ticker: str,
        filing_type: str,
        filing_date: str,
        fiscal_period: str
    ) -> Optional[ExtractedFacts]:
        """
        Extract facts from parsed sections using LLM.
        """
        # Combine all sections - GPT-4o-mini has 128K context
        # Typical filing sections: 40-60K tokens total, well within limit
        # We prioritize and limit to 80K tokens to leave room for prompt/response
        chunks = self.chunker.chunk_sections(sections, max_total_tokens=80000)

        if not chunks:
            logger.warning(f"No text chunks to process for {ticker}")
            return None

        # Combine all chunks into single text
        primary_text = "\n\n".join(chunks)

        # Build prompt
        schema_str = json.dumps(ExtractedFacts.model_json_schema(), indent=2)

        prompt = EXTRACTION_USER_PROMPT.format(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            filing_text=primary_text,
            schema=schema_str
        )

        try:
            # Call LLM
            facts_dict = self._call_llm(prompt)

            # Clean up null strings (LLM sometimes returns "null", "NULL", "None" as strings)
            facts_dict = self._clean_null_strings(facts_dict)

            # Add metadata
            facts_dict["ticker"] = ticker
            facts_dict["filing_type"] = filing_type
            facts_dict["filing_date"] = filing_date
            facts_dict["fiscal_period"] = fiscal_period

            # Validate with Pydantic
            facts = ExtractedFacts(**facts_dict)

            # Schema v2: third validation level — grounding.
            # Drop facts whose source_quote is not found in the filing text.
            if self.enable_grounding:
                facts = self._apply_grounding(
                    facts, primary_text, ticker, filing_type, filing_date
                )

            # Log if extraction has low quality indicators
            if not facts.risk_factors:
                self.incident_logger.log(
                    ticker=ticker,
                    filing_date=filing_date,
                    filing_type=filing_type,
                    incident_type="empty_risk_factors",
                    message="No risk factors extracted",
                    details={"sections_provided": list(sections.keys()) if 'sections' in dir() else []}
                )

            logger.info(f"Extracted facts from {ticker} {filing_type} {filing_date}")
            return facts

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {ticker}: {e}")
            self.incident_logger.log(
                ticker=ticker,
                filing_date=filing_date,
                filing_type=filing_type,
                incident_type="json_decode_error",
                message=str(e),
                details={"error_type": "JSONDecodeError"}
            )
            return None

        except Exception as e:
            logger.error(f"Error calling LLM for {ticker}: {e}")
            self.incident_logger.log(
                ticker=ticker,
                filing_date=filing_date,
                filing_type=filing_type,
                incident_type="llm_error",
                message=str(e),
                details={"error_type": type(e).__name__}
            )
            return None

    def _process_single_filing(
        self,
        filing_file: Path,
        ticker: str,
        ticker_output_dir: Path
    ) -> Tuple[str, Optional[ExtractedFacts]]:
        """
        Process a single filing. Thread-safe helper for parallel processing.

        Returns:
            Tuple of (filing_id, facts or None)
        """
        filename = filing_file.stem
        parts = filename.split('_')

        if len(parts) < 2:
            logger.warning(f"Cannot parse filename: {filename}")
            return filename, None

        filing_type = parts[0]
        filing_date = parts[1]
        filing_id = f"{filing_type}_{filing_date}"

        # Extract facts
        facts = self.extract_from_file(
            filing_path=filing_file,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date
        )

        if facts:
            # Save to file (thread-safe via separate files)
            output_file = ticker_output_dir / f"{filing_id}.json"
            with open(output_file, 'w') as f:
                json.dump(facts.model_dump(), f, indent=2)

        return filing_id, facts

    def extract_batch(
        self,
        filings_dir: Path,
        ticker: str,
        output_dir: Path,
        resume: bool = True
    ) -> Dict[str, ExtractedFacts]:
        """
        Extract facts from all filings for a ticker using parallel processing.

        Args:
            filings_dir: Directory containing filing HTML files
            ticker: Stock ticker
            output_dir: Directory to save extracted facts
            resume: Skip existing output files

        Returns:
            Dictionary mapping filing_id to ExtractedFacts
        """
        results = {}

        # Get all HTML files
        filing_files = sorted(filings_dir.glob("*.html"))

        if not filing_files:
            logger.warning(f"No HTML files found in {filings_dir}")
            return results

        # Create output directory
        ticker_output_dir = output_dir / ticker
        ticker_output_dir.mkdir(parents=True, exist_ok=True)

        # Filter files to process (skip already processed if resume=True)
        files_to_process = []
        for filing_file in filing_files:
            filename = filing_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                filing_id = f"{parts[0]}_{parts[1]}"
                output_file = ticker_output_dir / f"{filing_id}.json"
                if resume and output_file.exists():
                    logger.info(f"Skipping {filing_id} (already exists)")
                    continue
            files_to_process.append(filing_file)

        if not files_to_process:
            logger.info(f"All filings for {ticker} already processed")
            return results

        logger.info(f"Processing {len(files_to_process)} filings for {ticker} with {self.max_workers} workers")

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_filing,
                    filing_file,
                    ticker,
                    ticker_output_dir
                ): filing_file
                for filing_file in files_to_process
            }

            for future in as_completed(futures):
                filing_file = futures[future]
                try:
                    filing_id, facts = future.result()
                    if facts:
                        results[filing_id] = facts
                except Exception as e:
                    logger.error(f"Error processing {filing_file}: {e}")

        logger.info(
            f"Extracted {len(results)} filings for {ticker}. "
            f"{self.cost_tracker.summary()}"
        )

        return results

    def extract_all(
        self,
        raw_data_dir: Path,
        output_dir: Path,
        tickers: List[str],
        resume: bool = True
    ) -> Dict[str, Dict[str, ExtractedFacts]]:
        """
        Extract facts from all filings for multiple tickers.

        Args:
            raw_data_dir: Base directory with raw data
            output_dir: Directory to save extracted facts
            tickers: List of tickers to process
            resume: Skip existing output files

        Returns:
            Nested dictionary: ticker -> filing_id -> ExtractedFacts
        """
        all_results = {}

        for ticker in tickers:
            filings_dir = raw_data_dir / ticker / "filings"

            if not filings_dir.exists():
                logger.warning(f"No filings directory for {ticker}")
                continue

            results = self.extract_batch(
                filings_dir=filings_dir,
                ticker=ticker,
                output_dir=output_dir,
                resume=resume
            )

            if results:
                all_results[ticker] = results

        logger.info(
            f"Extraction complete. Total: {self.cost_tracker.summary()}"
        )

        return all_results
