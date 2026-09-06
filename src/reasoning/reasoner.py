"""
LLM Reasoner for generating investment views.

Uses Grok 4.1 via OpenRouter to generate views from ReasonerInput.
Includes:
- Rate limiting
- Retry with exponential backoff
- JSON parsing and validation
"""

import json
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .schemas import (
    ReasonerInput,
    View,
    ViewOutput,
    ViewInputsSummary,
    FilingTrigger,
    Confidence,
)
from .prompts import REASONER_SYSTEM_PROMPT, format_user_prompt

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class ViewReasoner:
    """
    Generate investment views using LLM.

    Calls Grok 4.1 via OpenRouter to reason about
    extracted facts and analyst data.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "x-ai/grok-4.1-fast",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        max_tokens: int = 500,
        requests_per_minute: int = 30,
    ):
        """
        Initialize the reasoner.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: Grok 4.1)
            base_url: API base URL
            temperature: Generation temperature
            max_tokens: Max output tokens
            requests_per_minute: Rate limit
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"X-Title": "FilingAwareLLMViewsforPortfolioOptimization"}
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limiter = RateLimiter(requests_per_minute)

        logger.info(f"Initialized ViewReasoner with model={model}")

    @retry(
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APIConnectionError,
            ValueError,
        )),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
    )
    def _call_llm(self, user_prompt: str) -> Dict:
        """
        Call LLM with retry logic.

        Args:
            user_prompt: Formatted user prompt

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If response is empty or invalid JSON
        """
        self.rate_limiter.wait()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REASONER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        if not response.choices:
            raise ValueError("Empty response from LLM")

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty content from LLM")

        return json.loads(content)

    def generate_view(
        self,
        input_data: ReasonerInput,
    ) -> Optional[ViewOutput]:
        """
        Generate investment view for a single input.

        Args:
            input_data: ReasonerInput with all necessary data

        Returns:
            ViewOutput or None if generation fails
        """
        try:
            # Format prompt
            user_prompt = format_user_prompt(input_data)

            # Call LLM
            response = self._call_llm(user_prompt)

            # Parse view
            view_bps = response.get("view_bps", 0)
            confidence_str = response.get("confidence", "medium").lower()
            reasoning = response.get("reasoning", "")

            # Validate and clamp
            view_bps = max(-300, min(300, int(view_bps)))

            # Map confidence
            confidence_map = {
                "high": Confidence.HIGH,
                "medium": Confidence.MEDIUM,
                "low": Confidence.LOW,
            }
            confidence = confidence_map.get(confidence_str, Confidence.MEDIUM)

            # Build view
            view = View(
                view_bps=view_bps,
                confidence=confidence,
                reasoning=reasoning if len(reasoning) >= 20 else "View generated based on filing analysis.",
            )

            # Build inputs summary (all data passed to prompt)
            surprise = input_data.earnings_surprise
            guidance = input_data.quarterly_guidance

            inputs_summary = ViewInputsSummary(
                # Sector
                sector=input_data.sector,

                # Filing dates
                latest_10k_date=input_data.latest_10k_date,
                latest_10q_date=input_data.latest_10q_date,

                # Risk summary
                risk_summary=input_data.annual_risk_summary,
                num_risk_factors=len(input_data.annual_risk_factors) if input_data.annual_risk_factors else 0,

                # Sentiment
                sentiment=input_data.quarterly_sentiment,
                sentiment_rationale=input_data.quarterly_sentiment_rationale,

                # Guidance
                guidance_direction=guidance.direction if guidance else None,
                guidance_revenue_outlook=guidance.revenue_outlook if guidance else None,
                guidance_earnings_outlook=guidance.earnings_outlook if guidance else None,

                # Earnings surprise
                earnings_period=surprise.period if surprise else None,
                eps_actual=surprise.eps_actual if surprise else None,
                eps_estimate=surprise.eps_estimate if surprise else None,
                eps_surprise_pct=surprise.eps_surprise_pct if surprise else None,
                revenue_actual_b=surprise.revenue_actual if surprise else None,
                revenue_estimate_b=surprise.revenue_estimate if surprise else None,
                revenue_surprise_pct=surprise.revenue_surprise_pct if surprise else None,
                num_analysts=surprise.num_analysts if surprise else None,
                estimate_dispersion_pct=(surprise.estimate_dispersion * 100) if surprise and surprise.estimate_dispersion else None,

                # Price metrics
                price_return_1m_pct=input_data.price_return_1m,
                price_return_3m_pct=input_data.price_return_3m,
                price_volatility_60d_pct=input_data.price_volatility_60d,
            )

            # Build trigger info
            trigger = FilingTrigger(
                filing_type=input_data.trigger_filing_type,
                filing_date=input_data.trigger_filing_date,
            )

            # Build output
            output = ViewOutput(
                ticker=input_data.ticker,
                view_date=input_data.view_date.strftime("%Y-%m-%d"),
                fiscal_period=input_data.fiscal_period,
                triggered_by=trigger,
                inputs=inputs_summary,
                view=view,
            )

            logger.info(
                f"Generated view for {input_data.ticker} {input_data.fiscal_period}: "
                f"{view_bps} bps ({confidence.value})"
            )

            return output

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {input_data.ticker}: {e}")
            return None

        except Exception as e:
            logger.error(f"Error generating view for {input_data.ticker}: {e}")
            return None

    def generate_views_batch(
        self,
        inputs: List[ReasonerInput],
        output_dir: Path,
        resume: bool = True,
    ) -> List[ViewOutput]:
        """
        Generate views for multiple inputs.

        Args:
            inputs: List of ReasonerInput objects
            output_dir: Directory to save view JSONs
            resume: Skip existing output files

        Returns:
            List of generated ViewOutput objects
        """
        output_dir = Path(output_dir)
        results = []

        for i, input_data in enumerate(inputs):
            # Create ticker directory
            ticker_dir = output_dir / input_data.ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            # Filename: view_{view_date}_{fiscal_period}.json
            filename = f"view_{input_data.view_date}_{input_data.fiscal_period}.json"
            output_file = ticker_dir / filename

            if resume and output_file.exists():
                logger.debug(f"Skipping {input_data.ticker} {input_data.fiscal_period} (exists)")
                continue

            # Generate view
            view_output = self.generate_view(input_data)

            if view_output:
                # Save to file
                with open(output_file, 'w') as f:
                    json.dump(view_output.model_dump(mode='json'), f, indent=2, default=str)
                results.append(view_output)

            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(inputs)} views generated")

        logger.info(f"Generated {len(results)} new views")
        return results

    def save_summary_csv(
        self,
        output_dir: Path,
    ) -> Path:
        """
        Generate views_summary.csv from all view files.

        Args:
            output_dir: Directory containing view JSONs

        Returns:
            Path to generated CSV
        """
        import csv

        output_dir = Path(output_dir)
        csv_path = output_dir / "views_summary.csv"

        rows = []
        for ticker_dir in output_dir.iterdir():
            if not ticker_dir.is_dir():
                continue

            for view_file in ticker_dir.glob("view_*.json"):
                try:
                    with open(view_file, 'r') as f:
                        data = json.load(f)

                    rows.append({
                        "ticker": data.get("ticker"),
                        "view_date": data.get("view_date"),
                        "fiscal_period": data.get("fiscal_period"),
                        "filing_type": data.get("triggered_by", {}).get("filing_type"),
                        "view_bps": data.get("view", {}).get("view_bps"),
                        "confidence": data.get("view", {}).get("confidence"),
                        "sentiment": data.get("inputs", {}).get("sentiment"),
                        "eps_surprise": data.get("inputs", {}).get("eps_surprise_pct"),
                    })
                except Exception as e:
                    logger.error(f"Error reading {view_file}: {e}")

        # Sort by ticker and date
        rows.sort(key=lambda x: (x["ticker"], x["view_date"]))

        # Write CSV
        if rows:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"Saved {len(rows)} views to {csv_path}")

        return csv_path
