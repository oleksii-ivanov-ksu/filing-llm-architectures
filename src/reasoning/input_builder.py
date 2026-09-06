"""
Input Builder for LLM Reasoner.

Builds ReasonerInput from:
- Extracted facts (Phase 2)
- Analyst estimates (FMP)
- Income statement for actuals (FMP)
- Historical prices

View generation logic:
- Each filing triggers a view on filing_date + 1 day
- View includes all data available as of view_date
- Fiscal period derived from filing type and date
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd

from .schemas import (
    ReasonerInput,
    RiskFactorSummary,
    GuidanceSummary,
    EarningsSurprise,
)

logger = logging.getLogger(__name__)


def normalize_fiscal_period(fiscal_period: str) -> str:
    """
    Normalize fiscal period string to consistent format.

    Input formats: "FY 2023", "Q1 2024", "FY2023", "Q1_2024"
    Output format: "FY2023", "Q1_2024"
    """
    if not fiscal_period:
        return "Unknown"

    # Remove extra spaces and normalize
    fp = fiscal_period.strip().upper()

    # Handle "FY 2023" -> "FY2023"
    if fp.startswith("FY"):
        fp = fp.replace("FY ", "FY").replace("FY", "FY")
        return fp

    # Handle "Q1 2024" -> "Q1_2024"
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if fp.startswith(q):
            # Replace space with underscore
            fp = fp.replace(f"{q} ", f"{q}_").replace(f"{q}", f"{q}")
            if "_" not in fp:
                # Add underscore if missing
                fp = fp[:2] + "_" + fp[2:]
            return fp

    return fiscal_period


class InputBuilder:
    """
    Build ReasonerInput objects for view generation.

    Loads and caches data from:
    - data/extracted/{ticker}/*.json
    - data/raw/{ticker}/income_statement.json
    - data/raw/{ticker}/analyst_estimates.json
    - data/raw/{ticker}/splits.json
    - data/raw/{ticker}/prices.csv
    - data/constituents.csv
    """

    def __init__(
        self,
        extracted_dir: Path,
        raw_dir: Path,
        constituents_path: Path,
    ):
        """
        Initialize InputBuilder.

        Args:
            extracted_dir: Path to data/extracted/
            raw_dir: Path to data/raw/
            constituents_path: Path to constituents.csv
        """
        self.extracted_dir = Path(extracted_dir)
        self.raw_dir = Path(raw_dir)

        # Load constituents for sector info
        self.constituents = pd.read_csv(constituents_path)
        self.sector_map = dict(zip(
            self.constituents['Symbol'],
            self.constituents['GICS Sector']
        ))

        # Cache for loaded data
        self._extracted_cache: Dict[str, Dict[str, dict]] = {}
        self._income_cache: Dict[str, List[dict]] = {}
        self._estimates_cache: Dict[str, List[dict]] = {}
        self._splits_cache: Dict[str, List[dict]] = {}
        self._prices_cache: Dict[str, pd.DataFrame] = {}

    def _load_extracted_facts(self, ticker: str) -> Dict[str, dict]:
        """Load all extracted facts for a ticker, keyed by filing_date."""
        if ticker in self._extracted_cache:
            return self._extracted_cache[ticker]

        facts_by_date = {}
        ticker_dir = self.extracted_dir / ticker

        if not ticker_dir.exists():
            logger.warning(f"No extracted data for {ticker}")
            return facts_by_date

        for json_file in ticker_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                filing_date = data.get('filing_date')
                filing_type = data.get('filing_type')
                if filing_date and filing_type:
                    key = f"{filing_type}_{filing_date}"
                    facts_by_date[key] = data
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        self._extracted_cache[ticker] = facts_by_date
        return facts_by_date

    def _load_income_statement(self, ticker: str) -> List[dict]:
        """Load income statement quarterly data."""
        if ticker in self._income_cache:
            return self._income_cache[ticker]

        income_path = self.raw_dir / ticker / "income_statement.json"
        if not income_path.exists():
            logger.warning(f"No income statement for {ticker}")
            return []

        try:
            with open(income_path, 'r') as f:
                data = json.load(f)
            quarterly = data.get('quarterly', [])
            self._income_cache[ticker] = quarterly
            return quarterly
        except Exception as e:
            logger.error(f"Error loading income statement for {ticker}: {e}")
            return []

    def _load_analyst_estimates(self, ticker: str) -> List[dict]:
        """Load analyst estimates data."""
        if ticker in self._estimates_cache:
            return self._estimates_cache[ticker]

        estimates_path = self.raw_dir / ticker / "analyst_estimates.json"
        if not estimates_path.exists():
            logger.warning(f"No analyst estimates for {ticker}")
            return []

        try:
            with open(estimates_path, 'r') as f:
                estimates = json.load(f)
            self._estimates_cache[ticker] = estimates
            return estimates
        except Exception as e:
            logger.error(f"Error loading analyst estimates for {ticker}: {e}")
            return []

    def _load_splits(self, ticker: str) -> List[dict]:
        """Load stock splits history."""
        if ticker in self._splits_cache:
            return self._splits_cache[ticker]

        splits_path = self.raw_dir / ticker / "splits.json"
        if not splits_path.exists():
            logger.debug(f"No splits data for {ticker}")
            return []

        try:
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            # Sort by date ascending for cumulative factor calculation
            splits.sort(key=lambda x: x.get('date', ''))
            self._splits_cache[ticker] = splits
            return splits
        except Exception as e:
            logger.error(f"Error loading splits for {ticker}: {e}")
            return []

    def _get_cumulative_split_factor(
        self,
        ticker: str,
        from_date: date,
        to_date: Optional[date] = None,
    ) -> float:
        """
        Calculate cumulative split factor between two dates.

        Used to adjust analyst estimates (which are NOT split-adjusted)
        to match income statement EPS (which IS split-adjusted).

        Args:
            ticker: Stock ticker
            from_date: Start date (e.g., estimate period end date)
            to_date: End date (default: today)

        Returns:
            Cumulative factor (e.g., 3.0 for a 3:1 split)
            Divide the original estimate by this factor to get adjusted value.
        """
        if to_date is None:
            to_date = date.today()

        splits = self._load_splits(ticker)
        if not splits:
            return 1.0

        cumulative_factor = 1.0

        for split in splits:
            split_date_str = split.get('date')
            if not split_date_str:
                continue

            try:
                split_date = datetime.strptime(split_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            # Only count splits that happened AFTER from_date and ON/BEFORE to_date
            if from_date < split_date <= to_date:
                numerator = split.get('numerator', 1)
                denominator = split.get('denominator', 1)
                if denominator != 0:
                    factor = numerator / denominator
                    cumulative_factor *= factor
                    logger.debug(
                        f"{ticker}: Split {numerator}:{denominator} on {split_date_str}, "
                        f"cumulative factor now {cumulative_factor:.4f}"
                    )

        return cumulative_factor

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        """Load price data as DataFrame."""
        if ticker in self._prices_cache:
            return self._prices_cache[ticker]

        prices_path = self.raw_dir / ticker / "prices.csv"
        if not prices_path.exists():
            logger.warning(f"No prices for {ticker}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(prices_path, parse_dates=['date'])
            df = df.sort_values('date')
            df['return'] = df['close'].pct_change()
            self._prices_cache[ticker] = df
            return df
        except Exception as e:
            logger.error(f"Error loading prices for {ticker}: {e}")
            return pd.DataFrame()

    def _get_latest_filing(
        self,
        facts: Dict[str, dict],
        filing_type: str,
        before_date: date,
    ) -> Optional[dict]:
        """
        Get the latest filing of given type before a date.

        Args:
            facts: Dict of extracted facts keyed by "{type}_{date}"
            filing_type: "10-K" or "10-Q"
            before_date: Must have filing_date <= this date

        Returns:
            Most recent filing dict or None
        """
        candidates = []

        for key, data in facts.items():
            if not key.startswith(filing_type):
                continue

            filing_date_str = data.get('filing_date')
            if not filing_date_str:
                continue

            try:
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
                if filing_date <= before_date:
                    candidates.append((filing_date, data))
            except ValueError:
                continue

        if not candidates:
            return None

        # Return most recent
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _find_matching_estimate(
        self,
        estimates: List[dict],
        target_date: str,
    ) -> Optional[dict]:
        """
        Find estimate matching the target period date.

        Handles date mismatches (e.g., 2023-12-31 vs 2023-12-30)
        by finding estimate within 5 days of target.
        """
        if not target_date:
            return None

        try:
            target = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return None

        for est in estimates:
            est_date_str = est.get('date')
            if not est_date_str:
                continue
            try:
                est_date = datetime.strptime(est_date_str, "%Y-%m-%d").date()
                # Allow up to 5 days difference
                if abs((target - est_date).days) <= 5:
                    return est
            except ValueError:
                continue

        return None

    def _calculate_earnings_surprise(
        self,
        ticker: str,
        view_date: date,
    ) -> Optional[EarningsSurprise]:
        """
        Calculate earnings surprise for the most recent period
        where filingDate < view_date.

        Point-in-time safe: only uses data that was publicly available.

        Note: FMP provides split-adjusted data for BOTH income_statement
        and analyst_estimates, so no manual adjustment is needed.
        """
        income = self._load_income_statement(ticker)
        estimates = self._load_analyst_estimates(ticker)

        if not income or not estimates:
            return None

        # Find most recent quarter where filing was before view_date
        for period in income:
            filing_date_str = period.get('filingDate')
            if not filing_date_str:
                continue

            try:
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            # Point-in-time check
            if filing_date >= view_date:
                continue

            # Get period info
            period_date = period.get('date')
            fiscal_year = period.get('fiscalYear', '')
            quarter = period.get('period', '')
            period_label = f"{quarter} {fiscal_year}"

            # Get actuals (split-adjusted by FMP)
            eps_actual = period.get('epsDiluted') or period.get('eps')
            revenue_actual = period.get('revenue')

            # Get estimates for this period (fuzzy date match)
            # FMP provides split-adjusted estimates
            estimate = self._find_matching_estimate(estimates, period_date)
            if not estimate:
                continue

            eps_estimate = estimate.get('epsAvg')
            eps_high = estimate.get('epsHigh')
            eps_low = estimate.get('epsLow')
            revenue_estimate = estimate.get('revenueAvg')
            num_analysts = estimate.get('numAnalystsEps')

            # Calculate surprises
            eps_surprise_pct = None
            if eps_actual is not None and eps_estimate and eps_estimate != 0:
                eps_surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

            revenue_surprise_pct = None
            if revenue_actual and revenue_estimate and revenue_estimate != 0:
                revenue_surprise_pct = ((revenue_actual - revenue_estimate) / abs(revenue_estimate)) * 100

            # Nullify extreme EPS surprise due to GAAP vs Adjusted mismatch.
            # FMP income_statement reports GAAP EPS (includes one-time items
            # like goodwill impairments, spinoff gains, restructuring charges),
            # while analyst_estimates contains adjusted/operating EPS.
            # When |surprise| > 200%, it's almost always a GAAP distortion,
            # not a real earnings beat/miss. We nullify ALL EPS fields so
            # the LLM prompt shows "N/A" and doesn't compute its own
            # misleading surprise from raw GAAP vs adjusted values.
            if eps_surprise_pct is not None and abs(eps_surprise_pct) > 200:
                logger.info(
                    f"{ticker}: Nullifying extreme eps_surprise={eps_surprise_pct:.1f}% "
                    f"(actual={eps_actual}, estimate={eps_estimate}) — likely GAAP/adjusted mismatch"
                )
                eps_surprise_pct = None
                eps_actual = None
                eps_estimate = None
                eps_high = None
                eps_low = None
                revenue_surprise_pct = None

            # Calculate dispersion
            dispersion = None
            if eps_high is not None and eps_low is not None and eps_estimate and eps_estimate != 0:
                dispersion = (eps_high - eps_low) / abs(eps_estimate)

            # Convert revenue to billions
            revenue_actual_b = revenue_actual / 1e9 if revenue_actual else None
            revenue_estimate_b = revenue_estimate / 1e9 if revenue_estimate else None

            return EarningsSurprise(
                period=period_label,
                filing_date=filing_date_str,
                eps_actual=eps_actual,
                eps_estimate=eps_estimate,
                eps_surprise_pct=eps_surprise_pct,
                revenue_actual=revenue_actual_b,
                revenue_estimate=revenue_estimate_b,
                revenue_surprise_pct=revenue_surprise_pct,
                num_analysts=num_analysts,
                estimate_dispersion=dispersion,
            )

        return None

    def _calculate_price_metrics(
        self,
        ticker: str,
        as_of_date: date,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate price returns and volatility as of a date.

        Returns:
            (return_1m, return_3m, volatility_60d) in percentages
        """
        prices = self._load_prices(ticker)

        if prices.empty:
            return None, None, None

        # Filter to data available as of date
        mask = prices['date'].dt.date <= as_of_date
        available = prices[mask]

        if len(available) < 63:  # Need ~3 months of data
            return None, None, None

        # Current price (last available)
        current_price = available['close'].iloc[-1]

        # 1-month return (21 trading days)
        return_1m = None
        if len(available) >= 22:
            price_21d_ago = available['close'].iloc[-22]
            return_1m = ((current_price - price_21d_ago) / price_21d_ago) * 100

        # 3-month return (63 trading days)
        return_3m = None
        if len(available) >= 64:
            price_63d_ago = available['close'].iloc[-64]
            return_3m = ((current_price - price_63d_ago) / price_63d_ago) * 100

        # 60-day volatility (annualized)
        volatility_60d = None
        if len(available) >= 60:
            returns_60d = available['return'].iloc[-60:]
            daily_vol = returns_60d.std()
            volatility_60d = daily_vol * (252 ** 0.5) * 100  # Annualized %

        return return_1m, return_3m, volatility_60d

    def build_input_for_filing(
        self,
        ticker: str,
        filing_data: dict,
    ) -> Optional[ReasonerInput]:
        """
        Build ReasonerInput for a specific filing.

        Args:
            ticker: Stock ticker
            filing_data: Extracted facts from the filing

        Returns:
            ReasonerInput or None if insufficient data
        """
        filing_type = filing_data.get('filing_type')
        filing_date_str = filing_data.get('filing_date')

        if not filing_type or not filing_date_str:
            return None

        # View date = filing_date + 1 day
        try:
            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
            view_date = filing_date + timedelta(days=1)
        except ValueError:
            return None

        # Fiscal period - use from extracted data (more accurate for non-standard fiscal years)
        fiscal_period_raw = filing_data.get('fiscal_period', '')
        fiscal_period = normalize_fiscal_period(fiscal_period_raw)

        # Get sector
        sector = self.sector_map.get(ticker, "Unknown")

        # Load all extracted facts
        facts = self._load_extracted_facts(ticker)

        # Get latest 10-K and 10-Q as of view_date
        latest_10k = self._get_latest_filing(facts, "10-K", view_date)
        latest_10q = self._get_latest_filing(facts, "10-Q", view_date)

        # Extract risk factors from 10-K
        annual_risk_factors = []
        annual_risk_summary = None
        annual_key_events = []

        if latest_10k:
            for rf in latest_10k.get('risk_factors', []):
                annual_risk_factors.append(RiskFactorSummary(
                    category=rf.get('category', 'other'),
                    severity=rf.get('severity', 'medium'),
                    description=rf.get('description', '')[:200],
                ))
            annual_risk_summary = latest_10k.get('risk_summary')
            annual_key_events = [
                e.get('description', '')[:100]
                for e in latest_10k.get('key_events', [])
            ]

        # Extract quarterly data from 10-Q (or 10-K if it's the trigger)
        quarterly_sentiment = None
        quarterly_sentiment_rationale = None
        quarterly_guidance = None
        quarterly_key_events = []

        # Use the trigger filing for sentiment/guidance
        trigger_filing = filing_data
        quarterly_sentiment = trigger_filing.get('sentiment')
        quarterly_sentiment_rationale = trigger_filing.get('sentiment_rationale')

        guidance_data = trigger_filing.get('guidance', {})
        if guidance_data:
            quarterly_guidance = GuidanceSummary(
                direction=guidance_data.get('direction'),
                revenue_outlook=guidance_data.get('revenue_outlook'),
                earnings_outlook=guidance_data.get('earnings_outlook'),
                statement=guidance_data.get('statement'),
            )

        quarterly_key_events = [
            e.get('description', '')[:100]
            for e in trigger_filing.get('key_events', [])
        ]

        # Calculate earnings surprise (point-in-time safe)
        earnings_surprise = self._calculate_earnings_surprise(ticker, view_date)

        # Calculate price metrics
        return_1m, return_3m, volatility_60d = self._calculate_price_metrics(
            ticker, view_date
        )

        return ReasonerInput(
            ticker=ticker,
            view_date=view_date,
            fiscal_period=fiscal_period,
            sector=sector,
            trigger_filing_type=filing_type,
            trigger_filing_date=filing_date_str,
            latest_10k_date=latest_10k.get('filing_date') if latest_10k else None,
            annual_risk_factors=annual_risk_factors,
            annual_risk_summary=annual_risk_summary,
            annual_key_events=annual_key_events,
            latest_10q_date=latest_10q.get('filing_date') if latest_10q else None,
            quarterly_sentiment=quarterly_sentiment,
            quarterly_sentiment_rationale=quarterly_sentiment_rationale,
            quarterly_guidance=quarterly_guidance,
            quarterly_key_events=quarterly_key_events,
            earnings_surprise=earnings_surprise,
            price_return_1m=return_1m,
            price_return_3m=return_3m,
            price_volatility_60d=volatility_60d,
        )

    def get_all_filings(
        self,
        ticker: str,
        start_year: int = 2006,
        end_year: int = 2025,
    ) -> List[dict]:
        """
        Get all filings for a ticker within the study period.

        Args:
            ticker: Stock ticker
            start_year: First year (inclusive)
            end_year: Last year (inclusive)

        Returns:
            List of filing dicts sorted by date
        """
        facts = self._load_extracted_facts(ticker)
        filings = []

        for key, data in facts.items():
            filing_date_str = data.get('filing_date')
            if not filing_date_str:
                continue

            try:
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
                if start_year <= filing_date.year <= end_year:
                    filings.append(data)
            except ValueError:
                continue

        # Sort by filing date
        filings.sort(key=lambda x: x.get('filing_date', ''))
        return filings

    def build_all_inputs(
        self,
        tickers: List[str],
        start_year: int = 2006,
        end_year: int = 2025,
    ) -> List[ReasonerInput]:
        """
        Build ReasonerInput for all filings in the study period.

        Args:
            tickers: List of tickers
            start_year: First year (inclusive)
            end_year: Last year (inclusive)

        Returns:
            List of ReasonerInput objects
        """
        inputs = []
        total_filings = 0

        for ticker in tickers:
            filings = self.get_all_filings(ticker, start_year, end_year)
            total_filings += len(filings)

            for filing_data in filings:
                input_data = self.build_input_for_filing(ticker, filing_data)
                if input_data:
                    inputs.append(input_data)

            if len(inputs) % 100 == 0 and len(inputs) > 0:
                logger.info(f"Built {len(inputs)} inputs so far...")

        logger.info(f"Built {len(inputs)} inputs from {total_filings} filings")
        return inputs
