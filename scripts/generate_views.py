#!/usr/bin/env python3
"""
Generate investment views using LLM Reasoner.

Usage:
    python scripts/generate_views.py --test           # Test on 10 views
    python scripts/generate_views.py --ticker AMZN    # Single ticker
    python scripts/generate_views.py --ticker AMZN --year 2024  # Specific year
    python scripts/generate_views.py --resume         # Full run with resume
    python scripts/generate_views.py --dry-run        # Show what would be generated

Output:
    data/views/{TICKER}/view_{VIEW_DATE}_{FISCAL_PERIOD}.json
    data/views/views_summary.csv
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reasoning import (
    InputBuilder,
    ViewReasoner,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    with open('config/universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)

    return config, universe


def get_all_tickers(universe: dict) -> list:
    """Get all tickers from universe config."""
    u = universe['universe']
    if 'tickers' in u:
        return u['tickers']
    return u.get('core', []) + u.get('holdout', [])


def count_existing_views(output_dir: Path, tickers: list) -> dict:
    """Count already generated views per ticker."""
    counts = {}
    for ticker in tickers:
        ticker_dir = output_dir / ticker
        if ticker_dir.exists():
            counts[ticker] = len(list(ticker_dir.glob("view_*.json")))
        else:
            counts[ticker] = 0
    return counts


def main():
    parser = argparse.ArgumentParser(description='Generate investment views using LLM')
    parser.add_argument('--test', action='store_true', help='Test mode (10 views)')
    parser.add_argument('--ticker', type=str, help='Process single ticker')
    parser.add_argument('--year', type=int, help='Process specific year only')
    parser.add_argument('--resume', action='store_true', help='Skip existing views')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated')
    parser.add_argument('--summary-only', action='store_true', help='Only generate summary CSV')
    parser.add_argument('--fix-extreme', action='store_true',
                        help='Regenerate views with |eps_surprise| > 200%% (GAAP/adjusted mismatch fix)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--extracted-dir', type=str, default='data/extracted',
                        help='Input dir with extracted facts (use data/v2/extracted_v2 for v2)')
    parser.add_argument('--output-dir', type=str, default='data/views',
                        help='Output dir for views (use data/v2/views_v2 for v2)')
    parser.add_argument('--model', type=str, default=None,
                        help='Override reasoning model (OpenRouter slug); default: config.yaml')
    args = parser.parse_args()

    # Load config
    config, universe = load_config()
    if args.model:  # CLI overrides config for the reasoning model
        config['openai']['model'] = args.model

    # Paths (CLI overrides for v2: --extracted-dir data/v2/extracted_v2 --output-dir data/v2/views_v2)
    base_dir = Path('.')
    extracted_dir = base_dir / args.extracted_dir
    raw_dir = base_dir / 'data' / 'raw'
    output_dir = base_dir / args.output_dir
    constituents_path = base_dir / 'data' / 'constituents.csv'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = get_all_tickers(universe)

    logger.info(f"Processing {len(tickers)} tickers")

    # Year range
    start_year = args.year if args.year else 2006
    end_year = args.year if args.year else 2025

    # Test mode: limit to 2 tickers
    if args.test:
        tickers = tickers[:2]
        logger.info(f"Test mode: {len(tickers)} tickers")

    # Summary only mode
    if args.summary_only:
        api_key = config['openai']['api_key']
        model = config['openai']['model']
        reasoner = ViewReasoner(
            api_key=api_key,
            model=model,
            base_url=config['openai']['base_url'],
        )
        csv_path = reasoner.save_summary_csv(output_dir)
        logger.info(f"Summary saved to {csv_path}")
        return

    # Fix extreme mode: find and delete views with |eps_surprise| > 200%
    if args.fix_extreme:
        import csv
        import json as json_mod

        csv_path = output_dir / "views_summary.csv"
        if not csv_path.exists():
            logger.error("views_summary.csv not found. Run --summary-only first.")
            return

        # Read summary and find extreme views
        extreme_views = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                eps = row.get('eps_surprise')
                if eps and eps != '':
                    try:
                        eps_val = float(eps)
                        if abs(eps_val) > 200:
                            extreme_views.append(row)
                    except ValueError:
                        pass

        logger.info(f"Found {len(extreme_views)} views with |eps_surprise| > 200%")

        if not extreme_views:
            logger.info("No extreme views to fix.")
            return

        # Delete the extreme view files so they get regenerated
        deleted = 0
        # Track which tickers need regeneration
        tickers_to_fix = set()
        extreme_keys = set()  # (ticker, view_date, fiscal_period)

        for row in extreme_views:
            ticker = row['ticker']
            view_date = row['view_date']
            fiscal_period = row['fiscal_period']

            # Find and delete the view file
            ticker_dir = output_dir / ticker
            pattern = f"view_{view_date}_{fiscal_period}.json"
            view_file = ticker_dir / pattern

            if view_file.exists():
                view_file.unlink()
                deleted += 1
                tickers_to_fix.add(ticker)
                extreme_keys.add((ticker, view_date, fiscal_period))

        logger.info(f"Deleted {deleted} extreme view files across {len(tickers_to_fix)} tickers")
        logger.info(f"Tickers to regenerate: {sorted(tickers_to_fix)}")

        # Override tickers list to only process affected ones
        tickers = sorted(tickers_to_fix)
        # Set resume so only deleted files get regenerated
        args.resume = True

    # Initialize InputBuilder
    logger.info("Initializing InputBuilder...")
    input_builder = InputBuilder(
        extracted_dir=extracted_dir,
        raw_dir=raw_dir,
        constituents_path=constituents_path,
    )

    # Dry run: show what would be generated
    if args.dry_run:
        existing = count_existing_views(output_dir, tickers)
        total_existing = sum(existing.values())

        # Count total filings
        total_filings = 0
        for ticker in tickers:
            filings = input_builder.get_all_filings(ticker, start_year, end_year)
            total_filings += len(filings)

        print(f"\nDry run summary:")
        print(f"  Tickers: {len(tickers)}")
        print(f"  Year range: {start_year}-{end_year}")
        print(f"  Total filings: {total_filings}")
        print(f"  Already generated: {total_existing}")
        print(f"  Would generate: ~{total_filings - total_existing}")

        if args.resume:
            print(f"\n  --resume flag: Will skip existing files")
        else:
            print(f"\n  Without --resume: Will overwrite existing files")

        print(f"\nTickers with existing views:")
        for ticker in sorted(existing.keys()):
            if existing[ticker] > 0:
                print(f"  {ticker}: {existing[ticker]} views")

        return

    # Config for reasoner
    openai_config = config['openai']
    reasoner_config = config.get('reasoner', {})

    # Counter for progress
    progress_lock = Lock()
    progress = {'done': 0, 'total': len(tickers), 'views': 0}

    def process_ticker(ticker: str) -> list:
        """Process a single ticker - runs in thread."""
        # Each thread gets its own reasoner (own rate limiter)
        reasoner = ViewReasoner(
            api_key=openai_config['api_key'],
            model=openai_config['model'],
            base_url=openai_config['base_url'],
            temperature=reasoner_config.get('temperature', 0.3),
            max_tokens=reasoner_config.get('max_tokens', 500),
            requests_per_minute=30,  # Per-worker rate limit
        )

        # Get filings for this ticker
        filings = input_builder.get_all_filings(ticker, start_year, end_year)
        if not filings:
            logger.warning(f"[{ticker}] No filings found")
            return []

        # Build inputs for this ticker
        ticker_inputs = []
        for filing in filings:
            inp = input_builder.build_input_for_filing(ticker, filing)
            if inp:
                ticker_inputs.append(inp)

        if not ticker_inputs:
            logger.warning(f"[{ticker}] No valid inputs")
            return []

        logger.info(f"[{ticker}] Processing {len(ticker_inputs)} filings...")

        # Generate views
        ticker_results = reasoner.generate_views_batch(
            inputs=ticker_inputs,
            output_dir=output_dir,
            resume=args.resume,
        )

        # Update progress
        with progress_lock:
            progress['done'] += 1
            progress['views'] += len(ticker_results)
            pct = progress['done'] / progress['total'] * 100
            logger.info(f"[{ticker}] Done. Progress: {progress['done']}/{progress['total']} ({pct:.0f}%), total views: {progress['views']}")

        return ticker_results

    # Estimate total filings
    total_filings = 0
    for ticker in tickers:
        filings = input_builder.get_all_filings(ticker, start_year, end_year)
        total_filings += len(filings)
    logger.info(f"Total filings to process: {total_filings}")

    # Process tickers in parallel
    num_workers = args.workers
    logger.info(f"Starting generation with {num_workers} parallel workers...")

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker_results = future.result()
                results.extend(ticker_results)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")

    logger.info(f"Generated {len(results)} views total")

    # Generate summary CSV (create a simple reasoner just for this)
    summary_reasoner = ViewReasoner(
        api_key=openai_config['api_key'],
        model=openai_config['model'],
        base_url=openai_config['base_url'],
    )
    csv_path = summary_reasoner.save_summary_csv(output_dir)
    logger.info(f"Summary saved to {csv_path}")

    # Print statistics
    if results:
        view_bps_values = [r.view.view_bps for r in results]
        confidence_counts = {}
        for r in results:
            conf = r.view.confidence.value
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        print(f"\n=== Generation Complete ===")
        print(f"Total views generated: {len(results)}")
        print(f"View bps range: [{min(view_bps_values)}, {max(view_bps_values)}]")
        print(f"View bps mean: {sum(view_bps_values) / len(view_bps_values):.1f}")
        print(f"Confidence distribution: {confidence_counts}")


if __name__ == '__main__':
    main()
