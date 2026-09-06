import os
import sys
import yaml
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add src to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fmp_client import FMPClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Number of parallel workers
MAX_WORKERS = 10


def load_config():
    """Load universe and main config."""
    with open('config/universe.yaml', 'r') as f:
        universe_config = yaml.safe_load(f)

    with open('config/config.yaml', 'r') as f:
        main_config = yaml.safe_load(f)

    return universe_config, main_config


def save_json(data, path: Path):
    """Helper to save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_csv(df: pd.DataFrame, path: Path):
    """Helper to save DataFrame as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def add_year(dt: str, years: int) -> str:
    """Add years to a date string."""
    d = datetime.strptime(dt, "%Y-%m-%d")
    try:
        return d.replace(year=d.year + years).strftime("%Y-%m-%d")
    except ValueError:
        # Handle Feb 29 -> Feb 28
        return (d.replace(year=d.year + years, day=28)).strftime("%Y-%m-%d")


def filter_by_date_range(records: list, start_date: str, end_date: str, date_field: str = 'date') -> list:
    """
    Filter records by date range.

    Args:
        records: List of dicts with date field
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        date_field: Name of date field in records

    Returns:
        Filtered list of records
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    filtered = []
    for record in records:
        date_str = record.get(date_field)
        if not date_str:
            continue
        try:
            # Handle both YYYY-MM-DD and YYYY-MM-DD HH:MM:SS formats
            record_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            if start_dt <= record_dt <= end_dt:
                filtered.append(record)
        except ValueError:
            # Skip records with invalid dates
            continue

    return filtered


def download_ticker_data(ticker: str, client: FMPClient, start_date: str, end_date: str, raw_data_dir: Path) -> dict:
    """
    Download all data for a single ticker.
    Returns a dict with status info.
    """
    print(f"[{ticker}] Starting download...", flush=True)
    result = {"ticker": ticker, "success": True, "errors": []}

    ticker_dir = raw_data_dir / ticker
    ticker_dir.mkdir(exist_ok=True)

    try:
        # # 1. Historical Prices
        # print(f"[{ticker}] Fetching prices...", flush=True)
        # prices = client.get_price_history(ticker, start_date, end_date)
        # if not prices.empty:
        #     save_csv(prices, ticker_dir / "prices.csv")
        #     print(f"[{ticker}] Prices saved: {len(prices)} rows", flush=True)
        # else:
        #     result["errors"].append("No price data")
        #     print(f"[{ticker}] WARNING: No price data!", flush=True)
        #
        # # 2. SEC Filings (10-K and 10-Q)
        # print(f"[{ticker}] Fetching SEC filings...", flush=True)
        # filings_10k = client.get_sec_filings(ticker, filing_type="10-K", limit=1000, start_date=start_date, end_date=end_date)
        # filings_10q = client.get_sec_filings(ticker, filing_type="10-Q", limit=1000, start_date=start_date, end_date=end_date)
        #
        # all_filings = filings_10k + filings_10q
        # save_json(all_filings, ticker_dir / "filings.json")
        # print(f"[{ticker}] Filings saved: {len(filings_10k)} 10-K, {len(filings_10q)} 10-Q", flush=True)
        #
        # # 3. Analyst Estimates
        # print(f"[{ticker}] Fetching analyst estimates...", flush=True)
        # estimates = client.get_analyst_estimates(ticker, limit=1000, period="quarter")
        # # Filter by date range
        # filtered_estimates = [
        #     e for e in estimates
        #     if e.get('date') and start_date <= e.get('date', '')[:10] <= end_date
        # ]
        # save_json(filtered_estimates, ticker_dir / "analyst_estimates.json")
        # print(f"[{ticker}] Analyst estimates saved: {len(filtered_estimates)} records", flush=True)

        # 4. Income Statement (quarterly + annual)
        print(f"[{ticker}] Fetching income statements...", flush=True)
        income_quarterly = filter_by_date_range(
            client.get_income_statement(ticker, period="quarter", limit=10000),
            start_date, end_date
        )
        income_annual = filter_by_date_range(
            client.get_income_statement(ticker, period="annual", limit=10000),
            start_date, end_date
        )
        save_json({"quarterly": income_quarterly, "annual": income_annual}, ticker_dir / "income_statement.json")
        print(f"[{ticker}] Income: {len(income_quarterly)}q + {len(income_annual)}a", flush=True)

        # 5. Balance Sheet (quarterly + annual)
        print(f"[{ticker}] Fetching balance sheets...", flush=True)
        balance_quarterly = filter_by_date_range(
            client.get_balance_sheet(ticker, period="quarter", limit=10000),
            start_date, end_date
        )
        balance_annual = filter_by_date_range(
            client.get_balance_sheet(ticker, period="annual", limit=10000),
            start_date, end_date
        )
        save_json({"quarterly": balance_quarterly, "annual": balance_annual}, ticker_dir / "balance_sheet.json")
        print(f"[{ticker}] Balance: {len(balance_quarterly)}q + {len(balance_annual)}a", flush=True)

        # 6. Cash Flow Statement (quarterly + annual)
        print(f"[{ticker}] Fetching cash flow statements...", flush=True)
        cashflow_quarterly = filter_by_date_range(
            client.get_cash_flow_statement(ticker, period="quarter", limit=10000),
            start_date, end_date
        )
        cashflow_annual = filter_by_date_range(
            client.get_cash_flow_statement(ticker, period="annual", limit=10000),
            start_date, end_date
        )
        save_json({"quarterly": cashflow_quarterly, "annual": cashflow_annual}, ticker_dir / "cash_flow.json")
        print(f"[{ticker}] Cash flow: {len(cashflow_quarterly)}q + {len(cashflow_annual)}a", flush=True)

        # 7. Stock Splits
        print(f"[{ticker}] Fetching stock splits...", flush=True)
        splits = client.get_stock_splits(ticker)
        save_json(splits, ticker_dir / "splits.json")
        print(f"[{ticker}] Splits: {len(splits)} records", flush=True)

    except Exception as e:
        result["success"] = False
        result["errors"].append(str(e))
        print(f"[{ticker}] ERROR: {e}", flush=True)

    print(f"[{ticker}] Done!", flush=True)
    return result


def main():
    logger.info("Starting Raw Data Download (Parallel)...")

    # Load configs
    universe, config = load_config()
    all_tickers = universe['universe']['core'] + universe['universe'].get('holdout', [])

    # Init client
    fmp_api_key = config.get('fmp', {}).get('api_key')
    base_url = config.get('fmp', {}).get('base_url')
    client = FMPClient(api_key=fmp_api_key, base_url=base_url)

    # Read main study period from config, add 1 year buffer before and after
    study_start = config.get('backtest', {}).get('start_date', "2015-01-01")
    study_end = config.get('backtest', {}).get('end_date', "2025-12-31")

    start_date = add_year(study_start, -1)
    end_date = add_year(study_end, 1)

    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target Universe: {len(all_tickers)} stocks")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Using {MAX_WORKERS} parallel workers")

    # Track results
    successful = 0
    failed = 0
    failed_tickers = []

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(download_ticker_data, ticker, client, start_date, end_date, raw_data_dir): ticker
            for ticker in all_tickers
        }

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            ticker = futures[future]
            try:
                result = future.result()
                if result["success"]:
                    successful += 1
                    if result["errors"]:
                        logger.warning(f"[{ticker}] Completed with warnings: {result['errors'][:3]}...")
                else:
                    failed += 1
                    failed_tickers.append(ticker)
                    logger.error(f"[{ticker}] Failed: {result['errors']}")
            except Exception as e:
                failed += 1
                failed_tickers.append(ticker)
                logger.error(f"[{ticker}] Exception: {e}")

    # Summary
    logger.info("=" * 50)
    logger.info("Download Complete!")
    logger.info(f"Successful: {successful}/{len(all_tickers)}")
    logger.info(f"Failed: {failed}/{len(all_tickers)}")
    if failed_tickers:
        logger.info(f"Failed tickers: {failed_tickers}")
    logger.info("Check data/raw/ directory.")


if __name__ == "__main__":
    main()