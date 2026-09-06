import os
import sys
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
import logging

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_filing(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """
    Download a single SEC filing from URL.
    
    Args:
        url: URL to the SEC filing
        output_path: Path to save the filing
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    # SEC requires proper headers
    headers = {
        'Host': 'www.sec.gov',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:146.0) Gecko/20100101 Firefox/146.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Priority': 'u=0, i',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save the content
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
    
    return False

def main():
    logger.info("Starting SEC Filings Download...")
    
    data_dir = Path("data/raw")
    
    # Get all ticker directories
    ticker_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    total_filings = 0
    downloaded_filings = 0
    skipped_filings = 0
    failed_filings = 0
    
    for ticker_dir in sorted(ticker_dirs):
        ticker = ticker_dir.name
        filings_json_path = ticker_dir / "filings.json"
        
        if not filings_json_path.exists():
            logger.warning(f"[{ticker}] No filings.json found, skipping")
            continue
        
        # Load filings metadata
        with open(filings_json_path, 'r') as f:
            filings = json.load(f)
        
        logger.info(f"[{ticker}] Processing {len(filings)} filings...")
        
        # Create filings directory
        filings_dir = ticker_dir / "filings"
        filings_dir.mkdir(exist_ok=True)
        
        for filing in tqdm(filings, desc=f"{ticker} filings"):
            total_filings += 1
            
            # Create filename from filing info
            filing_date = filing.get('filingDate', '').split()[0]  # Get date part
            form_type = filing.get('formType', 'UNKNOWN')
            final_link = filing.get('finalLink', '')
            
            if not final_link:
                logger.warning(f"[{ticker}] No finalLink for {form_type} on {filing_date}")
                failed_filings += 1
                continue
            
            # Create output filename
            output_filename = f"{form_type}_{filing_date}.html"
            output_path = filings_dir / output_filename
            
            # Skip if already downloaded
            if output_path.exists():
                skipped_filings += 1
                continue
            
            # Download the filing
            logger.debug(f"[{ticker}] Downloading {form_type} from {filing_date}")
            success = download_filing(final_link, output_path)
            
            if success:
                downloaded_filings += 1
            else:
                failed_filings += 1
            
            # Be respectful to SEC servers - rate limit
            time.sleep(0.2)  # 5 requests per second max
    
    logger.info("=" * 80)
    logger.info("SEC Filings Download Complete!")
    logger.info(f"Total filings: {total_filings}")
    logger.info(f"Downloaded: {downloaded_filings}")
    logger.info(f"Skipped (already exist): {skipped_filings}")
    logger.info(f"Failed: {failed_filings}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
