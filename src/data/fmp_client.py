import os
import sys
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry, RateLimitException
import diskcache
import pandas as pd
from typing import Optional, Dict, List, Any
import logging

# Configure logging with flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class FMPClient:
    """
    Client for Financial Modeling Prep (FMP) API with caching and rate limiting.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FMPClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: Optional[str] = None, base_url: str = None, cache_dir: str = ".cache"):
        if hasattr(self, "initialized"):
            return
            
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found. Please provide it or set env var.")
        
        self.base_url = base_url
        if not self.base_url:
            raise ValueError("base_url must be provided via config")
            
        self.cache = diskcache.Cache(cache_dir)
        self.initialized = True

    @sleep_and_retry
    @limits(calls=700, period=60) # Increased to 700 as per user config
    def _request(self, endpoint: str, params: Dict[str, Any] = {}) -> Any:
        full_url = f"{self.base_url}{endpoint}"
        
        # Merge API key into params
        request_params = params.copy()
        request_params["apikey"] = self.api_key
        
        # Create a cache key based on URL and sorted params
        # cache_key = f"{full_url}:{sorted(request_params.items())}"
        #
        # if cache_key in self.cache:
        #     logger.info(f"[{endpoint}] CACHE HIT")
        #     sys.stdout.flush()
        #     return self.cache[cache_key]

        try:
            sys.stdout.flush()

            response = requests.get(full_url, params=request_params)

            # Log status code and response body
            if response.text == '[]' or response.text == '{}' or "No Data for this symbol or invalid API call." in response.text:
                logger.info(f"[{endpoint}] Status: {response.status_code}, Size: {len(response.text)} bytes, URL: {response.url}")
            sys.stdout.flush()

            if response.status_code == 429:
                raise RateLimitException("", period_remaining=60)

            response.raise_for_status()
            data = response.json()

            # Cache the result (expire after 24h by default)
            # self.cache.set(cache_key, data, expire=86400)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            # Log status code and response body on error
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            sys.stdout.flush()
            raise

    def get_price_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical daily prices using Stable API endpoint.
        Endpoint: /stable/historical-price-eod/full
        """
        endpoint = "/historical-price-eod/full"
        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date
        }
        
        data = self._request(endpoint, params)
        
        # Response structure may vary, handle both direct list and nested structure
        if not data:
            logger.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()
        
        # Handle if data is wrapped in a dict with 'historical' key
        if isinstance(data, dict) and 'historical' in data:
            data = data['historical']
        
        if not data:
            logger.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        return df

    def get_sec_filings(self, symbol: str, filing_type: str = "10-K", limit: int = 1000, start_date: str = None, end_date: str = None, page: int = 0) -> List[Dict]:
        """
        Get list of SEC filings using Stable API.
        Endpoint: /stable/sec-filings-search/symbol
        """
        endpoint = "/sec-filings-search/symbol"
        params = {
            "symbol": symbol,
            "limit": limit,
            "page": page
        }
        
        # Add date filters if provided
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        has_next_page = True
        data = []
        while has_next_page:
            page_data = self._request(endpoint, params)
            data.extend(page_data)
            params['page'] = params['page'] + 1
            has_next_page = bool(page_data)
        
        if not data:
            return []
        
        # Filter by formType manually since endpoint returns all types
        if isinstance(data, list):
            return [f for f in data if f.get('formType') == filing_type]
        
        return []
        
    def get_analyst_estimates(self, symbol: str, limit: int = 1000, period: str = "quarter", page: int = 0) -> List[Dict]:
        """
        Get analyst estimates.
        Endpoint: /stable/analyst-estimates
        """
        endpoint = "/analyst-estimates"
        params = {
            "symbol": symbol,
            "limit": limit,
            "period": period,
            "page": page
        }
        data = self._request(endpoint, params)
        return data if isinstance(data, list) else []
        
    def get_market_cap(self, symbol: str, start_date: str = None, end_date: str = None, limit: int = 1000) -> pd.DataFrame:
        """
        Get historical market capitalization.
        Endpoint: /stable/historical-market-capitalization
        """
        endpoint = "/historical-market-capitalization"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        # Add date filters if provided
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        
        data = self._request(endpoint, params)
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        return df
    
    def get_grades_historical(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """
        Get historical analyst grades.
        Endpoint: /stable/grades-historical
        """
        endpoint = "/grades-historical"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        data = self._request(endpoint, params)
        return data if isinstance(data, list) else []
    
    def get_financial_report_json(self, symbol: str, year: int, period: str) -> Dict:
        """
        Get structured financial report data from SEC filings.
        Endpoint: /stable/financial-reports-json

        Args:
            symbol: Stock ticker
            year: Year (e.g., 2022)
            period: Period - 'FY' for annual (10-K), 'Q1', 'Q2', 'Q3', 'Q4' for quarterly (10-Q)

        Returns:
            List of financial report data dictionaries
        """
        endpoint = "/financial-reports-json"
        params = {
            "symbol": symbol,
            "year": year,
            "period": period
        }
        data = self._request(endpoint, params)
        if "No Data for this symbol or invalid API call." in str(data):
            return {}
        return data

    def get_income_statement(self, symbol: str, period: str = "quarter", limit: int = 10000) -> List[Dict]:
        """
        Get income statement data.
        Endpoint: /stable/income-statement

        Args:
            symbol: Stock ticker
            period: 'quarter' or 'annual'
            limit: Maximum number of records

        Returns:
            List of income statement records
        """
        endpoint = "/income-statement"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
        data = self._request(endpoint, params)
        return data if isinstance(data, list) else []

    def get_balance_sheet(self, symbol: str, period: str = "quarter", limit: int = 10000) -> List[Dict]:
        """
        Get balance sheet statement data.
        Endpoint: /stable/balance-sheet-statement

        Args:
            symbol: Stock ticker
            period: 'quarter' or 'annual'
            limit: Maximum number of records

        Returns:
            List of balance sheet records
        """
        endpoint = "/balance-sheet-statement"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
        data = self._request(endpoint, params)
        return data if isinstance(data, list) else []

    def get_cash_flow_statement(self, symbol: str, period: str = "quarter", limit: int = 10000) -> List[Dict]:
        """
        Get cash flow statement data.
        Endpoint: /stable/cash-flow-statement

        Args:
            symbol: Stock ticker
            period: 'quarter' or 'annual'
            limit: Maximum number of records

        Returns:
            List of cash flow statement records
        """
        endpoint = "/cash-flow-statement"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
        data = self._request(endpoint, params)
        return data if isinstance(data, list) else []

    def get_stock_splits(self, symbol: str) -> List[Dict]:
        """
        Get stock split history.
        Endpoint: /stable/splits

        Args:
            symbol: Stock ticker

        Returns:
            List of stock split records with date, numerator, denominator
        """
        endpoint = "/splits"
        params = {
            "symbol": symbol
        }
        data = self._request(endpoint, params)
        return data if isinstance(data, list) else []
