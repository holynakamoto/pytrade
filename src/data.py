"""
Data fetching module for the trading system.
Supports multiple data sources: Yahoo Finance, Polygon, Alpha Vantage.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and caches market data from various sources."""

    def __init__(self, config: dict):
        """
        Initialize the data fetcher.

        Args:
            config: Configuration dictionary with data source settings
        """
        self.config = config
        self.primary_source = config['data_source']['primary']
        self.cache_days = config['data_source']['cache_days']
        self.polygon_api_key = os.getenv('POLYGON_API_KEY') or config['data_source'].get('polygon_api_key')
        self.alphavantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY') or config['data_source'].get('alphavantage_api_key')

        # Data cache
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Stock/asset symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if use_cache and symbol in self._cache:
            logger.debug(f"Using cached data for {symbol}")
            df = self._cache[symbol].copy()
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            return df

        # Set default date range
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start = datetime.now() - timedelta(days=self.cache_days)
            start_date = start.strftime('%Y-%m-%d')

        # Try primary source
        try:
            df = self._fetch_from_source(symbol, start_date, end_date, self.primary_source)
            if df is not None and not df.empty:
                self._cache[symbol] = df
                logger.info(f"Successfully fetched {len(df)} bars for {symbol} from {self.primary_source}")
                return df
        except Exception as e:
            logger.error(f"Error fetching from {self.primary_source}: {e}")

        # Fallback to Yahoo Finance if primary fails
        if self.primary_source != 'yahoo':
            try:
                logger.warning(f"Falling back to Yahoo Finance for {symbol}")
                df = self._fetch_from_yahoo(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    self._cache[symbol] = df
                    return df
            except Exception as e:
                logger.error(f"Yahoo Finance fallback failed: {e}")

        # Return empty DataFrame if all sources fail
        logger.error(f"Failed to fetch data for {symbol} from all sources")
        return pd.DataFrame()

    def _fetch_from_source(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        source: str
    ) -> Optional[pd.DataFrame]:
        """Route to the appropriate data source."""
        if source == 'yahoo':
            return self._fetch_from_yahoo(symbol, start_date, end_date)
        elif source == 'polygon':
            return self._fetch_from_polygon(symbol, start_date, end_date)
        elif source == 'alphavantage':
            return self._fetch_from_alphavantage(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {source}")

    def _fetch_from_yahoo(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Missing required columns for {symbol}")

            return df[required]

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise

    def _fetch_from_polygon(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data from Polygon.io API."""
        if not self.polygon_api_key:
            raise ValueError("Polygon API key not set")

        try:
            # Convert dates to milliseconds
            start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)

            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_ms}/{end_ms}"
            params = {
                'apiKey': self.polygon_api_key,
                'adjusted': 'true',
                'sort': 'asc'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['status'] != 'OK' or 'results' not in data:
                raise ValueError(f"No data returned from Polygon for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('date')

            # Rename columns
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Polygon error for {symbol}: {e}")
            raise

    def _fetch_from_alphavantage(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API."""
        if not self.alphavantage_api_key:
            raise ValueError("Alpha Vantage API key not set")

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'apikey': self.alphavantage_api_key,
                'outputsize': 'full'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'Time Series (Daily)' not in data:
                raise ValueError(f"No data returned from Alpha Vantage for {symbol}")

            # Convert to DataFrame
            ts_data = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '6. volume': 'volume'
            })

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            raise

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, start_date, end_date)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        return results

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        logger.info("Data cache cleared")
