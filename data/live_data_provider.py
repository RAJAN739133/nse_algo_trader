"""
Multi-Source Live Data Provider
═══════════════════════════════════════════════════════════════

Fetches live market data from multiple sources to avoid rate limits.
Angel One is used ONLY for order execution, not data fetching.

Data Source Priority:
1. NSE Direct (via jugaad_data) - Free, no rate limit
2. Yahoo Finance - Free, 2 req/sec limit
3. Tradient API - Free, fast
4. Angel One REST - Backup only (has 429 rate limits)

Features:
- Automatic failover between sources
- Caching to reduce API calls
- Rate limit handling
- Parallel fetching for multiple stocks
"""

import os
import time
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("data/live_cache")
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class DataSource:
    """Represents a data source with its limits."""
    name: str
    requests_per_second: float
    enabled: bool = True
    last_request: float = 0
    error_count: int = 0
    max_errors: int = 5


class LiveDataProvider:
    """
    Multi-source live data provider with automatic failover.
    
    Usage:
        provider = LiveDataProvider()
        candles = provider.get_intraday_candles("RELIANCE")
        ltp = provider.get_ltp("HDFCBANK")
        bulk_data = provider.get_bulk_candles(["RELIANCE", "TCS", "INFY"])
    """
    
    def __init__(self, angel_broker=None):
        """
        Args:
            angel_broker: Optional AngelBroker instance for order execution only
        """
        self.angel_broker = angel_broker  # For orders only, NOT data
        
        # Data sources with rate limits
        self.sources = {
            "nse_direct": DataSource("NSE Direct", requests_per_second=2.0),
            "yahoo": DataSource("Yahoo Finance", requests_per_second=2.0),
            "tradient": DataSource("Tradient API", requests_per_second=5.0),
            "angel": DataSource("Angel One", requests_per_second=0.5, enabled=False),  # Disabled by default
        }
        
        # Cache: symbol -> (timestamp, data)
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 60  # 1 minute cache
        self._lock = threading.Lock()
        
        # Request tracking
        self._request_count = 0
        self._last_log_time = time.time()
    
    def _wait_for_rate_limit(self, source: DataSource):
        """Wait if needed to respect rate limits."""
        min_interval = 1.0 / source.requests_per_second
        elapsed = time.time() - source.last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        source.last_request = time.time()
    
    def _get_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached data if fresh."""
        with self._lock:
            if symbol in self._cache:
                ts, data = self._cache[symbol]
                if time.time() - ts < self._cache_ttl:
                    return data.copy()
        return None
    
    def _set_cache(self, symbol: str, data: pd.DataFrame):
        """Cache data."""
        with self._lock:
            self._cache[symbol] = (time.time(), data.copy())
    
    # ═══════════════════════════════════════════════════════════
    # DATA SOURCE IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════
    
    def _fetch_yahoo(self, symbol: str, interval: str = "5m") -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance."""
        source = self.sources["yahoo"]
        if not source.enabled or source.error_count >= source.max_errors:
            return None
        
        try:
            self._wait_for_rate_limit(source)
            
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Get intraday data
            df = ticker.history(period="5d", interval=interval)
            
            if df.empty:
                return None
            
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            
            if "datetime" not in df.columns and "date" in df.columns:
                df.rename(columns={"date": "datetime"}, inplace=True)
            
            df["symbol"] = symbol
            
            # Filter to today
            today = date.today()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df[df["datetime"].dt.date == today].reset_index(drop=True)
            
            if len(df) > 0:
                source.error_count = 0  # Reset on success
                return df
            return None
            
        except Exception as e:
            source.error_count += 1
            logger.debug(f"Yahoo fetch failed for {symbol}: {e}")
            return None
    
    def _fetch_nse_direct(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from NSE directly via jugaad_data."""
        source = self.sources["nse_direct"]
        if not source.enabled or source.error_count >= source.max_errors:
            return None
        
        try:
            self._wait_for_rate_limit(source)
            
            from jugaad_data.nse import stock_df
            
            today = date.today()
            df = stock_df(symbol, today, today)
            
            if df.empty:
                return None
            
            # jugaad_data returns daily data, not intraday
            # This is useful for daily analysis but not live trading
            source.error_count = 0
            return df
            
        except Exception as e:
            source.error_count += 1
            logger.debug(f"NSE direct fetch failed for {symbol}: {e}")
            return None
    
    def _fetch_tradient(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from Tradient API (free NSE data)."""
        source = self.sources["tradient"]
        if not source.enabled or source.error_count >= source.max_errors:
            return None
        
        try:
            self._wait_for_rate_limit(source)
            
            import urllib.request
            import json
            
            url = f"https://api.tradient.org/v1/stocks/{symbol}/quote"
            
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Accept": "application/json",
                }
            )
            
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            
            if data and "lastPrice" in data:
                source.error_count = 0
                # Return as single-row DataFrame for LTP
                return pd.DataFrame([{
                    "symbol": symbol,
                    "close": data["lastPrice"],
                    "high": data.get("dayHigh", data["lastPrice"]),
                    "low": data.get("dayLow", data["lastPrice"]),
                    "open": data.get("open", data["lastPrice"]),
                    "volume": data.get("totalTradedVolume", 0),
                    "datetime": datetime.now(),
                }])
            return None
            
        except Exception as e:
            source.error_count += 1
            logger.debug(f"Tradient fetch failed for {symbol}: {e}")
            return None
    
    def _fetch_angel(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from Angel One (backup, rate limited)."""
        source = self.sources["angel"]
        if not source.enabled or source.error_count >= source.max_errors:
            return None
        
        if not self.angel_broker or not self.angel_broker.is_connected():
            return None
        
        try:
            self._wait_for_rate_limit(source)
            
            df = self.angel_broker.get_candles_5min(symbol, days=1)
            
            if df is not None and len(df) > 0:
                today = date.today()
                df = df[df["datetime"].dt.date == today].reset_index(drop=True)
                if len(df) > 0:
                    source.error_count = 0
                    return df
            return None
            
        except Exception as e:
            source.error_count += 1
            if "429" in str(e) or "rate" in str(e).lower():
                logger.warning(f"Angel One rate limited - disabling for this session")
                source.enabled = False
            else:
                logger.debug(f"Angel fetch failed for {symbol}: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════
    # PUBLIC INTERFACE
    # ═══════════════════════════════════════════════════════════
    
    def get_intraday_candles(
        self, 
        symbol: str, 
        interval: str = "5m",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday candles for a symbol.
        Tries multiple sources with automatic failover.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            interval: Candle interval (default "5m")
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume, symbol
        """
        # Check cache first
        if use_cache:
            cached = self._get_from_cache(symbol)
            if cached is not None:
                return cached
        
        # Try sources in order
        df = None
        
        # 1. Yahoo Finance (most reliable for intraday)
        df = self._fetch_yahoo(symbol, interval)
        if df is not None and len(df) > 0:
            self._set_cache(symbol, df)
            return df
        
        # 2. Angel One (backup)
        df = self._fetch_angel(symbol)
        if df is not None and len(df) > 0:
            self._set_cache(symbol, df)
            return df
        
        logger.warning(f"All data sources failed for {symbol}")
        return None
    
    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Get last traded price for a symbol.
        Uses fastest available source.
        """
        # Try Tradient first (fastest for LTP)
        df = self._fetch_tradient(symbol)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        
        # Fallback to Yahoo
        df = self._fetch_yahoo(symbol, "1m")
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
        
        # Fallback to cached intraday
        cached = self._get_from_cache(symbol)
        if cached is not None and len(cached) > 0:
            return float(cached["close"].iloc[-1])
        
        return None
    
    def get_bulk_candles(
        self, 
        symbols: List[str],
        max_workers: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch candles for multiple symbols in parallel.
        Respects rate limits by limiting workers.
        
        Args:
            symbols: List of stock symbols
            max_workers: Max parallel requests (default 3 to respect rate limits)
        
        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_intraday_candles, sym): sym 
                for sym in symbols
            }
            
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        results[sym] = df
                except Exception as e:
                    logger.debug(f"Failed to fetch {sym}: {e}")
        
        logger.info(f"Fetched candles for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_bulk_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Get LTP for multiple symbols."""
        results = {}
        
        for sym in symbols:
            ltp = self.get_ltp(sym)
            if ltp is not None:
                results[sym] = ltp
            time.sleep(0.1)  # Small delay to avoid rate limits
        
        return results
    
    def get_source_status(self) -> Dict[str, Dict]:
        """Get status of all data sources."""
        return {
            name: {
                "enabled": src.enabled,
                "error_count": src.error_count,
                "rate_limit": f"{src.requests_per_second} req/sec",
            }
            for name, src in self.sources.items()
        }
    
    def reset_source(self, source_name: str):
        """Reset error count and re-enable a source."""
        if source_name in self.sources:
            self.sources[source_name].error_count = 0
            self.sources[source_name].enabled = True
            logger.info(f"Reset data source: {source_name}")
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════

_provider: Optional[LiveDataProvider] = None

def get_data_provider(angel_broker=None) -> LiveDataProvider:
    """Get or create the global data provider instance."""
    global _provider
    if _provider is None:
        _provider = LiveDataProvider(angel_broker)
    elif angel_broker is not None and _provider.angel_broker is None:
        _provider.angel_broker = angel_broker
    return _provider


def fetch_live_candles(symbol: str, broker=None) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch live candles.
    Drop-in replacement for fetch_intraday_candles.
    """
    provider = get_data_provider(broker)
    return provider.get_intraday_candles(symbol)
