"""Data downloader — Kite historical + yfinance + live WebSocket."""
import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class KiteDataDownloader:
    """Download historical OHLCV data from Kite Connect."""
    def __init__(self, kite):
        self.kite = kite

    def get_historical(self, symbol, days=60, interval="day"):
        """Get historical candles from Kite (last 60 days for intraday)."""
        try:
            instruments = self.kite.instruments("NSE") if hasattr(self.kite, 'instruments') else []
            token = None
            for inst in instruments:
                if inst["tradingsymbol"] == symbol:
                    token = inst["instrument_token"]
                    break
            if not token:
                logger.warning(f"Token not found for {symbol}")
                return pd.DataFrame()
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            data = self.kite.historical_data(token, from_date, to_date, interval)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Download failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_yfinance(self, symbol, years=1):
        """Fallback: download via yfinance (free, no API key)."""
        try:
            import yfinance as yf
            df = yf.download(f"{symbol}.NS", period=f"{years}y", progress=False)
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.error(f"yfinance failed for {symbol}: {e}")
            return pd.DataFrame()


class LiveDataFeed:
    """WebSocket live data feed from Kite."""
    def __init__(self, kite, api_key=""):
        self.kite = kite
        self.api_key = api_key
        self.ticks = {}

    def start(self, symbols):
        """Start receiving live ticks (mock for paper trading)."""
        logger.info(f"Live feed started for {len(symbols)} symbols")
        # In paper mode, just poll quotes periodically
        for sym in symbols:
            try:
                q = self.kite.quote([f"NSE:{sym}"])
                self.ticks[sym] = q.get(f"NSE:{sym}", {})
            except Exception:
                pass

    def get_ltp(self, symbol):
        """Get last traded price."""
        if symbol in self.ticks:
            return self.ticks[symbol].get("last_price", 0)
        return 0

    def stop(self):
        logger.info("Live feed stopped")
