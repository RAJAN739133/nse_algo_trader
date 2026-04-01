"""
Angel One Symbol Token Mapper
Maps NSE ticker names (RELIANCE, SBIN) to Angel One token IDs.
Angel One WebSocket needs numeric token IDs, not ticker names.
exchangeType: 1=NSE, 2=NFO, 3=BSE
"""
import json, logging
from pathlib import Path
from datetime import date

logger = logging.getLogger(__name__)

INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / f"angel_instruments_{date.today()}.json"

FALLBACK_TOKEN_MAP = {
    "RELIANCE": "2885", "HDFCBANK": "1333", "ICICIBANK": "4963",
    "TCS": "11536", "INFY": "1594", "BHARTIARTL": "10604",
    "SBIN": "3045", "ITC": "1660", "LT": "11483",
    "KOTAKBANK": "1922", "AXISBANK": "5900", "BAJFINANCE": "317",
    "MARUTI": "10999", "SUNPHARMA": "3351", "HCLTECH": "7229",
    "WIPRO": "3787", "TATASTEEL": "3499", "NTPC": "11630",
    "POWERGRID": "14977", "COALINDIA": "20374", "ONGC": "2475",
    "TITAN": "3506", "ASIANPAINT": "236", "ULTRACEMCO": "11532",
    "BAJAJFINSV": "16675", "NESTLEIND": "17963", "TECHM": "13538",
    "HINDALCO": "1363", "HINDUNILVR": "1394", "JSWSTEEL": "11723",
    "ADANIPORTS": "15083", "DRREDDY": "881", "CIPLA": "694",
    "EICHERMOT": "910", "BRITANNIA": "547", "DIVISLAB": "10940",
    "HEROMOTOCO": "1348", "BPCL": "526", "GRASIM": "1232",
    "APOLLOHOSP": "157", "TRENT": "1964", "TATACONSUM": "3432",
    "SBILIFE": "21808", "HDFCLIFE": "467", "M&M": "2031",
    "BAJAJ-AUTO": "16669", "SHREECEM": "3103", "INDUSINDBK": "5258",
    "IOC": "1624", "TATAMOTORS": "3456", "TATAMTRDVT": "3456",
    "INDIAVIX": "26017", "NIFTY": "26000", "BANKNIFTY": "26009",
}


class AngelSymbolMapper:
    def __init__(self):
        self._token_map = {}
        self._reverse_map = {}
        self._instruments = []

    def load(self, force_download=False):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not force_download and CACHE_FILE.exists():
            try:
                self._instruments = json.loads(CACHE_FILE.read_text())
                self._build_maps()
                logger.info(f"Loaded {len(self._token_map)} NSE symbols from cache")
                return True
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        try:
            import urllib.request
            logger.info("Downloading Angel One instrument master...")
            req = urllib.request.Request(INSTRUMENT_MASTER_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            self._instruments = json.loads(data)
            CACHE_FILE.write_text(data)
            self._build_maps()
            logger.info(f"Downloaded {len(self._instruments)} instruments, {len(self._token_map)} NSE equities")
            return True
        except Exception as e:
            logger.warning(f"Instrument download failed: {e}. Using fallback map.")
            self._token_map = FALLBACK_TOKEN_MAP.copy()
            self._reverse_map = {v: k for k, v in self._token_map.items()}
            return False

    def _build_maps(self):
        self._token_map = {}
        self._reverse_map = {}
        for inst in self._instruments:
            if inst.get("exch_seg") == "NSE" and inst.get("symbol", "").endswith("-EQ"):
                name = inst["name"]
                token = inst["token"]
                self._token_map[name] = token
                self._reverse_map[token] = name
        for ticker, token in FALLBACK_TOKEN_MAP.items():
            if ticker not in self._token_map:
                self._token_map[ticker] = token
                self._reverse_map[token] = ticker

    def get_token(self, ticker):
        return self._token_map.get(ticker) or FALLBACK_TOKEN_MAP.get(ticker)

    def get_ticker(self, token_id):
        return self._reverse_map.get(str(token_id))

    def get_tokens_for_list(self, tickers):
        mapped, unmapped, tokens = {}, [], []
        for ticker in tickers:
            token = self.get_token(ticker)
            if token:
                mapped[ticker] = token
                tokens.append(token)
            else:
                unmapped.append(ticker)
                logger.warning(f"No Angel One token for: {ticker}")
        token_list = [{"exchangeType": 1, "tokens": tokens}] if tokens else []
        return token_list, mapped, unmapped

    def get_trading_symbol(self, ticker):
        return f"{ticker}-EQ"
