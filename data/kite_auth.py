"""Zerodha Kite Connect Authentication."""
import os, json, logging, webbrowser
from pathlib import Path
from datetime import date

logger = logging.getLogger(__name__)
SESSION_FILE = Path("data/.kite_session")


class KiteAuth:
    """Handle Kite Connect login with session caching."""
    def __init__(self, api_key="", api_secret=""):
        self.api_key = api_key or os.getenv("KITE_API_KEY", "TEST_KEY")
        self.api_secret = api_secret or os.getenv("KITE_API_SECRET", "TEST_SECRET")
        self.kite = None

    def interactive_login(self):
        """Full login flow — opens browser for manual token entry."""
        if self.api_key == "TEST_KEY":
            logger.info("TEST MODE — no real Kite connection")
            return MockKite()
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            logger.warning("kiteconnect not installed. Using MockKite.")
            return MockKite()

        # Check cached session
        if SESSION_FILE.exists():
            try:
                session = json.loads(SESSION_FILE.read_text())
                if session.get("date") == str(date.today()):
                    kite = KiteConnect(api_key=self.api_key)
                    kite.set_access_token(session["access_token"])
                    kite.profile()  # test connection
                    logger.info("Reusing cached session")
                    return kite
            except Exception:
                pass

        kite = KiteConnect(api_key=self.api_key)
        login_url = kite.login_url()
        print(f"\n  Open this URL and login:\n  {login_url}\n")
        webbrowser.open(login_url)
        request_token = input("  Paste request_token from redirect URL: ").strip()
        data = kite.generate_session(request_token, api_secret=self.api_secret)
        kite.set_access_token(data["access_token"])
        # Cache session
        SESSION_FILE.write_text(json.dumps({
            "access_token": data["access_token"],
            "date": str(date.today())
        }))
        logger.info("Login successful, session cached")
        return kite


class MockKite:
    """Mock Kite API for paper trading — returns simulated data."""
    def profile(self): return {"user_name": "TestUser"}

    def quote(self, instruments):
        import numpy as np
        result = {}
        for inst in instruments:
            sym = inst.split(":")[-1]
            price = np.random.uniform(200, 3000)
            result[inst] = {
                "last_price": round(price, 2),
                "ohlc": {"open": round(price*0.998,2), "high": round(price*1.01,2),
                         "low": round(price*0.99,2), "close": round(price*0.997,2)},
                "volume": int(np.random.uniform(500000, 5000000)),
            }
        return result

    def place_order(self, **kwargs):
        logger.info(f"  [MOCK] Order: {kwargs.get('tradingsymbol')} {kwargs.get('transaction_type')} x{kwargs.get('quantity')}")
        return "MOCK_ORDER_ID"

    def orders(self): return []
    def positions(self): return {"net": [], "day": []}
