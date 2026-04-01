"""
Angel One SmartAPI Authentication
Install: pip install smartapi-python pyotp logzero
"""
import os, json, logging
from pathlib import Path
from datetime import date
import pyotp

logger = logging.getLogger(__name__)
SESSION_FILE = Path("data/.angel_session")


class AngelAuth:
    def __init__(self, api_key="", client_code="", pin="", totp_secret=""):
        self.api_key = api_key or os.getenv("ANGEL_API_KEY", "")
        self.client_code = client_code or os.getenv("ANGEL_CLIENT_CODE", "")
        self.pin = pin or os.getenv("ANGEL_PIN", "")
        self.totp_secret = totp_secret or os.getenv("ANGEL_TOTP_SECRET", "")
        self.smart_api = None
        self.auth_token = None
        self.feed_token = None
        self.refresh_token = None

    def login(self):
        if not all([self.api_key, self.client_code, self.pin, self.totp_secret]):
            logger.warning("Angel One credentials not configured. Using mock mode.")
            return self._mock_login()
        try:
            from SmartApi import SmartConnect
        except ImportError:
            logger.error("smartapi-python not installed. Run: pip install smartapi-python pyotp")
            return self._mock_login()

        # Check cached session
        if SESSION_FILE.exists():
            try:
                session = json.loads(SESSION_FILE.read_text())
                if session.get("date") == str(date.today()):
                    smart_api = SmartConnect(api_key=self.api_key)
                    smart_api.setAccessToken(session["auth_token"])
                    smart_api.getProfile(session["refresh_token"])
                    self.smart_api = smart_api
                    self.auth_token = session["auth_token"]
                    self.feed_token = session["feed_token"]
                    self.refresh_token = session["refresh_token"]
                    logger.info("Reusing cached Angel One session")
                    return smart_api
            except Exception as e:
                logger.debug(f"Cached session expired: {e}")

        # Fresh login
        smart_api = SmartConnect(api_key=self.api_key)
        totp = pyotp.TOTP(self.totp_secret).now()
        data = smart_api.generateSession(self.client_code, self.pin, totp)
        if not data or data.get("status") is False:
            logger.error(f"Angel One login failed: {data}")
            return self._mock_login()

        self.auth_token = data["data"]["jwtToken"]
        self.refresh_token = data["data"]["refreshToken"]
        self.feed_token = smart_api.getfeedToken()
        self.smart_api = smart_api

        SESSION_FILE.parent.mkdir(exist_ok=True)
        SESSION_FILE.write_text(json.dumps({
            "auth_token": self.auth_token,
            "feed_token": self.feed_token,
            "refresh_token": self.refresh_token,
            "date": str(date.today()),
        }))
        logger.info(f"Angel One login successful — client: {self.client_code}")
        return smart_api

    def get_credentials(self):
        return {
            "auth_token": self.auth_token,
            "api_key": self.api_key,
            "client_code": self.client_code,
            "feed_token": self.feed_token,
        }

    def _mock_login(self):
        logger.info("Running in MOCK mode — no real Angel One connection")
        self.auth_token = "MOCK_TOKEN"
        self.feed_token = "MOCK_FEED"
        self.refresh_token = "MOCK_REFRESH"
        return MockAngelAPI()


class MockAngelAPI:
    def getProfile(self, refresh_token=None):
        return {"data": {"name": "Paper Trader", "exchanges": ["NSE"]}}
    def ltpData(self, exchange, symbol, token):
        import random
        return {"data": {"ltp": round(random.uniform(200, 3000), 2)}}
    def getMarketData(self, mode, exchange_tokens):
        return {"data": {"fetched": [], "unfetched": []}}
    def getCandleData(self, params):
        return {"data": []}
    def placeOrder(self, orderparams):
        logger.info(f"[MOCK] Order placed: {orderparams}")
        return "MOCK_ORDER_ID"
    def orderBook(self):
        return {"data": []}
    def position(self):
        return {"data": []}
