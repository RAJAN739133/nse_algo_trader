"""
Angel One Paper/Live Broker
- Paper mode: logs dummy orders, tracks positions, uses real market data
- Live mode:  places real orders via SmartAPI (future)
- Real-time data via WebSocket or REST LTP fallback
"""
import json, logging, time
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    trading_symbol: str
    side: str          # "BUY" or "SELL"
    qty: int
    order_type: str    # "MARKET", "LIMIT", "SL"
    price: float
    trigger_price: float = 0.0
    status: str = "PLACED"
    fill_price: float = 0.0
    fill_time: Optional[str] = None
    product: str = "INTRADAY"
    exchange: str = "NSE"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Position:
    symbol: str
    side: str
    qty: int
    entry_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    entry_time: str = ""


class AngelBroker:
    """
    Angel One Broker — handles both paper and live mode.
    Paper mode: uses real Angel One data but places no real orders.
    Live mode:  places real orders (requires explicit opt-in).
    """

    def __init__(self, config):
        self.mode = config.get("broker", {}).get("mode", "paper")
        self.data_source = config.get("broker", {}).get("data_source", "yfinance")
        angel_cfg = config.get("broker", {}).get("angel_one", {})

        self.api_key = angel_cfg.get("api_key", "")
        self.client_code = angel_cfg.get("client_code", "")
        self.pin = angel_cfg.get("pin", "")
        self.totp_secret = angel_cfg.get("totp_secret", "")

        self.smart_api = None
        self.auth = None
        self.symbol_mapper = None
        self.realtime_provider = None

        # Paper trading state
        self._order_counter = 0
        self.orders = []
        self.positions = {}
        self.closed_trades = []
        self.day_pnl = 0.0

        # LTP cache
        self._ltp_cache = {}
        self._ltp_cache_time = {}

    # ════════════════════════════════════════════════
    # CONNECTION
    # ════════════════════════════════════════════════

    def connect(self):
        """Login to Angel One and load symbol mapper."""
        from data.angel_auth import AngelAuth
        from data.angel_symbols import AngelSymbolMapper

        self.auth = AngelAuth(
            api_key=self.api_key,
            client_code=self.client_code,
            pin=self.pin,
            totp_secret=self.totp_secret,
        )
        self.smart_api = self.auth.login()

        self.symbol_mapper = AngelSymbolMapper()
        self.symbol_mapper.load()

        # Verify connection
        try:
            profile = self.smart_api.getProfile(self.auth.refresh_token)
            name = profile.get("data", {}).get("name", "Unknown")
            logger.info(f"Angel One connected — {name} | Mode: {self.mode.upper()}")
            return True
        except Exception as e:
            logger.warning(f"Profile fetch failed ({e}), but session may still work")
            return True

    def is_connected(self):
        return self.smart_api is not None

    # ════════════════════════════════════════════════
    # REAL-TIME DATA
    # ════════════════════════════════════════════════

    def start_realtime(self, symbols, on_candle=None, on_tick=None):
        """Start real-time data streaming via Angel One WebSocket."""
        if not self.is_connected():
            logger.error("Not connected — call connect() first")
            return False

        from data.angel_ws import RealtimeDataProvider
        self.realtime_provider = RealtimeDataProvider(
            angel_auth=self.auth,
            symbol_mapper=self.symbol_mapper,
            candle_interval=5,
        )
        self.realtime_provider.on_candle = on_candle
        self.realtime_provider.on_tick = on_tick
        self.realtime_provider.start(symbols)
        logger.info(f"Real-time data started for {len(symbols)} symbols")
        return True

    def stop_realtime(self):
        if self.realtime_provider:
            self.realtime_provider.stop()

    def get_candles(self, symbol):
        """Get accumulated 5-min candles for a symbol."""
        if self.realtime_provider:
            return self.realtime_provider.get_candles(symbol)
        return None

    def get_ltp(self, symbol):
        """Get last traded price from Angel One REST API."""
        # Check cache (valid for 2 seconds)
        now = time.time()
        if symbol in self._ltp_cache:
            if now - self._ltp_cache_time.get(symbol, 0) < 2:
                return self._ltp_cache[symbol]

        if not self.is_connected():
            return None

        try:
            token = self.symbol_mapper.get_token(symbol)
            if not token:
                return None
            trading_sym = self.symbol_mapper.get_trading_symbol(symbol)
            data = self.smart_api.ltpData("NSE", trading_sym, token)
            if data and data.get("data"):
                ltp = float(data["data"].get("ltp", 0))
                if ltp > 0:
                    self._ltp_cache[symbol] = ltp
                    self._ltp_cache_time[symbol] = now
                    return ltp
        except Exception as e:
            logger.debug(f"LTP fetch failed for {symbol}: {e}")
        return self._ltp_cache.get(symbol)

    def get_market_data(self, symbols):
        """Get quotes for multiple symbols at once."""
        if not self.is_connected():
            return {}

        result = {}
        tokens = {}
        for sym in symbols:
            token = self.symbol_mapper.get_token(sym)
            if token:
                tokens[token] = sym

        if not tokens:
            return result

        try:
            exchange_tokens = {"NSE": list(tokens.keys())}
            data = self.smart_api.getMarketData(
                mode="FULL",
                exchange_tokens=exchange_tokens,
            )
            if data and data.get("data", {}).get("fetched"):
                for item in data["data"]["fetched"]:
                    token = str(item.get("symbolToken", ""))
                    sym = tokens.get(token)
                    if sym:
                        result[sym] = {
                            "ltp": item.get("ltp", 0) / 100,
                            "open": item.get("open", 0) / 100,
                            "high": item.get("high", 0) / 100,
                            "low": item.get("low", 0) / 100,
                            "close": item.get("close", 0) / 100,
                            "volume": item.get("tradeVolume", 0),
                        }
        except Exception as e:
            logger.warning(f"Market data fetch failed: {e}")
        return result

    # ════════════════════════════════════════════════
    # ORDER PLACEMENT
    # ════════════════════════════════════════════════

    def place_order(self, symbol, side, qty, order_type="MARKET",
                    price=0.0, trigger_price=0.0, product="INTRADAY"):
        """
        Place an order.
        Paper mode: logs the order and fills immediately at LTP.
        Live mode:  sends to Angel One (future implementation).
        """
        self._order_counter += 1
        order_id = f"PAPER_{date.today()}_{self._order_counter:04d}"
        trading_symbol = self.symbol_mapper.get_trading_symbol(symbol) if self.symbol_mapper else f"{symbol}-EQ"

        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            trading_symbol=trading_symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            trigger_price=trigger_price,
            product=product,
        )

        if self.mode == "paper":
            # Paper mode: fill immediately at current LTP or given price
            fill_price = price if price > 0 else (self.get_ltp(symbol) or 0)
            if fill_price <= 0:
                order.status = "REJECTED"
                order.fill_price = 0
                logger.warning(f"[PAPER] Order REJECTED — no price for {symbol}")
            else:
                order.status = "COMPLETE"
                order.fill_price = fill_price
                order.fill_time = datetime.now().isoformat()
                self._update_position(symbol, side, qty, fill_price)
                logger.info(
                    f"[PAPER] {side} {qty}x {symbol} @ Rs {fill_price:,.2f} "
                    f"| Order: {order_id}"
                )
        elif self.mode == "live":
            # Live mode: place real order via Angel One
            order = self._place_real_order(order)

        self.orders.append(order)
        return order

    def _place_real_order(self, order):
        """Place a real order via Angel One SmartAPI."""
        if not self.is_connected():
            order.status = "REJECTED"
            logger.error("Cannot place real order — not connected")
            return order

        try:
            token = self.symbol_mapper.get_token(order.symbol)
            params = {
                "variety": "NORMAL",
                "tradingsymbol": order.trading_symbol,
                "symboltoken": token,
                "transactiontype": order.side,
                "exchange": "NSE",
                "ordertype": order.order_type,
                "producttype": order.product,
                "duration": "DAY",
                "quantity": str(order.qty),
            }
            if order.order_type == "LIMIT":
                params["price"] = str(order.price)
            if order.trigger_price > 0:
                params["triggerprice"] = str(order.trigger_price)

            resp = self.smart_api.placeOrder(params)
            if resp:
                order.order_id = str(resp)
                order.status = "PLACED"
                logger.info(f"[LIVE] Order placed: {order.side} {order.qty}x {order.symbol} | ID: {resp}")
            else:
                order.status = "REJECTED"
                logger.error(f"[LIVE] Order rejected: {order.symbol}")
        except Exception as e:
            order.status = "REJECTED"
            logger.error(f"[LIVE] Order failed: {e}")
        return order

    def _update_position(self, symbol, side, qty, price):
        """Update paper positions after a fill."""
        key = symbol
        if key in self.positions:
            pos = self.positions[key]
            if pos.side == side:
                # Adding to position
                total_qty = pos.qty + qty
                pos.entry_price = (pos.entry_price * pos.qty + price * qty) / total_qty
                pos.qty = total_qty
            else:
                # Closing position
                if qty >= pos.qty:
                    # Full close
                    if pos.side == "BUY":
                        pnl = (price - pos.entry_price) * pos.qty
                    else:
                        pnl = (pos.entry_price - price) * pos.qty
                    self.day_pnl += pnl
                    self.closed_trades.append({
                        "symbol": symbol, "side": pos.side,
                        "entry": pos.entry_price, "exit": price,
                        "qty": pos.qty, "pnl": round(pnl, 2),
                        "time": datetime.now().isoformat(),
                    })
                    remaining = qty - pos.qty
                    del self.positions[key]
                    if remaining > 0:
                        self.positions[key] = Position(
                            symbol=symbol, side=side, qty=remaining,
                            entry_price=price, entry_time=datetime.now().isoformat(),
                        )
                else:
                    # Partial close
                    if pos.side == "BUY":
                        pnl = (price - pos.entry_price) * qty
                    else:
                        pnl = (pos.entry_price - price) * qty
                    self.day_pnl += pnl
                    pos.qty -= qty
        else:
            self.positions[key] = Position(
                symbol=symbol, side=side, qty=qty,
                entry_price=price, entry_time=datetime.now().isoformat(),
            )

    # ════════════════════════════════════════════════
    # POSITION & ORDER QUERIES
    # ════════════════════════════════════════════════

    def get_positions(self):
        """Get open positions."""
        # Update P&L with current prices
        for sym, pos in self.positions.items():
            ltp = self.get_ltp(sym) or pos.entry_price
            pos.current_price = ltp
            if pos.side == "BUY":
                pos.pnl = (ltp - pos.entry_price) * pos.qty
            else:
                pos.pnl = (pos.entry_price - ltp) * pos.qty
        return self.positions

    def get_order_book(self):
        return self.orders

    def square_off_all(self):
        """Square off all open positions at market."""
        squared = 0
        for sym, pos in list(self.positions.items()):
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            self.place_order(sym, close_side, pos.qty, order_type="MARKET")
            squared += 1
        logger.info(f"[PAPER] Squared off {squared} positions | Day P&L: Rs {self.day_pnl:+,.2f}")
        return squared

    # ════════════════════════════════════════════════
    # HISTORICAL CANDLES (REST)
    # ════════════════════════════════════════════════

    def get_historical_candles(self, symbol, interval="FIVE_MINUTE", days=1):
        """Fetch historical candles via Angel One REST API."""
        if not self.is_connected():
            return None
        try:
            import pandas as pd
            from datetime import timedelta
            token = self.symbol_mapper.get_token(symbol)
            if not token:
                return None
            trading_sym = self.symbol_mapper.get_trading_symbol(symbol)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M"),
            }
            data = self.smart_api.getCandleData(params)
            if data and data.get("data"):
                df = pd.DataFrame(data["data"],
                                  columns=["datetime", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["symbol"] = symbol
                return df
        except Exception as e:
            logger.warning(f"Historical candle fetch failed for {symbol}: {e}")
        return None

    # ════════════════════════════════════════════════
    # STATUS & REPORT
    # ════════════════════════════════════════════════

    def status(self):
        """Print broker status."""
        connected = "✅ Connected" if self.is_connected() else "❌ Disconnected"
        mode = self.mode.upper()
        source = self.data_source
        open_pos = len(self.positions)
        total_orders = len(self.orders)
        return {
            "connected": self.is_connected(),
            "mode": mode,
            "data_source": source,
            "open_positions": open_pos,
            "total_orders": total_orders,
            "day_pnl": round(self.day_pnl, 2),
            "status_line": f"{connected} | Mode: {mode} | Source: {source} | "
                           f"Positions: {open_pos} | Orders: {total_orders} | "
                           f"P&L: Rs {self.day_pnl:+,.2f}",
        }

    def save_day_report(self, filepath=None):
        """Save today's paper trades to JSON."""
        if not filepath:
            filepath = f"results/paper_orders_{date.today()}.json"
        Path(filepath).parent.mkdir(exist_ok=True)
        report = {
            "date": str(date.today()),
            "mode": self.mode,
            "orders": [asdict(o) for o in self.orders],
            "closed_trades": self.closed_trades,
            "day_pnl": round(self.day_pnl, 2),
        }
        Path(filepath).write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"Day report saved: {filepath}")
