"""
Angel One Real-Time Data Provider
Connects to Angel One WebSocket V2 for tick-by-tick data.
Builds OHLCV candles from ticks in real-time.
Fires callbacks on every tick AND on every candle close.
"""
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from queue import Queue, Empty
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODE_LTP = 1
MODE_QUOTE = 2
MODE_FULL = 3
EXCHANGE_NSE = 1
EXCHANGE_NFO = 2
EXCHANGE_BSE = 3


class TickData:
    __slots__ = ["symbol", "token", "ltp", "volume", "open", "high", "low",
                 "close", "timestamp", "best_bid", "best_ask"]
    def __init__(self, symbol, token, ltp, volume=0, open_=0, high=0, low=0,
                 close=0, timestamp=None, best_bid=0, best_ask=0):
        self.symbol = symbol
        self.token = token
        self.ltp = ltp
        self.volume = volume
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.timestamp = timestamp or datetime.now()
        self.best_bid = best_bid
        self.best_ask = best_ask


class CandleBuilder:
    def __init__(self, symbol, interval_minutes=5):
        self.symbol = symbol
        self.interval = timedelta(minutes=interval_minutes)
        self.candles = []
        self._current = None
        self._candle_start = None
        self._tick_count = 0
        self._cum_volume_at_start = 0

    def _get_candle_start(self, ts):
        minute = (ts.minute // (self.interval.seconds // 60)) * (self.interval.seconds // 60)
        return ts.replace(minute=minute, second=0, microsecond=0)

    def add_tick(self, tick):
        ts = tick.timestamp
        candle_start = self._get_candle_start(ts)
        if self._candle_start is None:
            self._start_new_candle(tick, candle_start)
            return None
        if candle_start == self._candle_start:
            self._update_candle(tick)
            return None
        completed = self._close_candle()
        self._start_new_candle(tick, candle_start)
        return completed

    def _start_new_candle(self, tick, candle_start):
        self._candle_start = candle_start
        self._current = {
            "datetime": candle_start, "symbol": self.symbol,
            "open": tick.ltp, "high": tick.ltp, "low": tick.ltp,
            "close": tick.ltp, "volume": 0,
        }
        self._cum_volume_at_start = tick.volume
        self._tick_count = 1

    def _update_candle(self, tick):
        c = self._current
        c["high"] = max(c["high"], tick.ltp)
        c["low"] = min(c["low"], tick.ltp)
        c["close"] = tick.ltp
        if tick.volume > self._cum_volume_at_start:
            c["volume"] = tick.volume - self._cum_volume_at_start
        self._tick_count += 1

    def _close_candle(self):
        if self._current is None:
            return None
        completed = self._current.copy()
        completed["ticks"] = self._tick_count
        self.candles.append(completed)
        return completed

    def get_candles_df(self):
        if not self.candles:
            return pd.DataFrame()
        return pd.DataFrame(self.candles)

    def get_all_candles_df(self):
        all_c = list(self.candles)
        if self._current:
            all_c.append(self._current)
        if not all_c:
            return pd.DataFrame()
        return pd.DataFrame(all_c)


class RealtimeDataProvider:
    def __init__(self, angel_auth, symbol_mapper, candle_interval=5):
        self.auth = angel_auth
        self.mapper = symbol_mapper
        self.candle_interval = candle_interval
        self.on_tick = None
        self.on_candle = None
        self.candle_builders = {}
        self.tick_queue = Queue(maxsize=10000)
        self._token_to_symbol = {}
        self._running = False
        self._ws_thread = None
        self._process_thread = None
        self._sws = None
        self.tick_count = 0
        self.candle_count = 0
        self.last_tick_time = {}

    def start(self, symbols):
        self._running = True
        for sym in symbols:
            self.candle_builders[sym] = CandleBuilder(sym, self.candle_interval)
        creds = self.auth.get_credentials()
        if creds["auth_token"] == "MOCK_TOKEN":
            logger.info("No Angel One credentials — falling back to yfinance polling")
            self._start_yfinance_fallback(symbols)
            return
        token_list, mapped, unmapped = self.mapper.get_tokens_for_list(symbols)
        self._token_to_symbol = {v: k for k, v in mapped.items()}
        if unmapped:
            logger.warning(f"Unmapped symbols (won't stream): {unmapped}")
        if not token_list:
            logger.error("No symbols mapped — cannot start WebSocket")
            self._start_yfinance_fallback(symbols)
            return
        self._ws_thread = threading.Thread(target=self._run_websocket, args=(creds, token_list), daemon=True, name="angel-ws")
        self._ws_thread.start()
        self._process_thread = threading.Thread(target=self._process_ticks, daemon=True, name="tick-processor")
        self._process_thread.start()
        logger.info(f"Real-time streaming started for {len(mapped)} symbols")

    def stop(self):
        self._running = False
        if self._sws:
            try:
                self._sws.close_connection()
            except Exception:
                pass
        logger.info(f"Streaming stopped. Ticks: {self.tick_count}, Candles: {self.candle_count}")

    def _run_websocket(self, creds, token_list):
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
            sws = SmartWebSocketV2(creds["auth_token"], creds["api_key"], creds["client_code"], creds["feed_token"],
                                   max_retry_attempt=5, retry_strategy=0, retry_delay=10, retry_multiplier=2, retry_duration=30)
            self._sws = sws
            correlation_id = "nse_algo_trader"
            mode = MODE_FULL
            def on_data(wsapp, message):
                try:
                    self.tick_queue.put_nowait(message)
                except Exception:
                    pass
            def on_open(wsapp):
                logger.info("Angel One WebSocket connected")
                sws.subscribe(correlation_id, mode, token_list)
                logger.info(f"Subscribed to {sum(len(t['tokens']) for t in token_list)} tokens")
            def on_error(wsapp, error):
                logger.error(f"WebSocket error: {error}")
            def on_close(wsapp):
                logger.info("WebSocket closed")
            sws.on_data = on_data
            sws.on_open = on_open
            sws.on_error = on_error
            sws.on_close = on_close
            sws.connect()
        except ImportError:
            logger.error("smartapi-python not installed — falling back to yfinance")
        except Exception as e:
            logger.error(f"WebSocket failed: {e}")

    def _process_ticks(self):
        while self._running:
            try:
                message = self.tick_queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                tick = self._parse_tick(message)
                if tick is None:
                    continue
                self.tick_count += 1
                self.last_tick_time[tick.symbol] = tick.timestamp
                builder = self.candle_builders.get(tick.symbol)
                if builder:
                    completed = builder.add_tick(tick)
                    if completed:
                        self.candle_count += 1
                        all_candles = builder.get_all_candles_df()
                        if self.on_candle:
                            try:
                                self.on_candle(tick.symbol, completed, all_candles)
                            except Exception as e:
                                logger.error(f"on_candle callback error: {e}")
                if self.on_tick:
                    try:
                        self.on_tick(tick)
                    except Exception as e:
                        logger.error(f"on_tick callback error: {e}")
            except Exception as e:
                logger.error(f"Tick processing error: {e}")

    def _parse_tick(self, message):
        if not isinstance(message, dict):
            return None
        token = str(message.get("token", ""))
        symbol = self._token_to_symbol.get(token)
        if not symbol:
            return None
        divisor = 100.0
        ltp = message.get("last_traded_price", 0) / divisor
        if ltp <= 0:
            return None
        return TickData(
            symbol=symbol, token=token, ltp=ltp,
            volume=message.get("volume_trade_for_the_day", 0),
            open_=message.get("open_price_of_the_day", 0) / divisor,
            high=message.get("high_price_of_the_day", 0) / divisor,
            low=message.get("low_price_of_the_day", 0) / divisor,
            close=message.get("closed_price", 0) / divisor,
            timestamp=datetime.now(),
            best_bid=message.get("best_5_buy_data", [{}])[0].get("price", 0) / divisor if message.get("best_5_buy_data") else 0,
            best_ask=message.get("best_5_sell_data", [{}])[0].get("price", 0) / divisor if message.get("best_5_sell_data") else 0,
        )

    def _start_yfinance_fallback(self, symbols):
        logger.info(f"Starting yfinance fallback polling for {len(symbols)} symbols")
        self._process_thread = threading.Thread(target=self._yfinance_poll_loop, args=(symbols,), daemon=True, name="yf-poller")
        self._process_thread.start()

    def _yfinance_poll_loop(self, symbols):
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return
        last_candle_counts = {s: 0 for s in symbols}
        while self._running:
            now = datetime.now()
            if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour >= 16:
                time.sleep(30)
                continue
            for sym in symbols:
                try:
                    ticker = yf.Ticker(f"{sym}.NS")
                    df = ticker.history(period="1d", interval="1m")
                    if df.empty:
                        continue
                    df = df.reset_index()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    if "datetime" not in df.columns and "date" in df.columns:
                        df.rename(columns={"date": "datetime"}, inplace=True)
                    times = pd.to_datetime(df["datetime"])
                    today = now.date()
                    df = df[times.dt.date == today].reset_index(drop=True)
                    if len(df) <= last_candle_counts[sym]:
                        continue
                    new_count = len(df) - last_candle_counts[sym]
                    df["symbol"] = sym
                    for idx in range(last_candle_counts[sym], len(df)):
                        row = df.iloc[idx]
                        tick = TickData(
                            symbol=sym, token="", ltp=float(row["close"]),
                            volume=int(row.get("volume", 0)),
                            open_=float(row["open"]), high=float(row["high"]),
                            low=float(row["low"]), close=float(row["close"]),
                            timestamp=pd.to_datetime(row["datetime"]).to_pydatetime(),
                        )
                        builder = self.candle_builders.get(sym)
                        if builder:
                            candle = {
                                "datetime": tick.timestamp, "symbol": sym,
                                "open": tick.open, "high": tick.high,
                                "low": tick.low, "close": tick.close,
                                "volume": tick.volume, "ticks": 0,
                            }
                            builder.candles.append(candle)
                            self.candle_count += 1
                            if self.on_candle:
                                try:
                                    all_df = builder.get_all_candles_df()
                                    self.on_candle(sym, candle, all_df)
                                except Exception as e:
                                    logger.error(f"on_candle error ({sym}): {e}")
                    last_candle_counts[sym] = len(df)
                    self.tick_count += new_count
                    self.last_tick_time[sym] = datetime.now()
                except Exception as e:
                    logger.warning(f"yfinance poll error for {sym}: {e}")
            time.sleep(15)

    def get_candles(self, symbol):
        builder = self.candle_builders.get(symbol)
        if builder:
            return builder.get_all_candles_df()
        return pd.DataFrame()
