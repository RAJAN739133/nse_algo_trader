#!/usr/bin/env python3
"""
Live Paper Trader V3 — Adaptive All-Day Strategy with Smart Stock Selection.

Key improvements over V2:
  1. DYNAMIC STOCK SELECTION: No hardcoded list — scans for best candidates
     based on liquidity, volatility, delivery %, and ML score.
  2. MULTI-SOURCE DATA: yfinance (5min candles) + jugaad_data (NSE delivery/VWAP)
  3. REGIME DETECTION: Auto-detects trending/ranging/volatile market per stock
  4. ADAPTIVE STRATEGY: Switches algo based on detected regime
     - Trending: ORB breakout + momentum continuation (follow the trend)
     - Ranging: VWAP mean-reversion (buy low, sell high in range)
     - Volatile: Wider stops, smaller position, only strongest signals
  5. DIRECTION RESPECT: New algos respect ML direction — no counter-trend entries
  6. TELEGRAM ON EVERY EVENT: Start, picks, entries, exits, EOD summary
  7. SMART RISK: Position sizing by ATR, sector correlation check, circuit breaker
  8. LIVE HOLIDAY CHECK: Verifies from data API, no hardcoded holiday list

Usage:
  python live_paper_v3.py                         # Auto-select best stocks
  python live_paper_v3.py --universe nifty50      # From Nifty 50
  python live_paper_v3.py --stocks HDFCBANK SBIN  # Specific stocks
  python live_paper_v3.py --backtest 2026-03-30   # Replay a past day

Schedule: Scheduler calls this at 9:05 AM Mon-Fri.
"""
import os, sys, time, logging, json, pickle, argparse
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.symbols import NIFTY_50, NIFTY_100_EXTRA, NIFTY_250_EXTRA, STOCK_SECTORS, get_universe
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel
from strategies.pro_strategy_v2 import ProStrategyV2
from strategies.claude_brain import ClaudeBrain
from strategies.claude_brain_v2 import ClaudeBrainV2

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/live_v3_{date.today()}.log"),
    ],
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

def load_config():
    for p in ["config/config_test.yaml", "config/config_prod.yaml", "config/config_example.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {"capital": {"total": 100000, "risk_per_trade": 0.01, "max_trades_per_day": 5,
                        "daily_loss_limit": 0.03},
            "alerts": {"telegram_enabled": False}}


# ════════════════════════════════════════════════════════════
# TELEGRAM — sends on EVERY event
# ════════════════════════════════════════════════════════════

def send_telegram(msg, config):
    alerts = config.get("alerts", {})
    if not alerts.get("telegram_enabled"):
        logger.info(f"  [TG OFF] {msg[:80]}...")
        return
    token = alerts.get("telegram_bot_token", "")
    chat_ids = alerts.get("telegram_chat_ids", [])
    single_id = alerts.get("telegram_chat_id", "")
    if single_id and single_id not in chat_ids:
        chat_ids.append(single_id)
    if not token or not chat_ids:
        return
    try:
        import urllib.request
        for chat_id in chat_ids:
            if not chat_id:
                continue
            data = json.dumps({"chat_id": chat_id, "text": msg}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=data, headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


# ════════════════════════════════════════════════════════════
# LIVE HOLIDAY CHECK — no hardcoded list
# ════════════════════════════════════════════════════════════

def is_market_open_today():
    """Check if market is open by trying to fetch live data."""
    d = date.today()
    if d.weekday() >= 5:
        return False, "Weekend"
    try:
        import yfinance as yf
        # Fetch last 5 days of 1-min data — if today has candles, market is open
        df = yf.Ticker("^NSEI").history(period="5d", interval="1m")
        if len(df) > 0:
            latest_date = pd.to_datetime(df.index[-1]).date()
            if latest_date == d:
                return True, "Market open"
            # No data today yet — check time
            now = datetime.now()
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                return True, "Pre-market (before 9:15)"
            elif now.hour >= 9 and now.minute >= 30:
                # It's past 9:30 and still no today's data — likely a holiday
                return False, f"No data after market hours — probable holiday"
        # No data at all — assume pre-market if early
        now = datetime.now()
        if now.hour < 9 or (now.hour == 9 and now.minute < 20):
            return True, "Pre-market (assuming open)"
        return False, "No market data — probable holiday"
    except:
        return True, "API check failed, assuming open"


# ════════════════════════════════════════════════════════════
# MULTI-SOURCE DATA
# ════════════════════════════════════════════════════════════

def fetch_nse_enrichment(symbols, target_date=None):
    """Fetch delivery %, VWAP, trade count from NSE via jugaad_data."""
    enrichment = {}
    try:
        from jugaad_data.nse import stock_df
        d = target_date or date.today()
        start = d - timedelta(days=5)
        for sym in symbols:
            try:
                df = stock_df(symbol=sym, from_date=start, to_date=d, series="EQ")
                if len(df) > 0:
                    latest = df.iloc[0]  # most recent
                    enrichment[sym] = {
                        "nse_vwap": float(latest.get("VWAP", 0)),
                        "delivery_pct": float(latest.get("DELIVERY %", 0)),
                        "no_of_trades": int(latest.get("NO OF TRADES", 0)),
                        "nse_volume": int(latest.get("VOLUME", 0)),
                    }
            except:
                continue
    except ImportError:
        logger.info("  jugaad_data not available, skipping NSE enrichment")
    return enrichment


def fetch_intraday_candles(symbol, target_date=None, broker=None):
    """Fetch today's 5-min candles. Tries Angel One first, then yfinance."""
    # ── Try Angel One REST API first (faster, more reliable for live) ──
    if broker and broker.is_connected() and (target_date is None or target_date == date.today()):
        try:
            df = broker.get_historical_candles(symbol, interval="FIVE_MINUTE", days=1)
            if df is not None and len(df) > 0:
                d = target_date or date.today()
                df = df[df["datetime"].dt.date == d].reset_index(drop=True)
                if len(df) > 0:
                    return df
        except Exception as e:
            logger.debug(f"  {symbol}: Angel One candle fetch failed — {e}")

    # ── Fallback: yfinance ──
    try:
        import yfinance as yf
        if target_date and target_date != date.today():
            start = target_date - timedelta(days=3)
            end = target_date + timedelta(days=2)
            df = yf.Ticker(f"{symbol}.NS").history(start=start.isoformat(), end=end.isoformat(), interval="5m")
        else:
            df = yf.Ticker(f"{symbol}.NS").history(period="5d", interval="5m")
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["symbol"] = symbol
        d = target_date or date.today()
        times = pd.to_datetime(df["datetime"])
        df = df[times.dt.date == d].reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning(f"  {symbol}: candle fetch failed — {e}")
        return None


def fetch_vix(target_date=None):
    """Fetch India VIX. For backtest dates, fetch historical VIX for that date."""
    try:
        import yfinance as yf
        if target_date and target_date != date.today():
            start = target_date - timedelta(days=5)
            end = target_date + timedelta(days=1)
            data = yf.Ticker("^INDIAVIX").history(start=start.isoformat(), end=end.isoformat())
            if len(data) > 0:
                # Get the closest date <= target_date
                data.index = pd.to_datetime(data.index)
                valid = data[data.index.date <= target_date]
                if len(valid) > 0:
                    return round(float(valid.iloc[-1]["Close"]), 2)
        else:
            data = yf.Ticker("^INDIAVIX").history(period="5d")
            if len(data) > 0:
                return round(float(data.iloc[-1]["Close"]), 2)
    except:
        pass
    return 15.0


# ════════════════════════════════════════════════════════════
# DYNAMIC STOCK SELECTION — no hardcoded list
# ════════════════════════════════════════════════════════════

def select_best_stocks(universe_symbols, config, target_date=None, top_n=8):
    """
    Scan universe and pick the best stocks to trade based on:
    1. Data availability (must have enough history)
    2. Liquidity (avg volume > 1M)
    3. ML score (direction confidence)
    4. Volatility sweet spot (ATR 1-4% — not too flat, not too wild)
    5. Delivery % (higher = institutional interest, from NSE data)
    6. Sector diversification (max 2 per sector)
    """
    logger.info(f"\n  Scanning {len(universe_symbols)} stocks for best candidates...")

    # Step 1: Load daily data + features
    loader = DataLoader()
    target = target_date.isoformat() if target_date else date.today().isoformat()
    df = loader.load_backtest_data(universe_symbols, target_date=target)

    featured = []
    for sym in (df["symbol"].unique() if not df.empty else []):
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            feat_df = add_features(sdf)
            featured.append(feat_df)

    if not featured:
        logger.warning("  No data available for scoring")
        return pd.DataFrame()

    all_feat = pd.concat(featured, ignore_index=True)

    # Step 2: ML scoring
    model, feats = None, None
    mp = Path("models/stock_predictor.pkl")
    if mp.exists():
        with open(mp, "rb") as f:
            d = pickle.load(f)
        model, feats = d["model"], d["features"]

    avail = [c for c in (feats or FEATURE_COLS) if c in all_feat.columns]
    scores = score_stocks(all_feat, model, avail if model else None)

    # Step 3: Compute tradability metrics per stock
    candidates = []
    nse_data = fetch_nse_enrichment(scores["symbol"].tolist(), target_date)

    for _, row in scores.iterrows():
        sym = row["symbol"]
        sdf = all_feat[all_feat["symbol"] == sym]
        if len(sdf) < 50:
            continue

        last = sdf.iloc[-1]
        avg_vol = sdf["volume"].tail(20).mean() if "volume" in sdf.columns else 0
        atr_pct = last.get("atr_pct", 0.02)
        rsi = row.get("rsi", 50)

        # Liquidity filter
        if avg_vol < 500000:
            continue

        # Volatility sweet spot: 0.8% - 4%
        if atr_pct < 0.008 or atr_pct > 0.04:
            continue

        # Composite score
        ml_score = row.get("score", 50)
        direction = row.get("direction", "LONG")

        # NSE enrichment bonus
        nse = nse_data.get(sym, {})
        delivery_bonus = min(10, nse.get("delivery_pct", 0) / 5)  # up to 10 pts for high delivery
        trade_count_bonus = min(5, nse.get("no_of_trades", 0) / 100000)  # up to 5 pts

        # Volatility fitness: prefer mid-range ATR
        vol_fitness = 10 - abs(atr_pct - 0.018) * 500  # peak at 1.8% ATR

        composite = ml_score + delivery_bonus + trade_count_bonus + vol_fitness
        sector = STOCK_SECTORS.get(sym, "Other")

        candidates.append({
            "symbol": sym, "ml_score": ml_score, "direction": direction,
            "rsi": rsi, "atr_pct": round(atr_pct, 4), "avg_volume": int(avg_vol),
            "delivery_pct": nse.get("delivery_pct", 0),
            "composite_score": round(composite, 1), "sector": sector,
        })

    if not candidates:
        return pd.DataFrame()

    cdf = pd.DataFrame(candidates).sort_values("composite_score", ascending=False)

    # Step 4: Sector diversification — max 2 per sector
    selected = []
    sector_count = {}
    for _, row in cdf.iterrows():
        sec = row["sector"]
        if sector_count.get(sec, 0) >= 2:
            continue
        selected.append(row)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) >= top_n:
            break

    result = pd.DataFrame(selected)
    logger.info(f"  Selected {len(result)} stocks from {len(candidates)} candidates")
    return result


# ════════════════════════════════════════════════════════════
# REGIME DETECTION — per stock, per hour
# ════════════════════════════════════════════════════════════

class RegimeDetector:
    """Detect market regime from candle data."""

    @staticmethod
    def detect(candles, idx, lookback=12):
        """Returns: 'trending_up', 'trending_down', 'ranging', 'volatile', 'choppy'"""
        if idx < lookback:
            return "unknown"

        recent = candles.iloc[max(0, idx - lookback):idx + 1]
        closes = recent["close"].values
        highs = recent["high"].values
        lows = recent["low"].values

        # Price change over lookback
        pct_change = (closes[-1] - closes[0]) / closes[0]

        # Average candle range as % of price
        avg_range = ((highs - lows) / closes).mean()

        # Directional consistency: how many candles move in same direction
        diffs = np.diff(closes)
        up_count = np.sum(diffs > 0)
        down_count = np.sum(diffs < 0)
        total = len(diffs)
        consistency = max(up_count, down_count) / total if total > 0 else 0.5

        # High volatility
        if avg_range > 0.008:  # >0.8% per candle
            if consistency > 0.65:
                return "trending_up" if pct_change > 0 else "trending_down"
            return "volatile"

        # Trending
        if abs(pct_change) > 0.005 and consistency > 0.6:
            return "trending_up" if pct_change > 0 else "trending_down"

        # Ranging
        if abs(pct_change) < 0.003 and avg_range < 0.005:
            return "ranging"

        # Choppy (back and forth with no clear direction)
        return "choppy"


# ════════════════════════════════════════════════════════════
# ADAPTIVE V3 TRADER — switches strategy per regime
# ════════════════════════════════════════════════════════════

class AdaptiveV3Trader:
    """
    V3 trader: adaptive strategy based on regime detection.
    Respects ML direction. Sends Telegram on every event.
    """

    def __init__(self, symbol, direction, config, ml_score=50):
        self.symbol = symbol
        self.direction = direction
        self.ml_score = ml_score
        self.capital = config["capital"]["total"]
        self.config = config
        self.cost_model = ZerodhaCostModel()
        self.strategy = ProStrategyV2(config.get("strategies", {}).get("pro", {}))
        self.regime_detector = RegimeDetector()

        # State
        self.orb = None
        self.in_trade = False
        self.entry_price = 0
        self.entry_time = None
        self.entry_idx = 0
        self.side = None
        self.sl = 0
        self.tgt = 0
        self.shares = 0
        self.sl_moved_to_be = False
        self.partial_exited = False
        self.trade_type = ""
        self.current_regime = "unknown"
        self.entry_regime = "unknown"  # regime at time of entry

        self.trades = []
        self.trade_count = 0
        self.max_trades = config["capital"].get("max_trades_per_day", 3)
        self.cooldown_until = 0
        self.day_pnl = 0.0
        self.daily_loss_limit = config["capital"].get("daily_loss_limit", 0.03) * self.capital

        self.processed_candles = 0

    def _close_trade(self, exit_price, exit_time, reason, shares_to_close=None):
        shares = shares_to_close or self.shares
        if self.side == "SHORT":
            gross = (self.entry_price - exit_price) * shares
        else:
            gross = (exit_price - self.entry_price) * shares
        costs = self.cost_model.calculate(self.entry_price * shares, exit_price * shares).total
        net = gross - costs

        self.trades.append({
            "symbol": self.symbol, "direction": self.side,
            "type": self.trade_type, "regime": self.entry_regime,
            "entry": round(self.entry_price, 2), "exit": round(exit_price, 2),
            "entry_time": str(self.entry_time).split("+")[0],
            "exit_time": str(exit_time).split("+")[0],
            "sl": round(self.sl, 2), "tgt": round(self.tgt, 2),
            "qty": shares, "gross": round(gross, 2),
            "costs": round(costs, 2), "net_pnl": round(net, 2),
            "reason": reason,
        })

        self.day_pnl += net

        emoji = "✅" if net > 0 else "❌"
        et_s = str(self.entry_time).split(" ")[-1].split("+")[0][:5]
        xt_s = str(exit_time).split(" ")[-1].split("+")[0][:5]
        action_entry = "BOUGHT" if self.side == "LONG" else "SOLD SHORT"
        action_exit = "SOLD" if self.side == "LONG" else "COVERED"
        msg = (
            f"{emoji} {self.symbol} — TRADE CLOSED\n"
            f"  {action_entry} at {et_s} @ Rs {self.entry_price:,.2f}\n"
            f"  {action_exit} at {xt_s} @ Rs {exit_price:,.2f}\n"
            f"  Qty: {shares} | P&L: Rs {net:+,.2f}\n"
            f"  Type: {self.trade_type} | Regime: {self.current_regime}\n"
            f"  Exit reason: {reason} | Day P&L: Rs {self.day_pnl:+,.2f}"
        )
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)

        if shares_to_close and shares_to_close < self.shares:
            self.shares -= shares_to_close
            self.partial_exited = True
        else:
            self.in_trade = False
            self.trade_count += 1
            self.sl_moved_to_be = False
            self.partial_exited = False

    def _enter_trade(self, signal, idx):
        self.entry_price = signal["entry"]
        self.entry_time = signal["time"]
        self.entry_idx = idx
        self.side = signal["side"]
        self.sl = signal["sl"]
        self.tgt = signal["tgt"]
        self.trade_type = signal["type"]
        risk = signal["risk"]

        # ── FIX 1: Minimum SL distance = 0.5% of price ──
        # Prevents qty explosion when ATR is tiny
        min_risk = self.entry_price * 0.005  # 0.5% minimum
        if risk < min_risk:
            # Widen SL to minimum distance
            if self.side == "LONG":
                self.sl = self.entry_price - min_risk
            else:
                self.sl = self.entry_price + min_risk
            risk = min_risk
            # Also adjust target to maintain R:R ratio
            original_rr = abs(signal["tgt"] - signal["entry"]) / max(signal["risk"], 0.01)
            if self.side == "LONG":
                self.tgt = self.entry_price + risk * max(original_rr, 1.5)
            else:
                self.tgt = self.entry_price - risk * max(original_rr, 1.5)

        # ── FIX 3: Regime-based position sizing ──
        risk_pct = self.strategy.max_risk_pct
        if self.current_regime == "volatile":
            risk_pct *= 0.5
        elif self.current_regime == "choppy":
            risk_pct *= 0.5  # was 0.6, more conservative now
        elif self.current_regime == "ranging":
            risk_pct *= 0.4  # heavily reduce in ranging
        # Afternoon trades get 60% position (riskier time)
        if "AFTERNOON" in self.trade_type:
            risk_pct *= 0.6

        # ── FIX 1b: Cap max shares to limit single-trade damage ──
        self.shares = max(1, int(self.capital * risk_pct / max(risk, min_risk)))
        max_shares = int(self.capital * 0.15 / self.entry_price)  # max 15% capital per trade
        self.shares = min(self.shares, max_shares)

        # ── FIX 5: Minimum expected profit filter — skip if profit won't cover costs ──
        expected_gross = abs(self.tgt - self.entry_price) * self.shares
        est_costs = self.cost_model.calculate(
            self.entry_price * self.shares, self.tgt * self.shares
        ).total
        if expected_gross < est_costs * 2:
            logger.info(
                f"  ⛔ {self.symbol} SKIP: expected gross Rs {expected_gross:.0f} "
                f"< 2× costs Rs {est_costs:.0f}"
            )
            return

        self.in_trade = True
        self.sl_moved_to_be = False
        self.partial_exited = False
        self.entry_regime = self.current_regime  # snapshot regime at entry

        side_emoji = "📈" if self.side == "LONG" else "📉"
        action = "BUYING" if self.side == "LONG" else "SELLING SHORT"
        entry_time_str = signal['time'].strftime('%H:%M') if hasattr(signal['time'], 'strftime') else str(signal['time'])
        msg = (
            f"{side_emoji} {self.symbol} — {action} NOW\n"
            f"  Time: {entry_time_str} | Price: Rs {self.entry_price:,.2f}\n"
            f"  Stop loss: Rs {self.sl:,.2f} | Target: Rs {self.tgt:,.2f}\n"
            f"  Qty: {self.shares} | Risk: {risk_pct*100:.1f}%\n"
            f"  Strategy: {self.trade_type} | Regime: {self.current_regime}\n"
            f"  Reason: {signal['reason']}"
        )
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)

    def process_new_candles(self, candles):
        if candles is None or len(candles) == 0:
            return
        start = self.processed_candles
        for i in range(start, len(candles)):
            self._process_candle(i, candles)
        self.processed_candles = len(candles)

    def _process_candle(self, i, candles):
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        h, l, c = row["high"], row["low"], row["close"]
        hour, minute = t.hour, t.minute

        # Update regime every 6 candles (30 min)
        if i % 6 == 0 and i >= 12:
            self.current_regime = self.regime_detector.detect(candles, i)

        # ── Circuit breaker: stop if daily loss exceeded ──
        if self.day_pnl < -self.daily_loss_limit and not self.in_trade:
            return

        # ── Phase 1: ORB Formation ──
        if self.orb is None:
            if i < self.strategy.orb_candles:
                return
            self.orb = self.strategy.compute_orb(candles, self.strategy.orb_candles)
            if self.orb is None or self.orb["range"] < 0.5:
                return
            msg = f"📐 {self.symbol} ORB: H={self.orb['high']:,.2f} L={self.orb['low']:,.2f} R={self.orb['range']:,.2f} | Regime: {self.current_regime}"
            logger.info(f"  {msg}")

        # ── Phase 2: Square off ──
        if hour >= 15 and minute >= 10:
            if self.in_trade:
                self._close_trade(c, t, "SQUARE_OFF")
            return

        # ── Phase 3: Manage open trade ──
        if self.in_trade:
            self._manage_trade(candles, i)
            return

        # ── Phase 4: Look for entries based on regime ──
        if self.trade_count >= self.max_trades:
            return
        if i < self.cooldown_until:
            return

        self._scan_for_entry(candles, i)

    def _manage_trade(self, candles, i):
        """Manage open position with trailing, partial exit, time-decay."""
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        h, l, c = row["high"], row["low"], row["close"]

        # Time-decay: cut flat trades after ~45 min (was 90 — held too long, paid costs)
        time_decay_candles = 9  # 9 × 5min = 45min
        if (i - self.entry_idx) >= time_decay_candles:
            pct_from_entry = abs(c - self.entry_price) / self.entry_price
            if pct_from_entry < 0.003:
                self._close_trade(c, t, "TIME_DECAY")
                return

        if self.side == "LONG":
            if l <= self.sl:
                self._close_trade(self.sl, t, "STOP_LOSS")
                self.cooldown_until = i + self.strategy.cooldown_candles
                return
            if h >= self.tgt:
                self._close_trade(self.tgt, t, "TARGET")
                return
            # Partial exit at 1x risk
            if not self.partial_exited:
                initial_risk = self.entry_price - self.sl
                if initial_risk > 0 and (c - self.entry_price) >= initial_risk * self.strategy.partial_exit_at_rr:
                    partial = max(1, int(self.shares * self.strategy.partial_exit_pct))
                    if partial < self.shares:
                        self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial)
            # Breakeven trail
            unrealised = c - self.entry_price
            initial_risk = self.entry_price - self.sl
            if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                self.sl = self.entry_price + 0.10
                self.sl_moved_to_be = True
                logger.info(f"  🔒 {self.symbol} SL → breakeven Rs {self.sl:,.2f}")
        else:  # SHORT
            if h >= self.sl:
                self._close_trade(self.sl, t, "STOP_LOSS")
                self.cooldown_until = i + self.strategy.cooldown_candles
                return
            if l <= self.tgt:
                self._close_trade(self.tgt, t, "TARGET")
                return
            if not self.partial_exited:
                initial_risk = self.sl - self.entry_price
                if initial_risk > 0 and (self.entry_price - c) >= initial_risk * self.strategy.partial_exit_at_rr:
                    partial = max(1, int(self.shares * self.strategy.partial_exit_pct))
                    if partial < self.shares:
                        self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial)
            unrealised = self.entry_price - c
            initial_risk = self.sl - self.entry_price
            if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                self.sl = self.entry_price - 0.10
                self.sl_moved_to_be = True
                logger.info(f"  🔒 {self.symbol} SL → breakeven Rs {self.sl:,.2f}")

    def _scan_for_entry(self, candles, i):
        """Scan for entry signals based on current regime and ML direction."""
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        c = row["close"]
        hour, minute = t.hour, t.minute

        regime = self.current_regime
        direction = self.direction

        # ── FIX: Intraday range filter — skip if stock is dead today ──
        if i > 24:  # after 2 hours of data
            day_high = candles["high"].iloc[:i+1].max()
            day_low = candles["low"].iloc[:i+1].min()
            day_range_pct = (day_high - day_low) / day_low if day_low > 0 else 0
            if day_range_pct < 0.004:  # less than 0.4% range all day
                return  # dead market, skip

        # ── Trending regime: ORB breakout + momentum ──
        if regime in ("trending_up", "trending_down", "unknown"):
            # ORB breakout (9:20-10:30) — WITH direction
            if hour < 10 or (hour == 10 and minute <= 30):
                signal = self.strategy.generate_orb_signal(candles, i, self.orb, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return

            # Pullback to ORB level (10:00-11:30) — LONG ONLY (short pullbacks have 0% WR)
            if 10 <= hour <= 11 and direction == "LONG":
                signal = self.strategy.generate_pullback_signal(candles, i, self.orb, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return

            # Momentum continuation (11:30-14:00) — SAME direction as ML
            if (hour == 11 and minute >= 30) or (12 <= hour <= 13):
                signal = self._generate_momentum_signal(candles, i, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return

        # ── Ranging regime: SKIP most trades ──
        # Ranging lost Rs -5,432 across 20 trades. Only allow VWAP with extreme signals.
        if regime in ("ranging", "choppy"):
            # Only VWAP with very extreme deviation (3σ instead of 1.5σ)
            if hour >= 11 and hour < 14:
                signal = self._generate_vwap_direction_signal(candles, i, direction)
                if signal:
                    # Extra filter: only enter if deviation is extreme
                    vwap, std = self.strategy.compute_vwap_proper(candles, i)
                    dev = abs(candles.iloc[i]["close"] - vwap) / max(std, 0.01)
                    if dev >= 2.5:  # was 1.5, now much stricter
                        self._enter_trade(signal, i)
                        return
            # NO afternoon trend in ranging — that's where the big losses came from
            return

        # ── All regimes: Afternoon trend-following (13:30-14:30) ──
        # Only enter in ML direction with strong confirmation
        if 13 <= hour <= 14 and minute <= 30:
            signal = self._generate_afternoon_trend_signal(candles, i, direction)
            if signal:
                self._enter_trade(signal, i)
                return

    def _generate_momentum_signal(self, candles, i, direction, lookback=8):
        """Momentum continuation in ML direction only."""
        if i < lookback + 3:
            return None
        row = candles.iloc[i]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])

        recent = candles.iloc[max(0, i - lookback):i + 1]
        price_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

        # Volume confirmation
        avg_vol = candles["volume"].iloc[max(0, i - 20):i].mean()
        curr_vol = candles["volume"].iloc[i]
        vol_ok = avg_vol > 0 and curr_vol > avg_vol * 1.2

        if not vol_ok:
            return None

        # MUST align with ML direction
        if direction == "LONG" and price_change > 0.006:
            atr = self._quick_atr(candles, i)
            sl = close - atr * 1.5
            risk = close - sl
            tgt = close + risk * 2.0
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "MOMENTUM_LONG",
                "reason": f"Momentum +{price_change*100:.1f}% with vol, regime={self.current_regime}"
            }
        elif direction == "SHORT" and price_change < -0.006:
            atr = self._quick_atr(candles, i)
            sl = close + atr * 1.5
            risk = sl - close
            tgt = close - risk * 2.0
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "MOMENTUM_SHORT",
                "reason": f"Momentum {price_change*100:.1f}% with vol, regime={self.current_regime}"
            }
        return None

    def _generate_vwap_direction_signal(self, candles, i, direction):
        """VWAP mean-reversion but only in ML direction."""
        row = candles.iloc[i]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        hour = t.hour

        if hour < 10 or hour >= 14:
            return None

        vwap, std = self.strategy.compute_vwap_proper(candles, i)
        if std == 0 or std < close * 0.003:
            return None

        rsi = self.strategy.compute_rsi(candles["close"].iloc[:i + 1], period=14)
        deviation = (close - vwap) / std

        # LONG only when ML says LONG and price is below VWAP
        if direction == "LONG" and deviation < -1.5 and rsi < 35:
            sl = close - std * 1.5
            risk = close - sl
            if risk <= 0:
                return None
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": vwap,
                "risk": risk, "time": t, "type": "VWAP_LONG",
                "reason": f"VWAP oversold {deviation:.1f}σ RSI={rsi:.0f}, ML=LONG"
            }

        # SHORT only when ML says SHORT and price is above VWAP
        if direction == "SHORT" and deviation > 1.5 and rsi > 65:
            sl = close + std * 1.5
            risk = sl - close
            if risk <= 0:
                return None
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": vwap,
                "risk": risk, "time": t, "type": "VWAP_SHORT",
                "reason": f"VWAP overbought +{deviation:.1f}σ RSI={rsi:.0f}, ML=SHORT"
            }
        return None

    def _generate_afternoon_trend_signal(self, candles, i, direction):
        """Afternoon entry — only strong signals in ML direction with volume."""
        if i < 30:
            return None
        row = candles.iloc[i]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])

        # Need very strong signal: RSI + VWAP + volume all aligned
        rsi = self.strategy.compute_rsi(candles["close"].iloc[:i + 1], period=7)
        vwap, std = self.strategy.compute_vwap_proper(candles, i)
        avg_vol = candles["volume"].iloc[max(0, i - 20):i].mean()
        curr_vol = candles["volume"].iloc[i]

        if avg_vol == 0 or curr_vol < avg_vol * 1.3:
            return None  # need strong volume

        # 5-candle trend alignment
        recent_close = candles["close"].iloc[max(0, i - 5):i + 1]
        trend_up = all(recent_close.diff().dropna() > 0)
        trend_down = all(recent_close.diff().dropna() < 0)

        if direction == "LONG" and trend_up and rsi > 55 and close > vwap:
            atr = self._quick_atr(candles, i)
            sl = close - atr * 1.2
            risk = close - sl
            tgt = close + risk * 1.5
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "AFTERNOON_TREND_LONG",
                "reason": f"5-candle uptrend + vol + above VWAP, RSI={rsi:.0f}"
            }

        if direction == "SHORT" and trend_down and rsi < 45 and close < vwap:
            atr = self._quick_atr(candles, i)
            sl = close + atr * 1.2
            risk = sl - close
            tgt = close - risk * 1.5
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "AFTERNOON_TREND_SHORT",
                "reason": f"5-candle downtrend + vol + below VWAP, RSI={rsi:.0f}"
            }
        return None

    def _quick_atr(self, candles, idx, period=14):
        """ATR with a floor of 0.5% of price to prevent micro-SL disasters."""
        if idx < period:
            raw = (candles["high"].iloc[:idx+1] - candles["low"].iloc[:idx+1]).mean()
        else:
            recent = candles.iloc[max(0, idx - period):idx + 1]
            raw = (recent["high"] - recent["low"]).mean()
        # Floor: ATR must be at least 0.5% of current price
        price = candles["close"].iloc[idx]
        return max(raw, price * 0.005)


# ════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════

def run(symbols=None, backtest_date=None):
    config = load_config()
    target_date = date.fromisoformat(backtest_date) if backtest_date else None
    is_backtest = target_date is not None
    today_str = str(target_date or date.today())

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  LIVE PAPER TRADER V3 — ADAPTIVE")
    logger.info(f"  Date: {today_str} {'(BACKTEST)' if is_backtest else '(LIVE)'}")
    logger.info(f"  Capital: Rs {config['capital']['total']:,}")
    logger.info(f"{'=' * 60}")

    # ── Holiday check (live API) ──
    if not is_backtest:
        is_open, reason = is_market_open_today()
        if not is_open:
            msg = f"📅 {today_str} — Market closed ({reason})"
            logger.info(msg)
            send_telegram(msg, config)
            return

    # ── Initialize Angel One broker (if configured) ──
    broker = None
    if not is_backtest and config.get("broker", {}).get("angel_one", {}).get("api_key"):
        try:
            from data.angel_broker import AngelBroker
            broker = AngelBroker(config)
            if broker.connect():
                logger.info(f"  🔗 Angel One connected | Mode: {broker.mode.upper()}")
                send_telegram(f"🔗 Angel One connected | Mode: {broker.mode.upper()}", config)
            else:
                broker = None
        except Exception as e:
            logger.warning(f"  Angel One init failed ({e}) — yfinance fallback")
            broker = None

    # ── VIX check ──
    vix = fetch_vix(target_date)

    # ── Initialize Angel One broker (if configured) ──
    broker = None
    if not is_backtest and config.get("broker", {}).get("angel_one", {}).get("api_key"):
        try:
            from data.angel_broker import AngelBroker
            broker = AngelBroker(config)
            if broker.connect():
                logger.info(f"  🔗 Angel One connected | Mode: {broker.mode.upper()}")
            else:
                broker = None
        except Exception as e:
            logger.warning(f"  Angel One init error: {e}")
            broker = None
    vix_limit = config.get("filters", {}).get("vix_extreme_threshold", 40)
    if vix > vix_limit:
        msg = f"⚠️ VIX {vix} > {vix_limit} — EXTREME. Sitting out."
        logger.warning(msg)
        send_telegram(msg, config)
        return

    # ── Claude Brain V2 morning analysis ──
    brain = ClaudeBrainV2(config=config)
    brain_advice = None
    if brain.enabled and not is_backtest:
        logger.info("  🧠 Consulting Claude Brain V2...")
        brain_advice = brain.morning_analysis(vix=vix, stock_scores=[])
        if brain_advice:
            logger.info(f"  🧠 Claude says: {brain_advice.get('risk_level', 'N/A')} | "
                        f"Max trades: {brain_advice.get('max_trades', 'N/A')} | "
                        f"Sentiment: {brain_advice.get('news_sentiment', 'N/A')} | "
                        f"{brain_advice.get('market_outlook', '')}")
            send_telegram(
                f"🧠 Claude Brain V2 — {today_str}\n"
                f"Risk: {brain_advice.get('risk_level', 'N/A')}\n"
                f"Sentiment: {brain_advice.get('news_sentiment', 'N/A')}\n"
                f"Max trades: {brain_advice.get('max_trades', 'N/A')}\n"
                f"Skip: {brain_advice.get('skip_stocks', [])}\n"
                f"Prefer: {brain_advice.get('preferred_stocks', [])}\n"
                f"Outlook: {brain_advice.get('market_outlook', '')}\n"
                f"Notes: {brain_advice.get('notes', '')}",
                config,
            )
            if brain_advice.get("max_trades"):
                config["capital"]["max_trades_per_day"] = brain_advice["max_trades"]

    # ── Dynamic stock selection ──
    if symbols:
        # If specific stocks given, still score them
        picks = select_best_stocks(symbols, config, target_date, top_n=len(symbols))
    else:
        # Scan full universe — pick best candidates
        universe = get_universe("nifty50")  # start with nifty50, expand if needed
        picks = select_best_stocks(universe, config, target_date, top_n=8)

    if picks.empty:
        msg = "⚠️ No stocks passed selection criteria today."
        logger.warning(msg)
        send_telegram(msg, config)
        return

    # ── Apply Claude Brain skip list ──
    if brain_advice and brain_advice.get("skip_stocks"):
        skip = brain_advice["skip_stocks"]
        before = len(picks)
        picks = picks[~picks["symbol"].isin(skip)]
        if len(picks) < before:
            logger.info(f"  🧠 Claude skipped {before - len(picks)} stocks: {skip}")

    # ── Telegram: picks ──
    pick_lines = [f"🤖 V3 Adaptive — {today_str}", f"VIX: {vix}", ""]
    for _, r in picks.iterrows():
        arrow = "📈" if r["direction"] == "LONG" else "📉"
        pick_lines.append(
            f"{arrow} {r['symbol']:<12} ML:{r['ml_score']:.0f} Comp:{r['composite_score']:.0f} "
            f"ATR:{r['atr_pct']*100:.1f}% Del:{r['delivery_pct']:.0f}% {r['direction']}"
        )
    pick_lines.append(f"\nRegime detection: ON | Direction lock: ON")
    pick_msg = "\n".join(pick_lines)
    logger.info(pick_msg)
    send_telegram(pick_msg, config)

    # ── Fetch intraday candles ──
    logger.info(f"\n  Fetching 5-min candles...")
    all_candles = {}
    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        candles = fetch_intraday_candles(sym, target_date, broker=broker)
        if candles is not None and len(candles) >= 10:
            all_candles[sym] = candles
            logger.info(f"    {sym}: {len(candles)} candles")

    if not all_candles and is_backtest:
        msg = "⚠️ No intraday data available."
        logger.warning(msg)
        send_telegram(msg, config)
        return

    # ── Create adaptive traders ──
    traders = {}
    if is_backtest:
        # Backtest: only create traders for stocks with candles
        for _, pick in picks.iterrows():
            sym = pick["symbol"]
            if sym not in all_candles:
                continue
            traders[sym] = AdaptiveV3Trader(
                sym, pick["direction"], config, ml_score=pick["ml_score"]
            )
    else:
        # Live: create traders for ALL picks — candles will arrive via polling
        for _, pick in picks.iterrows():
            sym = pick["symbol"]
            traders[sym] = AdaptiveV3Trader(
                sym, pick["direction"], config, ml_score=pick["ml_score"]
            )

    # ── Run: backtest (all candles at once) or live (polling loop) ──
    if is_backtest:
        logger.info(f"\n  Running backtest on {today_str}...")
        for sym, trader in traders.items():
            candles = all_candles[sym]
            trader.process_new_candles(candles)
    else:
        # Live polling loop
        logger.info(f"\n  {'─' * 50}")
        logger.info(f"  MARKET OPEN — Polling 5-min candles")
        logger.info(f"  {'─' * 50}")

        now = datetime.now()
        market_open = now.replace(hour=9, minute=20, second=0)
        if now < market_open:
            wait = (market_open - now).total_seconds()
            logger.info(f"  Waiting {wait/60:.0f} min for market open...")
            time.sleep(max(0, wait))

        market_close = now.replace(hour=15, minute=25, second=0)
        poll_interval = 65
        last_status = datetime.now()

        while datetime.now() < market_close:
            now_t = datetime.now()
            for sym, trader in traders.items():
                try:
                    candles = fetch_intraday_candles(sym, broker=broker)
                    if candles is not None and len(candles) > trader.processed_candles:
                        trader.process_new_candles(candles)
                except Exception as e:
                    logger.error(f"  {sym} error: {e}")

            # Status every 15 min
            if (now_t - last_status).total_seconds() >= 900:
                open_pos = sum(1 for t in traders.values() if t.in_trade)
                closed = sum(len(t.trades) for t in traders.values())
                total_pnl = sum(tr["net_pnl"] for t in traders.values() for tr in t.trades)
                regimes = {sym: t.current_regime for sym, t in traders.items() if t.current_regime != "unknown"}
                status_msg = (
                    f"⏱ {now_t.strftime('%H:%M')} | Open: {open_pos} | Closed: {closed} | "
                    f"P&L: Rs {total_pnl:+,.0f}\nRegimes: {regimes}"
                )
                logger.info(f"  {status_msg}")
                send_telegram(status_msg, config)
                last_status = now_t

                # ── Claude Brain V2: Live adjustment every 15 min ──
                if brain.enabled:
                    try:
                        live_state = {
                            "time": now_t.strftime("%H:%M"),
                            "vix": vix,
                            "day_pnl": total_pnl,
                            "trades_taken": closed,
                            "open_positions": [
                                {"symbol": s, "side": t.side, "pnl": t.day_pnl, "regime": t.current_regime}
                                for s, t in traders.items() if t.in_trade
                            ],
                            "stock_regimes": regimes,
                        }
                        adj = brain.live_adjustment(live_state)
                        if adj and adj.get("emergency_exits"):
                            for sym_exit in adj["emergency_exits"]:
                                if sym_exit in traders and traders[sym_exit].in_trade:
                                    logger.warning(f"  🧠 EMERGENCY EXIT: {sym_exit}")
                                    candles_ex = fetch_intraday_candles(sym_exit, broker=broker)
                                    if candles_ex is not None and len(candles_ex) > 0:
                                        traders[sym_exit]._close_trade(
                                            candles_ex.iloc[-1]["close"],
                                            pd.to_datetime(candles_ex.iloc[-1]["datetime"]),
                                            "CLAUDE_EMERGENCY"
                                        )
                        if adj and adj.get("notes") and adj["notes"] != "No live adjustment":
                            logger.info(f"  🧠 Brain: {adj['notes']}")
                    except Exception as brain_err:
                        logger.debug(f"Brain live adjustment error: {brain_err}")

            time.sleep(poll_interval)

    # ── End of day summary ──
    all_trades = []
    for sym, trader in traders.items():
        all_trades.extend(trader.trades)

    total_pnl = sum(t["net_pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["net_pnl"] > 0)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  END OF DAY — {today_str} — V3 Adaptive")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Trades: {len(all_trades)} | Won: {wins} | Lost: {len(all_trades) - wins}")
    logger.info(f"  Net P&L: Rs {total_pnl:+,.2f}")

    # Save CSV
    if all_trades:
        df = pd.DataFrame(all_trades)
        outpath = f"results/live_v3_{today_str}.csv"
        df.to_csv(outpath, index=False)
        logger.info(f"  Saved: {outpath}")

    # Telegram EOD
    summary = [f"📊 V3 Adaptive — {today_str}", ""]
    if all_trades:
        wr = wins / len(all_trades) * 100
        summary.append(f"📋 Trades: {len(all_trades)} | Won: {wins} | WR: {wr:.0f}%")
        summary.append("─" * 30)
        for t in all_trades:
            emoji = "✅" if t["net_pnl"] > 0 else "❌"
            summary.append(f"{emoji} {t['symbol']} {t['direction']} ({t['type']})")
            summary.append(f"  Rs {t['entry']:,.2f} → Rs {t['exit']:,.2f} | Rs {t['net_pnl']:+,.2f}")
            summary.append(f"  Regime: {t.get('regime', 'N/A')} | {t['reason']}")
        summary.append("─" * 30)
        summary.append(f"💰 Net P&L: Rs {total_pnl:+,.2f}")
    else:
        summary.append("📭 No trades today")

    summary.append("")
    summary.append("V3 features: Regime detection | Direction lock | Dynamic picks")
    eod_msg = "\n".join(summary)
    logger.info(eod_msg)
    send_telegram(eod_msg, config)

    # ── Claude Brain EOD analysis ──
    if brain.enabled and all_trades and not is_backtest:
        eod = brain.eod_analysis(all_trades, total_pnl, {})
        if eod:
            logger.info(f"  🧠 Claude EOD: {eod}")

    # ── Save broker report ──
    if broker:
        broker.save_day_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Adaptive Live Paper Trader")
    parser.add_argument("--universe", default=None, choices=["nifty50", "nifty100", "nifty250"])
    parser.add_argument("--stocks", nargs="+", default=None)
    parser.add_argument("--backtest", default=None, help="Backtest date YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=8, help="Top N stocks to trade")
    args = parser.parse_args()

    if args.stocks:
        run(args.stocks, backtest_date=args.backtest)
    elif args.universe:
        run(get_universe(args.universe), backtest_date=args.backtest)
    else:
        run(backtest_date=args.backtest)
