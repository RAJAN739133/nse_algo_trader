#!/usr/bin/env python3
"""
Strategy Comparison — V1 (start_bot) vs V2 (ProStrategyV2) vs V3-concept
Uses REAL yfinance 5-min intraday data. No synthetic/fake data.

Usage:
    python compare_strategies.py                    # Auto-detect last trading day
    python compare_strategies.py --date 2026-03-30  # Specific date
"""
import os, sys, logging, pickle, json, argparse
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config.symbols import NIFTY_50
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel
from strategies.pro_strategy_v2 import ProStrategyV2

Path("results").mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# STEP 0: Fetch real 5-min intraday data from yfinance
# ════════════════════════════════════════════════════════════

def fetch_5min_candles(symbol, target_date):
    """Fetch 5-min candles for a specific past date from yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        # yfinance needs a range — fetch 5 days around the target
        start = target_date - timedelta(days=3)
        end = target_date + timedelta(days=2)
        df = ticker.history(start=start.isoformat(), end=end.isoformat(), interval="5m")
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["symbol"] = symbol
        # Filter to target date only
        times = pd.to_datetime(df["datetime"])
        df = df[times.dt.date == target_date].reset_index(drop=True)
        if len(df) == 0:
            return None
        return df
    except Exception as e:
        logger.warning(f"  {symbol}: 5min fetch failed — {e}")
        return None


def fetch_daily_data(symbols, target_date):
    """Fetch daily OHLCV for ML scoring."""
    try:
        loader = DataLoader()
        df = loader.load_backtest_data(symbols, target_date=target_date.isoformat())
        return df
    except Exception as e:
        logger.error(f"  Daily data fetch failed: {e}")
        return pd.DataFrame()


def check_market_open_live(target_date):
    """Check if market was open by trying to fetch live data (no hardcoded holidays)."""
    import yfinance as yf
    # Try NIFTY 50 index or a liquid stock
    for sym in ["^NSEI", "RELIANCE.NS", "HDFCBANK.NS"]:
        try:
            ticker = yf.Ticker(sym)
            start = target_date
            end = target_date + timedelta(days=1)
            df = ticker.history(start=start.isoformat(), end=end.isoformat(), interval="1d")
            if len(df) > 0:
                return True, f"Data found for {sym}"
        except:
            continue
    return False, "No market data found — likely a holiday"


# ════════════════════════════════════════════════════════════
# V1 STRATEGY — Basic ORB + VWAP (from start_bot.py logic)
# ════════════════════════════════════════════════════════════

def run_v1_on_candles(candles, symbol, direction, config):
    """
    V1 logic from start_bot.py: Simple ML score → single entry/exit using daily data.
    Uses last daily close, ATR-based SL/TGT, simulates against intraday candles.
    """
    cost_model = ZerodhaCostModel()
    trades = []
    capital = config["capital"]["total"]

    if candles is None or len(candles) < 10:
        return trades

    # Simple ORB: first 3 candles
    orb_high = candles.iloc[:3]["high"].max()
    orb_low = candles.iloc[:3]["low"].min()
    orb_range = orb_high - orb_low
    if orb_range < 0.5:
        return trades

    entry_price = None
    entry_time = None
    sl = None
    tgt = None
    side = direction

    for i in range(3, len(candles)):
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        close = row["close"]
        high = row["high"]
        low = row["low"]
        hour, minute = t.hour, t.minute

        # Square off
        if hour >= 15 and minute >= 10:
            if entry_price is not None:
                exit_p = close
                if side == "LONG":
                    gross = (exit_p - entry_price) * shares
                else:
                    gross = (entry_price - exit_p) * shares
                costs = cost_model.calculate(entry_price * shares, exit_p * shares).total
                net = gross - costs
                trades.append({
                    "symbol": symbol, "side": side, "strategy": "V1_BASIC",
                    "entry": round(entry_price, 2), "exit": round(exit_p, 2),
                    "entry_time": str(entry_time), "exit_time": str(t),
                    "qty": shares, "net_pnl": round(net, 2), "reason": "SQUARE_OFF",
                })
            break

        # Entry logic — simple breakout, no volume/momentum checks
        if entry_price is None and hour < 11:
            buffer = close * 0.002
            if side == "LONG" and close > orb_high + buffer:
                entry_price = close
                entry_time = t
                sl = orb_low
                risk = entry_price - sl
                tgt = entry_price + risk * 1.5  # fixed 1.5 RR
                shares = max(1, int(capital * 0.01 / max(risk, 1)))
            elif side == "SHORT" and close < orb_low - buffer:
                entry_price = close
                entry_time = t
                sl = orb_high
                risk = sl - entry_price
                tgt = entry_price - risk * 1.5
                shares = max(1, int(capital * 0.01 / max(risk, 1)))
            continue

        # Exit logic — simple SL/TGT, no trailing
        if entry_price is not None:
            if side == "LONG":
                if low <= sl:
                    exit_p = sl
                    reason = "STOP_LOSS"
                elif high >= tgt:
                    exit_p = tgt
                    reason = "TARGET"
                else:
                    continue
                gross = (exit_p - entry_price) * shares
            else:
                if high >= sl:
                    exit_p = sl
                    reason = "STOP_LOSS"
                elif low <= tgt:
                    exit_p = tgt
                    reason = "TARGET"
                else:
                    continue
                gross = (entry_price - exit_p) * shares

            costs = cost_model.calculate(entry_price * shares, exit_p * shares).total
            net = gross - costs
            trades.append({
                "symbol": symbol, "side": side, "strategy": "V1_BASIC",
                "entry": round(entry_price, 2), "exit": round(exit_p, 2),
                "entry_time": str(entry_time), "exit_time": str(t),
                "qty": shares, "net_pnl": round(net, 2), "reason": reason,
            })
            break

    return trades


# ════════════════════════════════════════════════════════════
# V2 STRATEGY — ProStrategyV2 (full candle-by-candle)
# ════════════════════════════════════════════════════════════

def run_v2_on_candles(candles, symbol, direction, config):
    """V2: Full ProStrategyV2 with ORB + VWAP + Pullback, trailing, partial exit."""
    if candles is None or len(candles) < 10:
        return []

    cost_model = ZerodhaCostModel()
    strategy = ProStrategyV2(config.get("strategies", {}).get("pro", {}))
    capital = config["capital"]["total"]

    orb = None
    in_trade = False
    entry_price = 0
    entry_time = None
    entry_idx = 0
    side = None
    sl = 0
    tgt = 0
    shares = 0
    sl_moved_to_be = False
    partial_exited = False
    trade_type = ""
    trades = []
    trade_count = 0
    cooldown_until = 0

    def close_trade(exit_price, exit_time, reason, shares_to_close=None):
        nonlocal in_trade, trade_count, sl_moved_to_be, partial_exited, shares
        s = shares_to_close or shares
        if side == "SHORT":
            gross = (entry_price - exit_price) * s
        else:
            gross = (exit_price - entry_price) * s
        costs = cost_model.calculate(entry_price * s, exit_price * s).total
        net = gross - costs
        trades.append({
            "symbol": symbol, "side": side, "strategy": "V2_PRO",
            "type": trade_type, "entry": round(entry_price, 2),
            "exit": round(exit_price, 2),
            "entry_time": str(entry_time), "exit_time": str(exit_time),
            "qty": s, "net_pnl": round(net, 2), "reason": reason,
        })
        if shares_to_close and shares_to_close < shares:
            shares -= shares_to_close
            partial_exited = True
        else:
            in_trade = False
            trade_count += 1
            sl_moved_to_be = False
            partial_exited = False

    for i in range(len(candles)):
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        h, l, c = row["high"], row["low"], row["close"]
        hour, minute = t.hour, t.minute

        # ORB formation
        if orb is None:
            if i < strategy.orb_candles:
                continue
            orb = strategy.compute_orb(candles, strategy.orb_candles)
            if orb is None or orb["range"] < 0.5:
                continue

        # Square off
        if hour >= strategy.square_off_hour and minute >= strategy.square_off_minute:
            if in_trade:
                close_trade(c, t, "SQUARE_OFF")
            continue

        # Manage open trade
        if in_trade:
            # Time-decay
            if (i - entry_idx) >= strategy.time_decay_candles:
                pct_from_entry = abs(c - entry_price) / entry_price
                if pct_from_entry < 0.003:
                    close_trade(c, t, "TIME_DECAY")
                    continue

            if side == "LONG":
                if l <= sl:
                    close_trade(sl, t, "STOP_LOSS")
                    cooldown_until = i + strategy.cooldown_candles
                    continue
                if h >= tgt:
                    close_trade(tgt, t, "TARGET")
                    continue
                # Partial exit
                if not partial_exited:
                    initial_risk = entry_price - sl
                    if initial_risk > 0 and (c - entry_price) >= initial_risk * strategy.partial_exit_at_rr:
                        partial_shares = max(1, int(shares * strategy.partial_exit_pct))
                        if partial_shares < shares:
                            close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)
                # BE trail
                unrealised = c - entry_price
                initial_risk = entry_price - sl
                if initial_risk > 0 and unrealised >= initial_risk and not sl_moved_to_be:
                    sl = entry_price + 0.10
                    sl_moved_to_be = True
            else:  # SHORT
                if h >= sl:
                    close_trade(sl, t, "STOP_LOSS")
                    cooldown_until = i + strategy.cooldown_candles
                    continue
                if l <= tgt:
                    close_trade(tgt, t, "TARGET")
                    continue
                if not partial_exited:
                    initial_risk = sl - entry_price
                    if initial_risk > 0 and (entry_price - c) >= initial_risk * strategy.partial_exit_at_rr:
                        partial_shares = max(1, int(shares * strategy.partial_exit_pct))
                        if partial_shares < shares:
                            close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)
                unrealised = entry_price - c
                initial_risk = sl - entry_price
                if initial_risk > 0 and unrealised >= initial_risk and not sl_moved_to_be:
                    sl = entry_price - 0.10
                    sl_moved_to_be = True
            continue

        # New entry
        if trade_count >= 2:
            continue
        if i < cooldown_until:
            continue

        # ORB breakout (9:30-10:30)
        if hour < 10 or (hour == 10 and minute <= 30):
            signal = strategy.generate_orb_signal(candles, i, orb, direction)
            if signal:
                entry_price = signal["entry"]
                entry_time = signal["time"]
                entry_idx = i
                side = signal["side"]
                sl = signal["sl"]
                tgt = signal["tgt"]
                trade_type = signal["type"]
                risk = signal["risk"]
                shares = max(1, int(capital * strategy.max_risk_pct / max(risk, 0.01)))
                in_trade = True
                sl_moved_to_be = False
                partial_exited = False
                continue

        # Pullback (10:00-11:30)
        if 10 <= hour <= 11:
            signal = strategy.generate_pullback_signal(candles, i, orb, direction)
            if signal:
                entry_price = signal["entry"]
                entry_time = signal["time"]
                entry_idx = i
                side = signal["side"]
                sl = signal["sl"]
                tgt = signal["tgt"]
                trade_type = signal["type"]
                risk = signal["risk"]
                shares = max(1, int(capital * strategy.max_risk_pct / max(risk, 0.01)))
                in_trade = True
                sl_moved_to_be = False
                partial_exited = False
                continue

        # VWAP (10:30-14:00)
        if hour >= 10 and (hour < 14 or (hour == 14 and minute == 0)):
            signal = strategy.generate_vwap_signal(candles, i, direction)
            if signal:
                entry_price = signal["entry"]
                entry_time = signal["time"]
                entry_idx = i
                side = signal["side"]
                sl = signal["sl"]
                tgt = signal["tgt"]
                trade_type = signal["type"]
                risk = signal["risk"]
                shares = max(1, int(capital * strategy.max_risk_pct / max(risk, 0.01)))
                in_trade = True
                sl_moved_to_be = False
                partial_exited = False
                continue

    return trades


# ════════════════════════════════════════════════════════════
# V3 CONCEPT — Adaptive All-Day Strategy
# ════════════════════════════════════════════════════════════
# Key improvements over V2:
#   - Never sits out: keeps scanning even if initial scores are low
#   - Multiple algo windows throughout the day
#   - Momentum regime detection per hour
#   - VWAP + RSI mean-reversion in afternoon sessions
#   - Dynamic score re-evaluation mid-day with fresh candles

def run_v3_on_candles(candles, symbol, direction, config):
    """
    V3 Adaptive: Works the WHOLE DAY. Even low-score stocks get re-evaluated.
    Algo windows:
      - 09:20-10:30 : ORB breakout (like V2 but with tighter ADR filter)
      - 10:00-11:30 : Pullback entries
      - 10:30-14:00 : VWAP mean-reversion
      - 11:30-13:00 : Midday momentum scan (V3 NEW)
      - 13:00-15:00 : Afternoon reversal scan (V3 NEW)
    """
    if candles is None or len(candles) < 10:
        return []

    cost_model = ZerodhaCostModel()
    strategy = ProStrategyV2(config.get("strategies", {}).get("pro", {}))
    capital = config["capital"]["total"]

    orb = None
    in_trade = False
    entry_price = 0
    entry_time = None
    entry_idx = 0
    side = None
    sl = 0
    tgt = 0
    shares = 0
    sl_moved_to_be = False
    partial_exited = False
    trade_type = ""
    trades = []
    trade_count = 0
    cooldown_until = 0
    max_trades = 3  # V3 allows more trades per stock

    def close_trade(exit_price, exit_time, reason, shares_to_close=None):
        nonlocal in_trade, trade_count, sl_moved_to_be, partial_exited, shares
        s = shares_to_close or shares
        if side == "SHORT":
            gross = (entry_price - exit_price) * s
        else:
            gross = (exit_price - entry_price) * s
        costs = cost_model.calculate(entry_price * s, exit_price * s).total
        net = gross - costs
        trades.append({
            "symbol": symbol, "side": side, "strategy": "V3_ADAPTIVE",
            "type": trade_type, "entry": round(entry_price, 2),
            "exit": round(exit_price, 2),
            "entry_time": str(entry_time), "exit_time": str(exit_time),
            "qty": s, "net_pnl": round(net, 2), "reason": reason,
        })
        if shares_to_close and shares_to_close < shares:
            shares -= shares_to_close
            partial_exited = True
        else:
            in_trade = False
            trade_count += 1
            sl_moved_to_be = False
            partial_exited = False

    def enter_trade(signal, idx):
        nonlocal entry_price, entry_time, entry_idx, side, sl, tgt, trade_type
        nonlocal shares, in_trade, sl_moved_to_be, partial_exited
        entry_price = signal["entry"]
        entry_time = signal["time"]
        entry_idx = idx
        side = signal["side"]
        sl = signal["sl"]
        tgt = signal["tgt"]
        trade_type = signal["type"]
        risk = signal["risk"]
        shares = max(1, int(capital * strategy.max_risk_pct / max(risk, 0.01)))
        in_trade = True
        sl_moved_to_be = False
        partial_exited = False

    # ── V3 NEW: Midday momentum detector ──
    def detect_midday_momentum(candles, idx, min_candles=6):
        """Detect strong directional momentum in 11:30-13:00 window."""
        if idx < min_candles:
            return None
        recent = candles.iloc[max(0, idx - min_candles):idx + 1]
        price_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]
        avg_vol = candles["volume"].iloc[max(0, idx - 20):idx].mean()
        curr_vol = candles["volume"].iloc[idx]
        vol_ok = avg_vol > 0 and curr_vol > avg_vol * 1.2

        row = candles.iloc[idx]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])

        if abs(price_change) > 0.008 and vol_ok:  # 0.8%+ move with volume
            if price_change > 0:
                sl_price = close - abs(price_change) * close * 0.5
                risk = close - sl_price
                tgt_price = close + risk * 1.5
                return {
                    "side": "LONG", "entry": close, "sl": sl_price,
                    "tgt": tgt_price, "risk": risk, "time": t,
                    "type": "MIDDAY_MOMENTUM",
                    "reason": f"Midday momentum +{price_change*100:.1f}% with vol"
                }
            else:
                sl_price = close + abs(price_change) * close * 0.5
                risk = sl_price - close
                tgt_price = close - risk * 1.5
                return {
                    "side": "SHORT", "entry": close, "sl": sl_price,
                    "tgt": tgt_price, "risk": risk, "time": t,
                    "type": "MIDDAY_MOMENTUM",
                    "reason": f"Midday momentum {price_change*100:.1f}% with vol"
                }
        return None

    # ── V3 NEW: Afternoon reversal detector ──
    def detect_afternoon_reversal(candles, idx):
        """Detect mean-reversion opportunity in 13:00-15:00 window using RSI + VWAP."""
        if idx < 20:
            return None
        row = candles.iloc[idx]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])

        rsi = strategy.compute_rsi(candles["close"].iloc[:idx + 1], period=7)  # faster RSI
        vwap, std = strategy.compute_vwap_proper(candles, idx)
        if std == 0:
            return None

        deviation = (close - vwap) / std

        # Oversold bounce
        if rsi < 25 and deviation < -1.5:
            sl_price = close - std * 1.0
            risk = close - sl_price
            if risk <= 0:
                return None
            return {
                "side": "LONG", "entry": close, "sl": sl_price,
                "tgt": vwap, "risk": risk, "time": t,
                "type": "AFTERNOON_REVERSAL",
                "reason": f"Afternoon RSI={rsi:.0f} at {deviation:.1f}σ below VWAP"
            }

        # Overbought fade
        if rsi > 75 and deviation > 1.5:
            sl_price = close + std * 1.0
            risk = sl_price - close
            if risk <= 0:
                return None
            return {
                "side": "SHORT", "entry": close, "sl": sl_price,
                "tgt": vwap, "risk": risk, "time": t,
                "type": "AFTERNOON_REVERSAL",
                "reason": f"Afternoon RSI={rsi:.0f} at {deviation:.1f}σ above VWAP"
            }
        return None

    # ── Main candle loop ──
    for i in range(len(candles)):
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        h, l, c = row["high"], row["low"], row["close"]
        hour, minute = t.hour, t.minute

        # ORB formation
        if orb is None:
            if i < strategy.orb_candles:
                continue
            orb = strategy.compute_orb(candles, strategy.orb_candles)
            if orb is None or orb["range"] < 0.5:
                continue

        # Square off
        if hour >= 15 and minute >= 10:
            if in_trade:
                close_trade(c, t, "SQUARE_OFF")
            continue

        # ── Manage open trade (same as V2) ──
        if in_trade:
            if (i - entry_idx) >= strategy.time_decay_candles:
                pct_from_entry = abs(c - entry_price) / entry_price
                if pct_from_entry < 0.003:
                    close_trade(c, t, "TIME_DECAY")
                    continue

            if side == "LONG":
                if l <= sl:
                    close_trade(sl, t, "STOP_LOSS")
                    cooldown_until = i + strategy.cooldown_candles
                    continue
                if h >= tgt:
                    close_trade(tgt, t, "TARGET")
                    continue
                if not partial_exited:
                    initial_risk = entry_price - sl
                    if initial_risk > 0 and (c - entry_price) >= initial_risk * strategy.partial_exit_at_rr:
                        partial_shares = max(1, int(shares * strategy.partial_exit_pct))
                        if partial_shares < shares:
                            close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)
                unrealised = c - entry_price
                initial_risk = entry_price - sl
                if initial_risk > 0 and unrealised >= initial_risk and not sl_moved_to_be:
                    sl = entry_price + 0.10
                    sl_moved_to_be = True
            else:
                if h >= sl:
                    close_trade(sl, t, "STOP_LOSS")
                    cooldown_until = i + strategy.cooldown_candles
                    continue
                if l <= tgt:
                    close_trade(tgt, t, "TARGET")
                    continue
                if not partial_exited:
                    initial_risk = sl - entry_price
                    if initial_risk > 0 and (entry_price - c) >= initial_risk * strategy.partial_exit_at_rr:
                        partial_shares = max(1, int(shares * strategy.partial_exit_pct))
                        if partial_shares < shares:
                            close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)
                unrealised = entry_price - c
                initial_risk = sl - entry_price
                if initial_risk > 0 and unrealised >= initial_risk and not sl_moved_to_be:
                    sl = entry_price - 0.10
                    sl_moved_to_be = True
            continue

        # ── Entry scanning — ALL DAY LONG ──
        if trade_count >= max_trades:
            continue
        if i < cooldown_until:
            continue

        # Window 1: ORB breakout (9:20-10:30)
        if hour < 10 or (hour == 10 and minute <= 30):
            signal = strategy.generate_orb_signal(candles, i, orb, direction)
            if signal:
                enter_trade(signal, i)
                continue

        # Window 2: Pullback (10:00-11:30)
        if 10 <= hour <= 11:
            signal = strategy.generate_pullback_signal(candles, i, orb, direction)
            if signal:
                enter_trade(signal, i)
                continue

        # Window 3: VWAP (10:30-14:00)
        if hour >= 10 and (hour < 14 or (hour == 14 and minute == 0)):
            signal = strategy.generate_vwap_signal(candles, i, direction)
            if signal:
                enter_trade(signal, i)
                continue

        # Window 4 (V3 NEW): Midday momentum (11:30-13:00)
        if (hour == 11 and minute >= 30) or hour == 12:
            signal = detect_midday_momentum(candles, i)
            if signal:
                enter_trade(signal, i)
                continue

        # Window 5 (V3 NEW): Afternoon reversal (13:00-15:00)
        if 13 <= hour <= 14:
            signal = detect_afternoon_reversal(candles, i)
            if signal:
                enter_trade(signal, i)
                continue

    return trades


# ════════════════════════════════════════════════════════════
# MAIN — Run all 3 strategies, compare
# ════════════════════════════════════════════════════════════

def find_last_trading_day(target_date=None):
    """Find the last trading day with actual data (API-based, not hardcoded)."""
    d = target_date or (date.today() - timedelta(days=1))
    for attempt in range(10):
        is_open, reason = check_market_open_live(d)
        if is_open:
            return d
        logger.info(f"  {d} ({d.strftime('%A')}): {reason}")
        d -= timedelta(days=1)
    return None


def main():
    parser = argparse.ArgumentParser(description="Strategy Comparison V1 vs V2 vs V3")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--stocks", nargs="+", default=None, help="Specific stocks")
    parser.add_argument("--top", type=int, default=10, help="Top N stocks to test")
    args = parser.parse_args()

    # Load config
    for p in ["config/config_test.yaml", "config/config_prod.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                config = yaml.safe_load(f)
            break

    # ── Determine target date ──
    if args.date:
        target = date.fromisoformat(args.date)
        logger.info(f"\n  Checking if {target} had market data (live API check, no hardcoded holidays)...")
        is_open, reason = check_market_open_live(target)
        if not is_open:
            logger.warning(f"  {target}: {reason}")
            logger.info(f"  Finding last actual trading day...")
            target = find_last_trading_day(target)
            if target is None:
                logger.error("  Could not find a recent trading day!")
                return
    else:
        logger.info(f"\n  Finding last trading day (live API check)...")
        target = find_last_trading_day()
        if target is None:
            logger.error("  Could not find a recent trading day!")
            return

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  STRATEGY COMPARISON — {target} ({target.strftime('%A')})")
    logger.info(f"  Using REAL yfinance 5-min intraday data. No cheating.")
    logger.info(f"{'=' * 70}")

    # ── Step 1: ML Scoring to pick stocks + direction ──
    symbols = args.stocks or NIFTY_50[:args.top]
    logger.info(f"\n  Step 1: Scoring {len(symbols)} stocks...")
    daily_df = fetch_daily_data(symbols, target)

    featured = []
    for sym in (daily_df["symbol"].unique() if not daily_df.empty else []):
        sdf = daily_df[daily_df["symbol"] == sym].copy()
        if len(sdf) > 50:
            featured.append(add_features(sdf))

    picks = []
    if featured:
        all_feat = pd.concat(featured, ignore_index=True)
        model, feats = None, None
        mp = Path("models/stock_predictor.pkl")
        if mp.exists():
            with open(mp, "rb") as f:
                d = pickle.load(f)
            model, feats = d["model"], d["features"]
        avail = [c for c in (feats or FEATURE_COLS) if c in all_feat.columns]
        scores = score_stocks(all_feat, model, avail if model else None)
        # V3 approach: don't filter by score — take ALL stocks, even low scores
        picks = scores.head(args.top)
        logger.info(f"\n  Stock Scores:")
        for _, r in picks.iterrows():
            d = r.get("direction", "LONG")
            arrow = "📈" if d == "LONG" else "📉"
            logger.info(f"    {arrow} {r['symbol']:<12} Score: {r['score']:>5.1f} {d} RSI={r['rsi']:.1f}")
    else:
        logger.warning("  No daily data — will use default LONG direction for all")
        picks = pd.DataFrame([{"symbol": s, "score": 50, "direction": "LONG", "rsi": 50} for s in symbols])

    # ── Step 2: Fetch 5-min candles for each stock ──
    logger.info(f"\n  Step 2: Fetching 5-min intraday candles for {target}...")
    all_candles = {}
    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        candles = fetch_5min_candles(sym, target)
        if candles is not None and len(candles) >= 10:
            all_candles[sym] = (candles, pick.get("direction", "LONG"))
            logger.info(f"    {sym}: {len(candles)} candles fetched")
        else:
            logger.warning(f"    {sym}: No 5-min data for {target}")

    if not all_candles:
        logger.error(f"\n  No intraday data found for {target}. Market was likely closed.")
        return

    # ── Step 3: Run all 3 strategies ──
    logger.info(f"\n  Step 3: Running V1 vs V2 vs V3 on {len(all_candles)} stocks...")
    v1_trades, v2_trades, v3_trades = [], [], []

    for sym, (candles, direction) in all_candles.items():
        logger.info(f"\n  ─── {sym} ({direction}) | {len(candles)} candles ───")

        t1 = run_v1_on_candles(candles, sym, direction, config)
        v1_trades.extend(t1)
        for t in t1:
            e = "✅" if t["net_pnl"] > 0 else "❌"
            logger.info(f"    V1: {e} {t['side']} | Rs {t['entry']:,.2f} → Rs {t['exit']:,.2f} | Rs {t['net_pnl']:+,.2f} | {t['reason']}")

        t2 = run_v2_on_candles(candles, sym, direction, config)
        v2_trades.extend(t2)
        for t in t2:
            e = "✅" if t["net_pnl"] > 0 else "❌"
            logger.info(f"    V2: {e} {t['side']} ({t.get('type','')}) | Rs {t['entry']:,.2f} → Rs {t['exit']:,.2f} | Rs {t['net_pnl']:+,.2f} | {t['reason']}")

        t3 = run_v3_on_candles(candles, sym, direction, config)
        v3_trades.extend(t3)
        for t in t3:
            e = "✅" if t["net_pnl"] > 0 else "❌"
            logger.info(f"    V3: {e} {t['side']} ({t.get('type','')}) | Rs {t['entry']:,.2f} → Rs {t['exit']:,.2f} | Rs {t['net_pnl']:+,.2f} | {t['reason']}")

    # ── Step 4: Comparison Report ──
    def summarize(trades, name):
        total = sum(t["net_pnl"] for t in trades)
        wins = sum(1 for t in trades if t["net_pnl"] > 0)
        losses = len(trades) - wins
        wr = wins / len(trades) * 100 if trades else 0
        avg = total / len(trades) if trades else 0
        best = max((t["net_pnl"] for t in trades), default=0)
        worst = min((t["net_pnl"] for t in trades), default=0)
        return {
            "name": name, "trades": len(trades), "wins": wins, "losses": losses,
            "win_rate": round(wr, 1), "total_pnl": round(total, 2),
            "avg_pnl": round(avg, 2), "best": round(best, 2), "worst": round(worst, 2),
        }

    s1 = summarize(v1_trades, "V1_BASIC")
    s2 = summarize(v2_trades, "V2_PRO")
    s3 = summarize(v3_trades, "V3_ADAPTIVE")

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON REPORT — {target} ({target.strftime('%A')})")
    print(f"  Data: REAL yfinance 5-min candles | Capital: Rs {config['capital']['total']:,}")
    print(f"{'=' * 70}")
    print(f"\n  {'Strategy':<16} {'Trades':>6} {'Wins':>5} {'Losses':>6} {'WR%':>5} {'Total P&L':>12} {'Avg P&L':>10} {'Best':>10} {'Worst':>10}")
    print(f"  {'─' * 85}")
    for s in [s1, s2, s3]:
        emoji = "🏆" if s["total_pnl"] == max(s1["total_pnl"], s2["total_pnl"], s3["total_pnl"]) and s["total_pnl"] != 0 else "  "
        print(f"  {emoji}{s['name']:<14} {s['trades']:>6} {s['wins']:>5} {s['losses']:>6} {s['win_rate']:>4.0f}% {s['total_pnl']:>+11,.2f} {s['avg_pnl']:>+9,.2f} {s['best']:>+9,.2f} {s['worst']:>+9,.2f}")

    print(f"\n  V1: Basic ORB breakout, fixed 1.5 RR, no trailing, no volume check")
    print(f"  V2: ProStrategyV2 — ADR filter, volume flow, partial exit, BE trail, time-decay")
    print(f"  V3: Adaptive — All V2 + midday momentum + afternoon reversal, 3 trades/stock, never sits out")

    # Key differences
    print(f"\n  KEY DIFFERENCES:")
    if s3["trades"] > s2["trades"]:
        print(f"  ✅ V3 found {s3['trades'] - s2['trades']} MORE trade opportunities (all-day scanning)")
    if s2["win_rate"] > s1["win_rate"]:
        print(f"  ✅ V2 win rate {s2['win_rate']:.0f}% vs V1 {s1['win_rate']:.0f}% (better filtering)")
    if s3["total_pnl"] > s2["total_pnl"]:
        print(f"  ✅ V3 P&L Rs {s3['total_pnl']:+,.2f} vs V2 Rs {s2['total_pnl']:+,.2f} (more opportunities)")
    elif s2["total_pnl"] > s3["total_pnl"]:
        print(f"  ⚠️  V2 P&L Rs {s2['total_pnl']:+,.2f} > V3 Rs {s3['total_pnl']:+,.2f} (V3 extra trades hurt)")

    # Save results
    all_trades = v1_trades + v2_trades + v3_trades
    if all_trades:
        df = pd.DataFrame(all_trades)
        outpath = f"results/comparison_{target}.csv"
        df.to_csv(outpath, index=False)
        print(f"\n  📁 Full results saved: {outpath}")

    # Summary JSON
    summary = {"date": str(target), "v1": s1, "v2": s2, "v3": s3,
               "stocks_tested": len(all_candles),
               "v1_trades": v1_trades, "v2_trades": v2_trades, "v3_trades": v3_trades}
    with open(f"results/comparison_{target}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Done! Total stocks with data: {len(all_candles)}")


if __name__ == "__main__":
    main()
