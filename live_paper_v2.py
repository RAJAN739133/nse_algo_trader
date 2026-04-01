#!/usr/bin/env python3
"""
Live Paper Trader V2 — Uses ProStrategyV2 on real-time 5-min candles.

Combines:
  - Real-time 5-min candle polling from yfinance
  - ProStrategyV2 candle-by-candle logic (ADR filter, volume flow, etc.)
  - Multi-recipient Telegram alerts
  - Zerodha cost model

Usage:
  python live_paper_v2.py                        # Nifty 50
  python live_paper_v2.py --universe nifty100    # Broader
  python live_paper_v2.py --stocks HDFCBANK SBIN # Specific

Schedule: Scheduler calls this at 9:05 AM Mon-Fri.
"""
import os, sys, time, logging, json, pickle
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.symbols import NIFTY_50, get_universe
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel
from strategies.pro_strategy_v2 import ProStrategyV2

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/live_v2_{date.today()}.log"),
    ],
)
logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 50


# ════════════════════════════════════════════════════════════
# TELEGRAM
# ════════════════════════════════════════════════════════════

def load_config():
    for p in ["config/config_test.yaml", "config/config_prod.yaml", "config/config_example.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {"capital": {"total": 100000, "risk_per_trade": 0.01, "max_trades_per_day": 5, "daily_loss_limit": 0.03}}


def send_telegram(msg, config):
    alerts = config.get("alerts", {})
    if not alerts.get("telegram_enabled"):
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
# DATA FETCHING
# ════════════════════════════════════════════════════════════

def fetch_intraday_candles(symbol):
    """Fetch today's 5-min candles so far from yfinance."""
    try:
        import yfinance as yf
        df = yf.Ticker(f"{symbol}.NS").history(period="5d", interval="5m")
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["symbol"] = symbol

        # Filter to today only
        times = pd.to_datetime(df["datetime"])
        today = datetime.now().date()
        df = df[times.dt.date == today].reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning(f"  {symbol}: candle fetch failed — {e}")
        return None


def fetch_vix():
    try:
        import yfinance as yf
        vix = yf.Ticker("^INDIAVIX")
        data = vix.history(period="1d")
        if len(data) > 0:
            return round(float(data.iloc[-1]["Close"]), 2)
    except:
        pass
    return 15.0


# ════════════════════════════════════════════════════════════
# LIVE V2 TRADER — Uses ProStrategyV2 candle-by-candle
# ════════════════════════════════════════════════════════════

class LiveV2Trader:
    """
    Processes 5-min candles as they arrive using ProStrategyV2.
    Maintains state across polling intervals.
    """

    def __init__(self, symbol, direction, config):
        self.symbol = symbol
        self.direction = direction
        self.capital = config["capital"]["total"]
        self.config = config
        self.cost_model = ZerodhaCostModel()
        self.strategy = ProStrategyV2(config.get("strategies", {}).get("pro", {}))

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

        self.trades = []
        self.trade_count = 0
        self.max_trades = config["capital"].get("max_trades_per_day", 2)
        self.cooldown_until = 0

        # Track how many candles we've already processed
        self.processed_candles = 0

    def _close_trade(self, exit_price, exit_time, reason, shares_to_close=None):
        shares = shares_to_close or self.shares
        if self.side == "SHORT":
            gross = (self.entry_price - exit_price) * shares
        else:
            gross = (exit_price - self.entry_price) * shares
        costs = self.cost_model.calculate(self.entry_price * shares, exit_price * shares).total
        net = gross - costs

        et = str(self.entry_time).split("+")[0]
        xt = str(exit_time).split("+")[0]

        self.trades.append({
            "symbol": self.symbol, "direction": self.side,
            "entry": round(self.entry_price, 2), "exit": round(exit_price, 2),
            "entry_time": et, "exit_time": xt,
            "sl": round(self.sl, 2), "tgt": round(self.tgt, 2),
            "qty": shares, "gross": round(gross, 2),
            "costs": round(costs, 2), "net_pnl": round(net, 2),
            "reason": reason, "type": self.trade_type,
        })

        emoji = "✅" if net > 0 else "❌"
        et_s = str(self.entry_time).split(" ")[-1].split("+")[0][:5]
        xt_s = str(exit_time).split(" ")[-1].split("+")[0][:5]
        msg = (
            f"{emoji} {self.symbol} {self.side} ({self.trade_type})\n"
            f"  Entry: {et_s} @ Rs {self.entry_price:,.2f}\n"
            f"  Exit:  {xt_s} @ Rs {exit_price:,.2f}\n"
            f"  Qty: {shares} | P&L: Rs {net:+,.2f}\n"
            f"  Reason: {reason}"
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
        self.shares = max(1, int(self.capital * self.strategy.max_risk_pct / max(risk, 0.01)))
        self.in_trade = True
        self.sl_moved_to_be = False
        self.partial_exited = False

        side_emoji = "📈" if self.side == "LONG" else "📉"
        msg = (
            f"{side_emoji} {self.symbol} {self.side} ({self.trade_type})\n"
            f"  Entry: {signal['time'].strftime('%H:%M')} @ Rs {self.entry_price:,.2f}\n"
            f"  SL: Rs {self.sl:,.2f} | TGT: Rs {self.tgt:,.2f}\n"
            f"  Qty: {self.shares}\n"
            f"  {signal['reason']}"
        )
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)

    def process_new_candles(self, candles):
        """Process only NEW candles since last poll."""
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

        # ── Phase 1: ORB Formation ──
        if self.orb is None:
            if i < self.strategy.orb_candles:
                return
            self.orb = self.strategy.compute_orb(candles, self.strategy.orb_candles)
            if self.orb is None or self.orb["range"] < 0.5:
                return
            msg = f"📐 {self.symbol} ORB formed: High={self.orb['high']:,.2f} Low={self.orb['low']:,.2f} Range={self.orb['range']:,.2f}"
            logger.info(f"  {msg}")

        # ── Phase 2: Square off ──
        if hour >= self.strategy.square_off_hour and minute >= self.strategy.square_off_minute:
            if self.in_trade:
                self._close_trade(c, t, "SQUARE_OFF")
            return

        # ── Phase 3: Manage open trade ──
        if self.in_trade:
            # Time-decay exit
            if (i - self.entry_idx) >= self.strategy.time_decay_candles:
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
                # Partial exit
                if not self.partial_exited:
                    initial_risk = self.entry_price - self.sl
                    if initial_risk > 0 and (c - self.entry_price) >= initial_risk * self.strategy.partial_exit_at_rr:
                        partial_shares = max(1, int(self.shares * self.strategy.partial_exit_pct))
                        if partial_shares < self.shares:
                            self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)
                # Breakeven trail
                unrealised = c - self.entry_price
                initial_risk = self.entry_price - self.sl
                if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                    old_sl = self.sl
                    self.sl = self.entry_price + 0.10
                    self.sl_moved_to_be = True
                    logger.info(f"  🔒 {self.symbol} SL → breakeven Rs {self.sl:,.2f} (was Rs {old_sl:,.2f})")

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
                        partial_shares = max(1, int(self.shares * self.strategy.partial_exit_pct))
                        if partial_shares < self.shares:
                            self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)
                unrealised = self.entry_price - c
                initial_risk = self.sl - self.entry_price
                if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                    old_sl = self.sl
                    self.sl = self.entry_price - 0.10
                    self.sl_moved_to_be = True
                    logger.info(f"  🔒 {self.symbol} SL → breakeven Rs {self.sl:,.2f} (was Rs {old_sl:,.2f})")
            return

        # ── Phase 4: Look for entry ──
        if self.trade_count >= self.max_trades:
            return
        if i < self.cooldown_until:
            return

        # ORB breakout (9:30 - 10:30)
        if hour < 10 or (hour == 10 and minute <= 30):
            signal = self.strategy.generate_orb_signal(candles, i, self.orb, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return

        # Pullback (10:00 - 11:30)
        if 10 <= hour <= 11:
            signal = self.strategy.generate_pullback_signal(candles, i, self.orb, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return

        # VWAP (10:30 - 14:00)
        if hour >= 10 and (hour < 14 or (hour == 14 and minute == 0)):
            signal = self.strategy.generate_vwap_signal(candles, i, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run(symbols=None):
    config = load_config()
    symbols = symbols or NIFTY_50

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  LIVE PAPER TRADER V2 — {date.today()}")
    logger.info(f"  Strategy: ProStrategyV2 (research-backed)")
    logger.info(f"  Capital: Rs {config['capital']['total']:,} | Stocks: {len(symbols)}")
    logger.info(f"{'=' * 60}")

    send_telegram(
        f"🤖 Rajan Stock Bot V2 starting\n"
        f"📅 {date.today()}\n"
        f"💰 Capital: Rs {config['capital']['total']:,}\n"
        f"📊 Universe: {len(symbols)} stocks\n"
        f"🔬 Strategy: ProStrategyV2\n"
        f"  ADR filter | Volume flow | Adaptive R:R\n"
        f"  Time-decay | Partial exit | Engulfing boost",
        config,
    )

    # ── Pre-market: Score stocks ──
    logger.info(f"\n  Scoring stocks...")
    loader = DataLoader()
    df = loader.load_backtest_data(symbols, target_date=date.today().isoformat())
    featured = []
    for sym in (df["symbol"].unique() if not df.empty else []):
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            featured.append(add_features(sdf))

    if not featured:
        msg = "No data available for scoring. Sitting out."
        logger.error(f"  {msg}")
        send_telegram(f"⚠️ {msg}", config)
        return

    all_feat = pd.concat(featured, ignore_index=True)
    model, feats = None, None
    mp = Path("models/stock_predictor.pkl")
    if mp.exists():
        with open(mp, "rb") as f:
            d = pickle.load(f)
        model, feats = d["model"], d["features"]
        logger.info(f"  ML model loaded")

    avail = [c for c in (feats or FEATURE_COLS) if c in all_feat.columns]
    scores = score_stocks(all_feat, model, avail if model else None)
    picks = scores[scores["score"] >= SCORE_THRESHOLD].head(5)
    if picks.empty:
        picks = scores.head(3)

    logger.info(f"\n  ─── PICKS ───")
    pick_lines = []
    for _, r in picks.iterrows():
        d = r.get("direction", "LONG")
        arrow = "📈" if d == "LONG" else "📉"
        logger.info(f"  {arrow} {r['symbol']:<12} Score:{r['score']:>4.0f} {d} RSI={r['rsi']:.1f}")
        pick_lines.append(f"  {arrow} {r['symbol']}: {r['score']:.0f}/100 {d} (RSI {r['rsi']:.1f})")

    # ── Check VIX ──
    vix = fetch_vix()
    logger.info(f"  India VIX: {vix}")
    if vix > config.get("filters", {}).get("vix_skip_threshold", 25):
        msg = f"⚠️ VIX {vix} too high — NO TRADING TODAY"
        logger.warning(f"  {msg}")
        send_telegram(msg, config)
        return

    send_telegram(
        f"📊 Pre-market scoring done\n"
        f"VIX: {vix}\n\n"
        f"Today's picks:\n" + "\n".join(pick_lines),
        config,
    )

    # ── Create traders for each pick ──
    traders = {}
    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        direction = pick.get("direction", "LONG")
        traders[sym] = LiveV2Trader(sym, direction, config)
        traders[sym].pick_direction = direction

    # ── Wait for market open ──
    now = datetime.now()
    market_open = now.replace(hour=9, minute=20, second=0, microsecond=0)
    if now < market_open:
        wait = (market_open - now).total_seconds()
        logger.info(f"\n  Waiting {wait / 60:.0f} min for market (9:20 AM)...")
        time.sleep(max(0, wait))

    # ── Main polling loop ──
    logger.info(f"\n  {'─' * 50}")
    logger.info(f"  MARKET OPEN — Polling 5-min candles")
    logger.info(f"  {'─' * 50}")

    market_close = now.replace(hour=15, minute=25, second=0, microsecond=0)
    poll_interval = 65  # slightly over 5 min — ensures new candle is ready

    last_status = datetime.now()

    while datetime.now() < market_close:
        now_t = datetime.now()

        for sym, trader in traders.items():
            try:
                candles = fetch_intraday_candles(sym)
                if candles is not None and len(candles) > trader.processed_candles:
                    new_count = len(candles) - trader.processed_candles
                    logger.info(f"  {now_t.strftime('%H:%M')} {sym}: {new_count} new candle(s) (total: {len(candles)})")
                    trader.process_new_candles(candles)
            except Exception as e:
                logger.error(f"  {sym} error: {e}")

        # Status update every 15 min
        if (now_t - last_status).total_seconds() >= 900:
            open_pos = sum(1 for t in traders.values() if t.in_trade)
            closed = sum(len(t.trades) for t in traders.values())
            total_pnl = sum(tr["net_pnl"] for t in traders.values() for tr in t.trades)
            logger.info(
                f"  ⏱ {now_t.strftime('%H:%M')} | Open: {open_pos} | Closed: {closed} | P&L: Rs {total_pnl:+,.0f}"
            )
            last_status = now_t

        time.sleep(poll_interval)

    # ── End of day ──
    all_trades = []
    for sym, trader in traders.items():
        all_trades.extend(trader.trades)

    total_pnl = sum(t["net_pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["net_pnl"] > 0)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  END OF DAY — {date.today()} — V2 Strategy")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Trades: {len(all_trades)} | Won: {wins} | Lost: {len(all_trades) - wins}")
    logger.info(f"  Net P&L: Rs {total_pnl:+,.2f}")

    for t in all_trades:
        e = "✅" if t["net_pnl"] > 0 else "❌"
        logger.info(
            f"  {e} {t['symbol']} {t['direction']} ({t['type']}) | "
            f"Rs {t['entry']:,.2f} → Rs {t['exit']:,.2f} | Rs {t['net_pnl']:+,.2f} | {t['reason']}"
        )

    # Save CSV
    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"results/live_v2_{date.today()}.csv", index=False)

    # Telegram summary
    summary = [f"🤖 Rajan Stock Bot V2 — {date.today()}"]
    summary.append(f"📊 Strategy: ProStrategyV2")
    summary.append(f"💰 Capital: Rs {config['capital']['total']:,}")
    summary.append("")
    if all_trades:
        summary.append(f"📋 Trades: {len(all_trades)} | Won: {wins} | Lost: {len(all_trades) - wins}")
        summary.append("───────────────────────────")
        for t in all_trades:
            emoji = "✅" if t["net_pnl"] > 0 else "❌"
            summary.append(f"{emoji} {t['symbol']} {t['direction']} ({t['type']})")
            summary.append(f"  Rs {t['entry']:,.2f} → Rs {t['exit']:,.2f}")
            summary.append(f"  Qty: {t['qty']} | P&L: Rs {t['net_pnl']:+,.2f} | {t['reason']}")
            summary.append("")
        summary.append("───────────────────────────")
        summary.append(f"💰 Net P&L: Rs {total_pnl:+,.2f}")
        wr = wins / len(all_trades) * 100 if all_trades else 0
        summary.append(f"🎯 Win rate: {wr:.0f}%")
    else:
        summary.append("📭 No trades — all filtered by V2 strategy")
    summary.append("")
    summary.append("🔬 V2 filters active:")
    summary.append("  ADR check | Volume flow | Adaptive R:R")
    summary.append("  Time-decay | Partial exit | Engulfing boost")

    send_telegram("\n".join(summary), config)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║  LIVE PAPER TRADER V2 — ProStrategyV2            ║
    ╠══════════════════════════════════════════════════╣
    ║  Real 5-min candles from yfinance                ║
    ║  V2 Strategy: ADR, volume flow, adaptive R:R     ║
    ║  Alerts to Telegram (Rajan + Gourav)             ║
    ║  Results saved to results/                       ║
    ║                                                  ║
    ║  Press Ctrl+C anytime to stop safely             ║
    ╚══════════════════════════════════════════════════╝
    """)
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--universe", default="nifty50", choices=["nifty50", "nifty100", "nifty250"])
        parser.add_argument("--stocks", nargs="+", default=None)
        args = parser.parse_args()

        if args.stocks:
            run(args.stocks)
        else:
            run(get_universe(args.universe))
    except KeyboardInterrupt:
        print("\n  Stopped by user. Check results/ for any saved trades.")
