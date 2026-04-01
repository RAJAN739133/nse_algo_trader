"""
Live Paper Trader — Run during market hours with REAL prices, ZERO money.

This fetches REAL stock prices from yfinance every 60 seconds,
runs your ORB + VWAP strategies, and logs trades to Telegram.
No Zerodha account needed. No money at risk.

Usage (run at 9:10 AM before market opens):
  python live_paper.py

What it does:
  9:10  — Downloads yesterday's data, runs ML scoring, picks stocks
  9:15  — Market opens. Starts polling prices every 60s
  9:30  — ORB range formed. Checks for breakout signals
  9:30-10:30 — ORB trading window (buys breakouts)
  Continuous — Monitors stops, trailing, targets
  14:00-15:10 — VWAP window (mean reversion on calm days)
  15:10 — Square off everything
  15:15 — Daily summary → Telegram + results/ folder
"""
import os, sys, time, logging, json, pickle
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.symbols import DEFAULT_UNIVERSE
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/live_paper_{date.today()}.log")])
logger = logging.getLogger(__name__)


def load_config():
    for p in ["config/config_test.yaml", "config/config_example.yaml"]:
        if Path(p).exists():
            with open(p) as f: return yaml.safe_load(f)
    return {"capital": {"total": 100000, "risk_per_trade": 0.01, "max_trades_per_day": 2, "daily_loss_limit": 0.03},
            "filters": {"max_gap_percent": 3.0, "vix_skip_threshold": 25},
            "strategies": {"orb": {"atr_stop_multiplier": 1.5, "trail_after_rr": 1.5}, "vwap": {"entry_band": 1.0, "stop_band": 2.0}}}


def fetch_live_price(symbol):
    """Get current price from yfinance (15-min delay in free tier)."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(period="1d", interval="1m")
        if len(data) > 0:
            latest = data.iloc[-1]
            return {
                "symbol": symbol,
                "price": round(float(latest["Close"]), 2),
                "high": round(float(data["High"].max()), 2),
                "low": round(float(data["Low"].min()), 2),
                "open": round(float(data.iloc[0]["Open"]), 2),
                "volume": int(data["Volume"].sum()),
                "time": str(data.index[-1]),
            }
        # Fallback to daily
        info = ticker.fast_info
        return {"symbol": symbol, "price": round(float(info.last_price), 2),
                "high": round(float(info.day_high), 2), "low": round(float(info.day_low), 2),
                "open": round(float(info.open), 2), "volume": int(info.last_volume or 0)}
    except Exception as e:
        logger.warning(f"  Price fetch failed for {symbol}: {e}")
        return None


def fetch_vix():
    """Get India VIX."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^INDIAVIX")
        data = vix.history(period="1d")
        if len(data) > 0:
            return round(float(data.iloc[-1]["Close"]), 2)
    except: pass
    return 15.0  # default if fetch fails


def send_telegram(msg, config):
    """Send alert to Telegram (supports multiple recipients)."""
    alerts = config.get("alerts", {})
    if not alerts.get("telegram_enabled"): return
    token = alerts.get("telegram_bot_token", "")
    chat_ids = alerts.get("telegram_chat_ids", [])
    single_id = alerts.get("telegram_chat_id", "")
    if single_id and single_id not in chat_ids:
        chat_ids.append(single_id)
    if not token or not chat_ids: return
    try:
        import urllib.request
        for chat_id in chat_ids:
            if not chat_id:
                continue
            data = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
            req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                         data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


class LivePaperTrader:
    def __init__(self, config):
        self.config = config
        self.capital = config["capital"]["total"]
        self.positions = {}
        self.closed = []
        self.pnl = 0
        self.trade_count = 0
        self.max_trades = config["capital"]["max_trades_per_day"]
        self.cost_model = ZerodhaCostModel()
        self.orb_ranges = {}   # symbol → {high, low}
        self.orb_done = set()

    def buy(self, sym, price, qty, sl, target, strat):
        if sym in self.positions or self.trade_count >= self.max_trades:
            return False
        if self.pnl < -self.capital * self.config["capital"]["daily_loss_limit"]:
            logger.warning(f"  Daily loss limit hit! No more trades.")
            return False
        cost = self.cost_model.total_cost(price, qty, "intraday")
        self.positions[sym] = {"entry": price, "qty": qty, "sl": sl, "target": target,
                               "strat": strat, "side": "LONG",
                               "time": datetime.now().strftime("%H:%M"), "cost": cost, "high": price, "low": price}
        self.trade_count += 1
        msg = f"📈 {sym} | BUY @ Rs {price:,.2f} x{qty} | SL Rs {sl:,.2f} | TGT Rs {target:,.2f}"
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)
        return True

    def short_sell(self, sym, price, qty, sl, target, strat):
        """Enter a SHORT position — sell first, buy later to cover."""
        if sym in self.positions or self.trade_count >= self.max_trades:
            return False
        if self.pnl < -self.capital * self.config["capital"]["daily_loss_limit"]:
            logger.warning(f"  Daily loss limit hit! No more trades.")
            return False
        cost = self.cost_model.total_cost(price, qty, "intraday")
        self.positions[sym] = {"entry": price, "qty": qty, "sl": sl, "target": target,
                               "strat": strat, "side": "SHORT",
                               "time": datetime.now().strftime("%H:%M"), "cost": cost, "high": price, "low": price}
        self.trade_count += 1
        msg = f"📉 {sym} | SHORT @ Rs {price:,.2f} x{qty} | SL Rs {sl:,.2f} | TGT Rs {target:,.2f}"
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)
        return True

    def sell(self, sym, price, reason):
        """Exit a LONG position."""
        if sym not in self.positions: return
        p = self.positions.pop(sym)
        gross = (price - p["entry"]) * p["qty"]
        costs = p["cost"] + self.cost_model.total_cost(price, p["qty"], "intraday")
        net = gross - costs
        self.pnl += net
        self.closed.append({"symbol": sym, "strategy": p["strat"], "side": "LONG", "entry": p["entry"],
                            "exit": round(price, 2), "qty": p["qty"], "entry_time": p["time"],
                            "exit_time": datetime.now().strftime("%H:%M"),
                            "gross": round(gross, 2), "costs": round(costs, 2), "net_pnl": round(net, 2), "reason": reason})
        emoji = "✅" if net > 0 else "❌"
        msg = f"{emoji} {sym} | BUY Rs {p['entry']:,.2f} → SELL Rs {price:,.2f} | P&L: Rs {net:+,.2f} | {reason}"
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)

    def cover(self, sym, price, reason):
        """Exit a SHORT position — buy to cover."""
        if sym not in self.positions: return
        p = self.positions.pop(sym)
        gross = (p["entry"] - price) * p["qty"]  # profit when price falls
        costs = p["cost"] + self.cost_model.total_cost(price, p["qty"], "intraday")
        net = gross - costs
        self.pnl += net
        self.closed.append({"symbol": sym, "strategy": p["strat"], "side": "SHORT", "entry": p["entry"],
                            "exit": round(price, 2), "qty": p["qty"], "entry_time": p["time"],
                            "exit_time": datetime.now().strftime("%H:%M"),
                            "gross": round(gross, 2), "costs": round(costs, 2), "net_pnl": round(net, 2), "reason": reason})
        emoji = "✅" if net > 0 else "❌"
        msg = f"{emoji} {sym} | SHORT Rs {p['entry']:,.2f} → COVER Rs {price:,.2f} | P&L: Rs {net:+,.2f} | {reason}"
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)

    def close_position(self, sym, price, reason):
        """Close any position (LONG or SHORT) based on its side."""
        if sym not in self.positions: return
        if self.positions[sym]["side"] == "SHORT":
            self.cover(sym, price, reason)
        else:
            self.sell(sym, price, reason)

    def check_stops(self, quotes):
        for sym in list(self.positions):
            q = quotes.get(sym)
            if not q: continue
            p = self.positions[sym]
            px = q["price"]

            if p["side"] == "LONG":
                if px > p["high"]: p["high"] = px
                if px <= p["sl"]:
                    self.sell(sym, p["sl"], "STOP_LOSS")
                elif px >= p["target"]:
                    self.sell(sym, p["target"], "TARGET")
                elif p["high"] > p["entry"] * 1.01:
                    trail = p["entry"] + (p["high"] - p["entry"]) * 0.5
                    if px <= trail and trail > p["sl"]:
                        self.sell(sym, px, "TRAILING")
            else:  # SHORT
                if px < p["low"]: p["low"] = px
                if px >= p["sl"]:
                    self.cover(sym, p["sl"], "STOP_LOSS")
                elif px <= p["target"]:
                    self.cover(sym, p["target"], "TARGET")
                elif p["low"] < p["entry"] * 0.99:
                    trail = p["entry"] - (p["entry"] - p["low"]) * 0.5
                    if px >= trail and trail < p["sl"]:
                        self.cover(sym, px, "TRAILING")

    def check_orb(self, sym, quote):
        if sym in self.orb_done or sym in self.positions: return
        orb = self.orb_ranges.get(sym)
        if not orb: return
        px = quote["price"]
        rng = orb["high"] - orb["low"]
        if rng < 1: return
        atr = rng * self.config["strategies"]["orb"].get("atr_stop_multiplier", 1.5)

        # LONG breakout: price above ORB high
        if px > orb["high"] * 1.001:
            sl = px - atr
            target = px + atr * 1.5
            risk = px - sl
            qty = max(1, int(self.capital * self.config["capital"]["risk_per_trade"] / risk))
            if self.buy(sym, px, qty, sl, target, "ORB_v2"):
                self.orb_done.add(sym)

        # SHORT breakdown: price below ORB low
        elif px < orb["low"] * 0.999:
            sl = px + atr
            target = px - atr * 1.5
            risk = sl - px
            qty = max(1, int(self.capital * self.config["capital"]["risk_per_trade"] / risk))
            if self.short_sell(sym, px, qty, sl, target, "ORB_v2_SHORT"):
                self.orb_done.add(sym)

    def check_vwap(self, sym, quote, vix):
        if sym in self.positions or vix > 18: return
        # Simplified VWAP check using current price vs open
        px = quote["price"]
        opn = quote.get("open", px)
        hi = quote.get("high", px)
        lo = quote.get("low", px)
        mid = (hi + lo) / 2  # rough VWAP proxy
        band = (hi - lo) * 0.5
        if band < 1: return

        # LONG: oversold bounce — price below lower band
        if px < mid - band * 0.5:
            sl = mid - band
            target = mid
            risk = px - sl
            if risk > 0:
                qty = max(1, int(self.capital * self.config["capital"]["risk_per_trade"] / risk))
                self.buy(sym, px, qty, sl, target, "VWAP_v2")

        # SHORT: overbought rejection — price above upper band
        elif px > mid + band * 0.5:
            sl = mid + band
            target = mid
            risk = sl - px
            if risk > 0:
                qty = max(1, int(self.capital * self.config["capital"]["risk_per_trade"] / risk))
                self.short_sell(sym, px, qty, sl, target, "VWAP_v2_SHORT")

    def square_off(self, quotes):
        for sym in list(self.positions):
            q = quotes.get(sym)
            if q: self.close_position(sym, q["price"], "SQUARE_OFF_3:10PM")


def run(symbols=None):
    config = load_config()
    if not symbols:
        symbols = DEFAULT_UNIVERSE[:10]
    trader = LivePaperTrader(config)

    logger.info(f"\n{'='*60}")
    logger.info(f"  LIVE PAPER TRADER — {date.today()}")
    logger.info(f"  Capital: Rs {config['capital']['total']:,} | Stocks: {len(symbols)}")
    logger.info(f"  Mode: PAPER (no real money)")
    logger.info(f"{'='*60}")

    # ── Pre-market: Score stocks ──
    logger.info(f"\n  Loading ML model + scoring stocks...")
    model, features = None, None
    model_path = Path("models/stock_predictor.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            d = pickle.load(f)
        model, features = d["model"], d["features"]
        import datetime
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        logger.info(f"  ML model loaded (age: {age.days}d {age.seconds//3600}h)")
        if age.days > 7:
            logger.warning(f"  ⚠️ Model is {age.days} days old — consider retraining")
    else:
        logger.info(f"  No ML model found — auto-training on synthetic data...")
        try:
            from data.train_pipeline import generate_synthetic, add_features, train_model
            raw = generate_synthetic(symbols[:10], years=10)
            featured = pd.concat([add_features(raw[raw["symbol"]==s].copy()) for s in raw["symbol"].unique()])
            model, features = train_model(featured)
            logger.info(f"  ✅ Model auto-trained and saved")
        except Exception as e:
            logger.warning(f"  Model training failed: {e} — running without ML scoring")

    # ── Get VIX ──
    vix = fetch_vix()
    logger.info(f"  India VIX: {vix}")
    if vix > config["filters"]["vix_skip_threshold"]:
        msg = f"VIX {vix} too high — NO TRADING TODAY"
        logger.warning(f"  {msg}")
        send_telegram(f"⚠️ *{msg}*\nCapital protected. Sitting out.", config)
        return

    # ── Pre-market scan ──
    logger.info(f"\n  Pre-market scan (fetching real prices)...")
    candidates = []
    prev_close = {}

    for sym in symbols:
        try:
            import yfinance as yf
            hist = yf.Ticker(f"{sym}.NS").history(period="5d")
            if len(hist) >= 2:
                pc = float(hist.iloc[-2]["Close"])
                today_open = float(hist.iloc[-1]["Open"])
                gap = (today_open - pc) / pc * 100
                prev_close[sym] = pc
                if abs(gap) <= config["filters"]["max_gap_percent"]:
                    candidates.append(sym)
                    logger.info(f"    {sym:>12} | Rs {today_open:>8.2f} | Gap: {gap:>+5.1f}% | OK")
                else:
                    logger.info(f"    {sym:>12} | Rs {today_open:>8.2f} | Gap: {gap:>+5.1f}% | SKIP")
        except Exception as e:
            logger.warning(f"    {sym}: Error — {e}")

    if not candidates:
        logger.info("  No candidates passed filters. Sitting out.")
        send_telegram("No stocks passed pre-market filters. Sitting out today.", config)
        return

    send_telegram(f"🔍 *Pre-market scan complete*\nVIX: {vix}\nCandidates: {', '.join(candidates)}", config)

    # ── Wait for market if needed ──
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now < market_open:
        wait = (market_open - now).total_seconds()
        logger.info(f"\n  Waiting {wait/60:.0f} minutes for market open (9:15 AM)...")
        logger.info(f"  You can Ctrl+C to stop anytime.")
        time.sleep(max(0, wait))

    # ── Main trading loop ──
    logger.info(f"\n  {'─'*50}")
    logger.info(f"  MARKET OPEN — Trading started")
    logger.info(f"  {'─'*50}")

    orb_time = now.replace(hour=9, minute=30)  # ORB range forms by 9:30
    orb_end = now.replace(hour=10, minute=30)
    vwap_start = now.replace(hour=14, minute=0)
    vwap_end = now.replace(hour=15, minute=10)
    square_off = now.replace(hour=15, minute=10)
    market_close = now.replace(hour=15, minute=30)
    poll_interval = 60  # seconds between price checks

    orb_formed = False

    while datetime.now() < market_close:
        now = datetime.now()
        t = now.strftime("%H:%M")

        # Fetch all prices
        quotes = {}
        for sym in candidates:
            q = fetch_live_price(sym)
            if q: quotes[sym] = q

        if not quotes:
            logger.warning(f"  {t} — No prices available, retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        # ── Form ORB range at 9:30 ──
        if now >= orb_time and not orb_formed:
            for sym, q in quotes.items():
                trader.orb_ranges[sym] = {"high": q["high"], "low": q["low"]}
                logger.info(f"  ORB {sym}: High={q['high']:.2f} Low={q['low']:.2f} Range={q['high']-q['low']:.2f}")
            orb_formed = True
            send_telegram(f"📊 *ORB ranges formed*\n" + "\n".join(
                f"{s}: {r['high']:.2f}-{r['low']:.2f}" for s, r in trader.orb_ranges.items()), config)

        # ── Check stops on open positions ──
        trader.check_stops(quotes)

        # ── ORB window: 9:30-10:30 ──
        if orb_time <= now <= orb_end:
            for sym in candidates:
                if sym in quotes:
                    trader.check_orb(sym, quotes[sym])

        # ── VWAP window: 14:00-15:10 ──
        if vwap_start <= now <= vwap_end:
            for sym in candidates:
                if sym in quotes:
                    trader.check_vwap(sym, quotes[sym], vix)

        # ── Square off at 15:10 ──
        if now >= square_off and trader.positions:
            logger.info(f"\n  ⏰ 3:10 PM — Squaring off all positions")
            trader.square_off(quotes)
            break

        # Status update every 5 minutes
        if now.minute % 5 == 0 and now.second < poll_interval:
            open_pos = len(trader.positions)
            logger.info(f"  {t} | Open: {open_pos} | Closed: {len(trader.closed)} | P&L: Rs {trader.pnl:+,.2f}")

        time.sleep(poll_interval)

    # ── Daily summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"  END OF DAY — {date.today()}")
    logger.info(f"{'='*60}")

    if trader.closed:
        total = sum(t["net_pnl"] for t in trader.closed)
        wins = [t for t in trader.closed if t["net_pnl"] > 0]
        logger.info(f"  Trades: {len(trader.closed)} | Won: {len(wins)} | Lost: {len(trader.closed)-len(wins)}")
        logger.info(f"  Net P&L: Rs {total:+,.2f}")
        for t in trader.closed:
            e = "WIN " if t["net_pnl"] > 0 else "LOSS"
            logger.info(f"    {e} {t['symbol']:>12} | {t['entry']:.2f} -> {t['exit']:.2f} | Rs {t['net_pnl']:+,.2f} | {t['reason']}")

        # Save results
        pd.DataFrame(trader.closed).to_csv(f"results/live_paper_{date.today()}.csv", index=False)

        # Telegram summary
        summary = f"📊 *Daily Summary — {date.today()}*\n"
        summary += f"Trades: {len(trader.closed)} | Won: {len(wins)}\n"
        summary += f"Net P&L: Rs {total:+,.2f}\n\n"
        for t in trader.closed:
            e = "✅" if t["net_pnl"] > 0 else "❌"
            summary += f"{e} {t['symbol']} Rs {t['net_pnl']:+,.2f} ({t['reason']})\n"
        send_telegram(summary, config)
    else:
        logger.info("  No trades taken today.")
        send_telegram("📊 *No trades today* — No signals met our criteria.", config)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║  LIVE PAPER TRADER — Real Prices, Zero Money     ║
    ╠══════════════════════════════════════════════════╣
    ║  Fetches REAL NSE prices via yfinance             ║
    ║  Runs ORB + VWAP strategies                      ║
    ║  Sends alerts to Telegram                        ║
    ║  Saves results to results/ folder                ║
    ║                                                  ║
    ║  Press Ctrl+C anytime to stop safely             ║
    ╚══════════════════════════════════════════════════╝
    """)
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--universe", default="nifty50", choices=["nifty50","nifty100","nifty250"])
        args = parser.parse_args()
        from config.symbols import get_universe
        run(get_universe(args.universe))
    except KeyboardInterrupt:
        print("\n  Stopped by user. Check results/ for any saved trades.")
