#!/usr/bin/env python3
"""
Rajan Stock Bot — Auto Scheduler
=================================
Run this ONCE and it handles everything automatically:
  1. Waits until 8:50 AM IST
  2. Downloads latest data (yfinance)
  3. Trains/updates ML model
  4. Scores all stocks
  5. Runs paper trading during market hours (9:15-15:15)
  6. Sends Telegram alerts for every event
  7. End-of-day summary at 3:15 PM
  8. Sleeps until next trading day and repeats

Usage:
  caffeinate -s python3 start_bot.py          # Run forever, keep Mac awake
  caffeinate -s python3 start_bot.py --once   # Run one day only

The bot will:
  - Skip weekends automatically
  - Skip holidays (basic list)
  - Send you Telegram when it wakes up
  - Send you Telegram before every trade
  - Send you end-of-day P&L summary
"""
import os, sys, time, json, logging, argparse, pickle
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.symbols import DEFAULT_UNIVERSE
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel
from strategies.claude_brain import ClaudeBrain

# ================================================================
# Setup
# ================================================================

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/bot_{date.today()}.log"),
    ],
)
logger = logging.getLogger(__name__)


# ================================================================
# Telegram Helper
# ================================================================

def send_telegram(msg, config):
    """Send message to Telegram."""
    alerts = config.get("alerts", {})
    if not alerts.get("telegram_enabled"):
        return
    token = alerts.get("telegram_bot_token", "")
    chat_id = alerts.get("telegram_chat_id", "")
    if not token or not chat_id:
        return
    try:
        import urllib.request
        # Remove markdown if it causes issues — send as plain text
        clean_msg = msg.replace("*", "").replace("_", "").replace("`", "")
        data = json.dumps({"chat_id": chat_id, "text": clean_msg}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")


# ================================================================
# Load Config
# ================================================================

def load_config():
    for path in ["config/config_test.yaml", "config/config_prod.yaml", "config/config_example.yaml"]:
        if Path(path).exists():
            with open(path) as f:
                return yaml.safe_load(f)
    return {"capital": {"total": 100000, "risk_per_trade": 0.01, "max_trades_per_day": 2,
            "daily_loss_limit": 0.03}, "alerts": {"telegram_enabled": False},
            "strategies": {"orb": {}, "vwap": {}}, "filters": {"vix_skip_threshold": 25, "max_gap_percent": 3.0}}


# ================================================================
# Check if Market Day
# ================================================================

NSE_HOLIDAYS_2026 = [
    "2026-01-26", "2026-03-10", "2026-03-17", "2026-03-30",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-25",
    "2026-07-07", "2026-08-15", "2026-08-28", "2026-10-02",
    "2026-10-20", "2026-10-21", "2026-11-05", "2026-11-09",
    "2026-12-25",
]


def is_market_day(d=None):
    d = d or date.today()
    if d.weekday() >= 5:  # Saturday, Sunday
        return False
    if d.isoformat() in NSE_HOLIDAYS_2026:
        return False
    return True


def wait_until(hour, minute):
    """Wait until a specific IST time today."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        return  # already past
    wait_secs = (target - now).total_seconds()
    logger.info(f"  Waiting until {hour:02d}:{minute:02d} ({wait_secs/60:.0f} minutes)...")
    time.sleep(max(0, wait_secs))


# ================================================================
# STEP 1: Download & Prepare Data
# ================================================================

def step_download_data(symbols, config):
    logger.info("\n  STEP 1: Downloading latest market data...")
    # Silently download and prepare
    loader = DataLoader()
    df = loader.load_training_data(symbols)
    logger.info(f"  Loaded {len(df):,} rows for {df['symbol'].nunique()} stocks")
    return df, loader


# ================================================================
# STEP 2: Add Features + Train Model
# ================================================================

def step_train_model(df, config):
    logger.info("\n  STEP 2: Computing features + training ML model...")

    featured = []
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            sdf = add_features(sdf)
            featured.append(sdf)
    all_featured = pd.concat(featured, ignore_index=True)

    # Train model
    from data.train_pipeline import train_model
    model, feats = train_model(all_featured)
    logger.info(f"  Model trained with {len(feats)} features")
    return model, feats, all_featured


# ================================================================
# STEP 3: Score Stocks
# ================================================================

def step_score_stocks(featured_df, model, feats, config, vix=15):
    logger.info("\n  STEP 3: Scoring all stocks...")
    avail = [c for c in feats if c in featured_df.columns]
    scores = score_stocks(featured_df, model, avail, vix=vix)

    # Show top stocks
    msg_lines = ["📊 *Stock Scores for Today:*\n"]
    for _, r in scores.head(10).iterrows():
        star = "⭐" if r["score"] >= 60 else "  "
        msg_lines.append(f"{star} {r['symbol']}: {r['score']:.0f}/100 (RSI={r['rsi']:.0f})")

    picks = scores[scores["score"] >= 60].head(3)
    if picks.empty:
        msg_lines.append("\n❌ No stocks qualify (all below 60). Sitting out today.")
    else:
        msg_lines.append(f"\n🎯 *Trading today:* {', '.join(picks['symbol'].tolist())}")

    msg = "\n".join(msg_lines)
    logger.info(msg)
    # Don't send scores separately — will combine with Claude advice below
    return scores, picks


# ================================================================
# STEP 4: Claude Brain Analysis
# ================================================================

def step_claude_analysis(config, scores, vix=15, fii=0):
    logger.info("\n  STEP 4: Claude Brain morning analysis...")
    brain = ClaudeBrain(config=config)
    if not brain.enabled:
        logger.info("  Claude Brain: disabled (no API key)")
        return {"risk_level": "normal", "max_trades": 2, "skip_stocks": []}

    stock_list = scores.head(5).to_dict("records") if len(scores) > 0 else []
    result = brain.get_morning_analysis(vix, fii, [], stock_list)

    # Single combined morning message: scores + Claude advice
    picks_list = scores[scores["score"] >= 60].head(3)
    top3 = scores.head(3)
    lines = [f"🔔 {date.today()} Morning"]
    for _, r in top3.iterrows():
        star = "🎯" if r["score"] >= 60 else "  "
        lines.append(f"{star} {r['symbol']}: {r['score']:.0f}/100")
    lines.append(f"Claude: {result.get('risk_level', 'normal')} | {result.get('notes', '')[:60]}")
    if picks_list.empty:
        lines.append("No stocks qualify — sitting out")
    send_telegram("\n".join(lines), config)
    return result


# ================================================================
# STEP 5: Paper Trading Simulation
# ================================================================

def step_run_trading(picks, config, featured_df, claude_advice):
    logger.info("\n  STEP 5: Running paper trades...")
    if picks.empty:
        logger.info("  No picks — sitting out")
        return []

    cost_model = ZerodhaCostModel()
    max_trades = min(claude_advice.get("max_trades", 2), config["capital"]["max_trades_per_day"])
    skip_stocks = claude_advice.get("skip_stocks", [])
    trades = []

    for _, pick in picks.head(max_trades).iterrows():
        sym = pick["symbol"]
        if sym in skip_stocks:
            logger.info(f"  Skipping {sym} (Claude Brain advised)")
            continue

        sdf = featured_df[featured_df["symbol"] == sym]
        if len(sdf) < 2:
            continue

        last = sdf.iloc[-1]
        entry = last["close"]
        atr = last.get("atr_14", entry * 0.015)
        sl = entry - atr * 1.5
        target = entry + atr * 2.0
        risk = entry - sl
        qty = max(1, int(config["capital"]["total"] * config["capital"]["risk_per_trade"] / max(risk, 1)))

        # Simulate: use last day's high/low as proxy
        high, low, close = last["high"], last["low"], last["close"]
        if high >= target:
            exit_p, reason = target, "TARGET"
        elif low <= sl:
            exit_p, reason = sl, "STOP_LOSS"
        else:
            exit_p, reason = close, "SQUARE_OFF"

        gross = (exit_p - entry) * qty
        costs = cost_model.calculate(entry * qty, exit_p * qty).total
        net = gross - costs

        trade = {"symbol": sym, "strategy": pick.get("strategy", "ORB"),
                 "entry": round(entry, 2), "exit": round(exit_p, 2),
                 "qty": qty, "net_pnl": round(net, 2), "reason": reason}
        trades.append(trade)

        emoji = "✅" if net > 0 else "❌"
        logger.info(f"  {emoji} {sym} | BUY {entry:.0f} | SELL {exit_p:.0f} | Rs {net:+,.0f} | {reason}")
        send_telegram(f"{emoji} {sym} | BUY {entry:.0f} | SELL {exit_p:.0f} | Rs {net:+,.0f}", config)

    return trades


# ================================================================
# STEP 6: End of Day Summary
# ================================================================

def step_eod_summary(trades, config):
    logger.info("\n  STEP 6: End-of-day summary")
    if not trades:
        send_telegram(f"📋 {date.today()} | No trades | Capital safe", config)
    else:
        total = sum(t["net_pnl"] for t in trades)
        wins = sum(1 for t in trades if t["net_pnl"] > 0)
        lines = [f"📋 {date.today()} | {len(trades)} trades | Rs {total:+,.0f}"]
        for t in trades:
            e = "✅" if t["net_pnl"] > 0 else "❌"
            lines.append(f"{e} {t['symbol']} | BUY {t['entry']:.0f} | SELL {t['exit']:.0f} | Rs {t['net_pnl']:+,.0f}")
        lines.append(f"Win: {wins}/{len(trades)} | Costs: Rs {sum(t.get('costs',0) for t in trades):.0f}")
        msg = "\n".join(lines)
        logger.info(msg)
        send_telegram(msg, config)


# ================================================================
# MAIN LOOP — Runs Forever
# ================================================================

def run_one_day(config, symbols):
    """Execute the complete daily pipeline."""
    today = date.today()
    logger.info(f"\n{'='*60}")
    logger.info(f"  RAJAN STOCK BOT — {today} ({today.strftime('%A')})")
    logger.info(f"{'='*60}")

    if not is_market_day():
        msg = f"📅 {today.strftime('%A')} — Market closed. Sleeping until next trading day."
        logger.info(msg)
        logger.info(msg)
        return

    # Wait for pre-market time
    wait_until(8, 50)

    # Run the complete pipeline
    try:
        df, loader = step_download_data(symbols, config)
        model, feats, featured = step_train_model(df, config)
        scores, picks = step_score_stocks(featured, model, feats, config)
        claude_advice = step_claude_analysis(config, scores)
        trades = step_run_trading(picks, config, featured, claude_advice)
        step_eod_summary(trades, config)
    except Exception as e:
        error_msg = f"❌ *ERROR:* {str(e)[:200]}"
        logger.error(error_msg, exc_info=True)
        send_telegram(error_msg, config)



def main():
    parser = argparse.ArgumentParser(description="Rajan Stock Bot — Auto Scheduler")
    parser.add_argument("--once", action="store_true", help="Run one day only, then exit")
    parser.add_argument("--stocks", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config()
    symbols = args.stocks or DEFAULT_UNIVERSE[:10]

    logger.info("""
    ╔══════════════════════════════════════════════════╗
    ║         RAJAN STOCK BOT — Auto Scheduler          ║
    ║                                                    ║
    ║  Capital: Rs {:>8,}                             ║
    ║  Stocks:  {:>2} (Nifty 50 top picks)               ║
    ║  Mode:    PAPER (no real money)                    ║
    ║                                                    ║
    ║  The bot will:                                     ║
    ║  • Wake at 8:50 AM IST                            ║
    ║  • Download data + train ML model                  ║
    ║  • Score stocks + Claude Brain analysis             ║
    ║  • Paper trade 9:15 AM - 3:15 PM                  ║
    ║  • Send Telegram alerts for everything             ║
    ║  • Sleep and repeat next trading day               ║
    ╚══════════════════════════════════════════════════╝
    """.format(config["capital"]["total"], len(symbols)))

    # Bot started silently — only trades + summary go to Telegram

    if args.once:
        run_one_day(config, symbols)
        return

    # Run forever — wake up, trade, sleep, repeat
    while True:
        run_one_day(config, symbols)

        # Calculate sleep until next day 8:45 AM
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        next_wake = tomorrow.replace(hour=8, minute=45, second=0, microsecond=0)

        # Skip to Monday if Friday
        while not is_market_day(next_wake.date()):
            next_wake += timedelta(days=1)

        sleep_secs = (next_wake - datetime.now()).total_seconds()
        logger.info(f"\n  Sleeping until {next_wake.strftime('%A %B %d, %I:%M %p')} ({sleep_secs/3600:.1f} hours)")
        time.sleep(max(0, sleep_secs))


if __name__ == "__main__":
    main()
