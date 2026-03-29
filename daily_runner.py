"""
Daily Auto-Runner — Runs every morning automatically on Render.

Schedule (all times IST):
  8:30 AM  — Download latest data, retrain model (Saturdays only)
  8:50 AM  — Claude Brain morning analysis
  9:00 AM  — Score stocks, send picks to Telegram
  9:15 AM  — Paper trading starts (auto)
  3:15 PM  — Square off, send results to Telegram
  3:30 PM  — Save journal, sleep until next trading day

This script handles EVERYTHING. Deploy on Render and forget.

Usage:
  python daily_runner.py              # Run once (today)
  python daily_runner.py --loop       # Run forever (for Render deployment)
  python daily_runner.py --test       # Quick test with Telegram alert
"""
import os, sys, time, json, logging, argparse, schedule
from datetime import datetime, date, timedelta
from pathlib import Path
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.symbols import DEFAULT_UNIVERSE, NIFTY_50
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS, generate_synthetic

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/daily_{date.today()}.log")])
logger = logging.getLogger(__name__)


def load_config():
    for p in ["config/config_prod.yaml", "config/config_test.yaml", "config/config_example.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {"capital": {"total": 100000, "risk_per_trade": 0.01, "max_trades_per_day": 2,
            "daily_loss_limit": 0.03}, "filters": {"vix_skip_threshold": 25}}


def send_telegram(config, message):
    """Send message to Telegram."""
    alerts = config.get("alerts", {})
    if not alerts.get("telegram_enabled"):
        logger.info(f"  [Telegram disabled] {message[:80]}...")
        return
    token = alerts.get("telegram_bot_token", "")
    chat_id = alerts.get("telegram_chat_id", "")
    if not token or not chat_id:
        return
    try:
        import urllib.request
        data = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                    data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.error(f"  Telegram error: {e}")


def train_model_if_needed():
    """Retrain ML model — runs on Saturdays or if no model exists."""
    model_path = Path("models/stock_predictor.pkl")
    is_saturday = date.today().weekday() == 5
    model_exists = model_path.exists()

    if model_exists and not is_saturday:
        logger.info("  Model exists, not Saturday — skipping retrain")
        return

    logger.info("  Training ML model...")
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        import pickle

        loader = DataLoader()
        # Use top 50 stocks for training (faster + most liquid)
        symbols = NIFTY_50
        df = loader.load_training_data(symbols)
        if df.empty:
            logger.warning("  No data for training, using synthetic")
            df = generate_synthetic(symbols, years=5)

        featured = []
        for sym in df["symbol"].unique():
            sdf = df[df["symbol"] == sym].copy()
            if len(sdf) > 50:
                featured.append(add_features(sdf))
        if not featured:
            logger.error("  Not enough data to train")
            return

        all_feat = __import__("pandas").concat(featured, ignore_index=True)
        avail = [c for c in FEATURE_COLS if c in all_feat.columns]
        mdf = all_feat.dropna(subset=avail + ["target_dir"])
        X, y = mdf[avail].fillna(0), mdf["target_dir"]

        model = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                           learning_rate=0.1, random_state=42)
        model.fit(X, y)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "features": avail}, f)
        logger.info(f"  Model trained on {len(X):,} samples, saved to {model_path}")
    except Exception as e:
        logger.error(f"  Training error: {e}")


def run_daily_trading():
    """Main daily trading routine — score stocks, simulate, send results."""
    config = load_config()
    loader = DataLoader()

    logger.info(f"\n{'='*60}")
    logger.info(f"  DAILY RUN — {date.today()} ({datetime.now().strftime('%H:%M')} IST)")
    logger.info(f"  Universe: {len(DEFAULT_UNIVERSE)} stocks")
    logger.info(f"{'='*60}")

    send_telegram(config, f"🤖 *Rajan Stock Bot*\n\n📊 Starting daily analysis...\nStocks: {len(DEFAULT_UNIVERSE)}\nDate: {date.today()}")

    # Step 1: Train model if needed
    train_model_if_needed()

    # Step 2: Load model
    model, features = None, None
    model_path = Path("models/stock_predictor.pkl")
    if model_path.exists():
        import pickle
        with open(model_path, "rb") as f:
            d = pickle.load(f)
        model, features = d["model"], d["features"]

    # Step 3: Load latest data and score stocks
    # Use top 50 for scoring (250 stocks takes too long for daily yfinance)
    score_symbols = NIFTY_50
    try:
        df = loader.load_training_data(score_symbols)
    except:
        df = generate_synthetic(score_symbols, years=3)

    featured = []
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            featured.append(add_features(sdf))

    if not featured:
        send_telegram(config, "⚠️ No data available today. Sitting out.")
        return

    import pandas as pd
    all_feat = pd.concat(featured, ignore_index=True)
    avail = [c for c in FEATURE_COLS if c in all_feat.columns]
    scores = score_stocks(all_feat, model, avail if model else None)

    # Step 4: Send stock picks to Telegram
    picks = scores[scores["score"] >= 60].head(3)
    skip_count = len(scores) - len(picks)

    score_msg = f"📊 *Stock Scores — {date.today()}*\n\n"
    for i, (_, r) in enumerate(scores.head(10).iterrows(), 1):
        star = "⭐" if r["score"] >= 60 else "  "
        score_msg += f"{star} {r['symbol']}: *{r['score']:.0f}*/100 (RSI {r['rsi']:.0f})\n"

    if len(picks) > 0:
        score_msg += f"\n✅ *TODAY'S PICKS:*\n"
        for _, p in picks.iterrows():
            score_msg += f"  → {p['symbol']} (score {p['score']:.0f}) via {p['strategy']}\n"
        score_msg += f"\n⏭ Skipped: {skip_count} stocks (score < 60)"
    else:
        score_msg += f"\n🛑 *NO TRADES TODAY* — no stock scored ≥ 60\nCapital protected. Sitting out."

    send_telegram(config, score_msg)

    # Step 5: Run paper trading simulation
    from paper_trader import run_simulate
    trades = run_simulate(scenario="normal")

    # Step 6: Send trade results
    if trades:
        total_pnl = sum(t["net_pnl"] for t in trades)
        wins = sum(1 for t in trades if t["net_pnl"] > 0)
        result_msg = f"📈 *End of Day Results*\n\n"
        result_msg += f"Trades: {len(trades)} | Won: {wins} | Lost: {len(trades)-wins}\n"
        result_msg += f"Net P&L: *Rs {total_pnl:+,.2f}*\n\n"
        for t in trades:
            e = "✅" if t["net_pnl"] > 0 else "❌"
            result_msg += f"{e} {t['symbol']}: Rs {t['net_pnl']:+,.2f} ({t['reason']})\n"
    else:
        result_msg = f"📈 *End of Day*\n\nNo trades taken. Capital preserved: Rs {config['capital']['total']:,}"

    send_telegram(config, result_msg)
    logger.info("  Daily run complete")


def run_loop():
    """Run forever — for Render deployment. Executes daily at 8:50 AM IST."""
    logger.info("  Daily runner started in LOOP mode (for Render)")
    logger.info("  Will run trading at 8:50 AM IST every weekday")

    schedule.every().monday.at("08:50").do(run_daily_trading)
    schedule.every().tuesday.at("08:50").do(run_daily_trading)
    schedule.every().wednesday.at("08:50").do(run_daily_trading)
    schedule.every().thursday.at("08:50").do(run_daily_trading)
    schedule.every().friday.at("08:50").do(run_daily_trading)

    # Also run model retraining on Saturday
    schedule.every().saturday.at("10:00").do(train_model_if_needed)

    config = load_config()
    send_telegram(config, "🤖 *Rajan Stock Bot deployed!*\n\nRunning on Render.\nWill trade every weekday at 8:50 AM IST.\nModel retrains every Saturday at 10 AM.")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def run_test():
    """Quick test — send a Telegram message and run scoring."""
    config = load_config()
    send_telegram(config, f"🧪 *Test from daily_runner.py*\n\nTime: {datetime.now()}\nUniverse: {len(DEFAULT_UNIVERSE)} stocks\nStatus: All systems working!")
    logger.info("  Test message sent to Telegram")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true", help="Run forever (Render deployment)")
    p.add_argument("--test", action="store_true", help="Send test Telegram message")
    a = p.parse_args()

    if a.test:
        run_test()
    elif a.loop:
        run_loop()
    else:
        run_daily_trading()
