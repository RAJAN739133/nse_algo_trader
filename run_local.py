"""
LOCAL DAILY RUNNER - Run this every morning from IntelliJ.

What it does:
1. Downloads latest real data from yfinance (free)
2. Runs ML model scoring on all NIFTY stocks
3. Simulates paper trades for today
4. Sends results to your Telegram
5. Saves results to results/ folder

Usage:
  python run_local.py                    # Normal run (today)
  python run_local.py --scenario crash   # Test crash day
  python run_local.py --real             # Use real yfinance data
  python run_local.py --train            # Retrain ML model first
"""
import os,sys,json,argparse,logging
from datetime import datetime,date
from pathlib import Path
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/local_{date.today()}.log")])
log = logging.getLogger(__name__)

def send_telegram(msg):
    """Send result to Telegram."""
    # Read from config or env
    import yaml
    token, chat_id = "", ""
    for cfg in ["config/config_test.yaml","config/config_prod.yaml"]:
        if Path(cfg).exists():
            with open(cfg) as f:
                c = yaml.safe_load(f) or {}
            a = c.get("alerts",{})
            if a.get("telegram_bot_token"): token = a["telegram_bot_token"]
            if a.get("telegram_chat_id"): chat_id = a["telegram_chat_id"]
    token = token or os.getenv("TELEGRAM_BOT_TOKEN","")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat_id:
        log.warning("Telegram not configured - skipping alert")
        return
    try:
        data = json.dumps({"chat_id":chat_id,"text":msg}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",data=data,
            headers={"Content-Type":"application/json"})
        urllib.request.urlopen(req,timeout=10)
        log.info("Telegram alert sent!")
    except Exception as e:
        log.error(f"Telegram failed: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",default="normal",choices=["normal","volatile","crash","rally","flat"])
    p.add_argument("--real",action="store_true",help="Use real yfinance data")
    p.add_argument("--train",action="store_true",help="Retrain ML model first")
    a = p.parse_args()

    log.info(f"\n{'='*50}")
    log.info(f"  RAJAN STOCK BOT - LOCAL RUN")
    log.info(f"  {date.today()} | {datetime.now().strftime('%H:%M')} IST")
    log.info(f"{'='*50}")

    # Step 1: Train ML model if requested
    if a.train:
        log.info("\n  Retraining ML model...")
        from data.train_pipeline import generate_synthetic, add_features, train_model, FEATURE_COLS
        from config.symbols import DEFAULT_UNIVERSE
        raw = generate_synthetic(DEFAULT_UNIVERSE[:10], years=10)
        featured = []
        import pandas as pd
        for sym in raw["symbol"].unique():
            sdf = raw[raw["symbol"]==sym].copy()
            sdf = add_features(sdf)
            featured.append(sdf)
        all_f = pd.concat(featured, ignore_index=True)
        train_model(all_f)
        log.info("  Model retrained!")

    # Step 2: Run paper trading
    log.info(f"\n  Running paper trade (scenario: {a.scenario})...")
    from paper_trader import run_simulate
    trades = run_simulate(scenario=a.scenario)

    # Step 3: Build summary
    total = sum(t.get("net_pnl",0) for t in trades) if trades else 0
    wins = sum(1 for t in trades if t.get("net_pnl",0)>0)

    if trades:
        msg = f"LOCAL Paper Trade - {date.today()}*\n"
        msg += f"Scenario: {a.scenario}\n"
        msg += f"Trades: {len(trades)} | Won: {wins} | Lost: {len(trades)-wins}\n"
        msg += f"Net P&L: Rs {total:+,.2f}\n\n"
        for t in trades:
            e = "+" if t.get("net_pnl",0)>0 else "-"
            msg += f"{e} {t['symbol']} Rs {t.get('net_pnl',0):+,.2f} ({t.get('reason','')})\n"
    else:
        msg = f"LOCAL Paper Trade - {date.today()}*\n"
        msg += f"Scenario: {a.scenario}\n"
        msg += f"No trades today - capital protected."

    # Step 4: Send to Telegram
    send_telegram(msg)

    # Step 5: Print summary
    log.info(f"\n{'='*50}")
    log.info(f"  DONE - Rs {total:+,.2f}")
    log.info(f"{'='*50}")

if __name__=="__main__": main()
