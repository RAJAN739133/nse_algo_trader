#!/usr/bin/env python3
"""
PythonAnywhere Deployment Script for NSE Algo Trader

This script runs the full trading day in one execution.
Set this as your scheduled task on PythonAnywhere.

Schedule: 09:00 AM IST (03:30 UTC)

Steps to deploy:
1. Create free account at pythonanywhere.com
2. Go to "Consoles" → "Bash" → Run:
   git clone https://github.com/RAJAN739133/nse_algo_trader.git
   cd nse_algo_trader
   pip3 install --user -r requirements.txt

3. Go to "Files" → Edit config/config_prod.yaml with your API keys

4. Go to "Tasks" → Add scheduled task:
   Time: 03:30 (UTC) = 09:00 IST
   Command: cd ~/nse_algo_trader && python3 pythonanywhere_setup.py

That's it! Bot will run every trading day automatically.
"""

import os
import sys
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/pa_trading_{date.today()}.log")
    ]
)
logger = logging.getLogger(__name__)

# NSE Holidays 2026
NSE_HOLIDAYS = {
    "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28",
    "2026-06-26", "2026-08-19", "2026-10-02", "2026-10-20",
    "2026-10-22", "2026-11-05", "2026-12-25"
}


def is_trading_day():
    """Check if today is a trading day."""
    today = date.today()
    if today.weekday() >= 5:  # Weekend
        return False, "Weekend"
    if str(today) in NSE_HOLIDAYS:
        return False, "NSE Holiday"
    return True, "Trading Day"


def send_telegram(msg):
    """Send Telegram notification."""
    try:
        import yaml
        config_path = Path("config/config_prod.yaml")
        if not config_path.exists():
            config_path = Path("config/config_test.yaml")
        if not config_path.exists():
            return
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        alerts = config.get("alerts", {})
        if not alerts.get("telegram_enabled"):
            return
        
        token = alerts.get("telegram_bot_token", "")
        chat_id = alerts.get("telegram_chat_id", "")
        
        if not token or not chat_id:
            return
        
        import urllib.request
        import json
        
        data = json.dumps({"chat_id": chat_id, "text": msg}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("Telegram notification sent")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


def wait_for_market_open():
    """Wait until market opens (9:15 AM IST)."""
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    
    if now < market_open:
        wait_seconds = (market_open - now).total_seconds()
        logger.info(f"Waiting {wait_seconds/60:.0f} minutes for market open...")
        time.sleep(max(0, wait_seconds))


def run_trading():
    """Run the main trading script."""
    logger.info("Starting live_paper_v3.py...")
    
    try:
        # Import and run directly (better than subprocess on PythonAnywhere)
        import live_paper_v3
        
        # The main() function handles everything
        if hasattr(live_paper_v3, 'main'):
            live_paper_v3.main()
        else:
            # Fallback: run as script
            exec(open("live_paper_v3.py").read())
            
    except Exception as e:
        logger.error(f"Trading error: {e}")
        send_telegram(f"❌ Trading Error: {str(e)[:100]}")
        raise


def generate_report():
    """Generate and send daily report."""
    try:
        import pandas as pd
        
        results_file = Path(f"results/live_v3_{date.today()}.csv")
        
        if results_file.exists():
            df = pd.read_csv(results_file)
            total_trades = len(df)
            
            if total_trades > 0:
                wins = len(df[df["net_pnl"] > 0])
                total_pnl = df["net_pnl"].sum()
                wr = wins / total_trades * 100
                
                msg = f"""📊 PythonAnywhere Trading Report
📅 {date.today()}
━━━━━━━━━━━━━━━━━━━━
📋 Trades: {total_trades}
✅ Wins: {wins} ({wr:.0f}%)
💰 P&L: ₹{total_pnl:+,.0f}
━━━━━━━━━━━━━━━━━━━━
🤖 Automated via PythonAnywhere"""
            else:
                msg = f"📊 {date.today()} - No trades executed"
        else:
            msg = f"📊 {date.today()} - No results file found"
        
        send_telegram(msg)
        logger.info("Daily report sent")
        
    except Exception as e:
        logger.warning(f"Report error: {e}")


def main():
    """Main entry point for PythonAnywhere scheduled task."""
    
    logger.info("=" * 60)
    logger.info(f"  NSE ALGO TRADER - PythonAnywhere")
    logger.info(f"  Date: {date.today()}")
    logger.info("=" * 60)
    
    # Check if trading day
    is_trading, reason = is_trading_day()
    
    if not is_trading:
        logger.info(f"Not a trading day: {reason}")
        send_telegram(f"📅 {date.today()} - Market closed ({reason})\n🛌 Bot resting.")
        return
    
    # Notify start
    send_telegram(f"🚀 Trading bot started\n📅 {date.today()}\n☁️ Running on PythonAnywhere")
    
    try:
        # Wait for market open
        wait_for_market_open()
        
        # Run trading
        run_trading()
        
        # Generate report
        generate_report()
        
        logger.info("Trading day complete!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        send_telegram(f"❌ Fatal Error: {str(e)[:200]}")
        raise


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    main()
