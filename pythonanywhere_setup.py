#!/usr/bin/env python3
"""
Cloud Deployment Script for NSE Algo Trader
Works with: Render.com, PythonAnywhere, or any cloud platform

Schedule: 09:00 AM IST (03:30 UTC)

For Render.com (FREE):
1. Create account at render.com
2. New → Cron Job → Connect GitHub repo
3. Schedule: 30 3 * * 1-5
4. Build: pip install -r requirements.txt
5. Start: python pythonanywhere_setup.py
6. Add environment variables (see below)

Environment Variables (set in Render dashboard):
  ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PIN, ANGEL_TOTP_SECRET
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
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
        import urllib.request
        import json
        
        # Try environment variables first (for Render.com)
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        
        # Fallback to config file
        if not token or not chat_id:
            import yaml
            config_path = Path("config/config_prod.yaml")
            if not config_path.exists():
                config_path = Path("config/config_test.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                alerts = config.get("alerts", {})
                token = token or alerts.get("telegram_bot_token", "")
                chat_id = chat_id or alerts.get("telegram_chat_id", "")
        
        if not token or not chat_id:
            logger.warning("Telegram not configured")
            return
        
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


def setup_credentials_from_env():
    """Set up credentials from environment variables (for Render.com)."""
    import yaml
    
    # Check if env vars are set
    angel_api_key = os.environ.get("ANGEL_API_KEY")
    if not angel_api_key:
        logger.info("Using config files (no env vars found)")
        return
    
    logger.info("Setting up credentials from environment variables...")
    
    # Create/update angel_config.yaml
    angel_config = {
        "angel_one": {
            "api_key": os.environ.get("ANGEL_API_KEY", ""),
            "client_code": os.environ.get("ANGEL_CLIENT_CODE", ""),
            "pin": os.environ.get("ANGEL_PIN", ""),
            "totp_secret": os.environ.get("ANGEL_TOTP_SECRET", ""),
        }
    }
    
    Path("config").mkdir(exist_ok=True)
    with open("config/angel_config.yaml", "w") as f:
        yaml.dump(angel_config, f)
    
    # Create/update config_prod.yaml if telegram vars exist
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if telegram_token:
        config = {
            "capital": {
                "total": 100000,
                "risk_per_trade": 0.01,
                "max_trades_per_day": 5,
                "daily_loss_limit": 0.03
            },
            "alerts": {
                "telegram_enabled": True,
                "telegram_bot_token": telegram_token,
                "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
            }
        }
        with open("config/config_prod.yaml", "w") as f:
            yaml.dump(config, f)
    
    logger.info("Credentials configured from environment")


def run_trading():
    """Run the main trading script."""
    logger.info("Starting live_paper_v3.py...")
    
    # Setup credentials from env vars (for cloud deployment)
    setup_credentials_from_env()
    
    try:
        # Import and run directly
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
                
                msg = f"""📊 Cloud Trading Report
📅 {date.today()}
━━━━━━━━━━━━━━━━━━━━
📋 Trades: {total_trades}
✅ Wins: {wins} ({wr:.0f}%)
💰 P&L: ₹{total_pnl:+,.0f}
━━━━━━━━━━━━━━━━━━━━
🤖 Automated via Render.com"""
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
