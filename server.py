"""
Render Deployment Server — Runs the algo bot + health check endpoint.

Render free tier needs a web server. This runs:
1. A tiny health-check HTTP server (keeps Render happy)
2. The trading bot in a background thread (does actual work)
3. Auto paper-trades every morning, sends results to Telegram

Start command for Render: python server.py
"""
import os, sys, json, threading, logging, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Bot state
bot_state = {"status": "starting", "last_run": None, "trades_today": 0, "pnl_today": 0}


class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — Render pings this to check if service is alive."""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "running",
            "bot": bot_state,
            "time": datetime.now().isoformat(),
        }).encode())
    def log_message(self, format, *args): pass  # suppress HTTP logs


def run_daily_trading():
    """Run one day of paper trading and send results to Telegram."""
    try:
        import yaml
        from paper_trader import run_simulate
        from setup_telegram import send_alert

        # Load config
        cfg_path = Path("config/config_test.yaml")
        if not cfg_path.exists():
            cfg_path = Path("config/config_example.yaml")

        bot_state["status"] = "trading"
        bot_state["last_run"] = datetime.now().isoformat()

        # Run simulation
        logger.info("Starting daily paper trade...")
        trades = run_simulate(scenario="normal")

        # Calculate results
        total_pnl = sum(t.get("net_pnl", 0) for t in trades) if trades else 0
        wins = sum(1 for t in trades if t.get("net_pnl", 0) > 0)
        bot_state["trades_today"] = len(trades)
        bot_state["pnl_today"] = total_pnl

        # Send Telegram summary
        if trades:
            msg = f"📊 *Daily Paper Trade Summary*\n"
            msg += f"Date: {date.today()}\n"
            msg += f"Trades: {len(trades)} | Won: {wins} | Lost: {len(trades)-wins}\n"
            msg += f"Net P&L: Rs {total_pnl:+,.2f}\n\n"
            for t in trades:
                emoji = "✅" if t.get("net_pnl", 0) > 0 else "❌"
                msg += f"{emoji} {t['symbol']} Rs {t.get('net_pnl',0):+,.2f} ({t.get('reason','')})\n"
        else:
            msg = f"📊 *Daily Paper Trade Summary*\nDate: {date.today()}\nNo trades today — market conditions didn't meet criteria.\nCapital protected ✅"

        send_alert(msg)
        logger.info(f"Daily run complete: {len(trades)} trades, P&L Rs {total_pnl:+,.2f}")
        bot_state["status"] = "idle"

    except Exception as e:
        logger.error(f"Trading error: {e}", exc_info=True)
        bot_state["status"] = f"error: {e}"


def trading_scheduler():
    """Run trading at market open, then sleep until next day."""
    import pytz
    ist = pytz.timezone("Asia/Kolkata") if "pytz" in sys.modules else None

    while True:
        now = datetime.now()
        # Run at 9:20 AM IST (or immediately on first deploy for testing)
        if bot_state["last_run"] is None:
            logger.info("First run — executing immediately for testing")
            run_daily_trading()
        
        # Sleep until next 9:20 AM
        # Simple approach: check every 5 minutes
        time.sleep(300)  # 5 minutes

        hour = datetime.now().hour
        minute = datetime.now().minute
        # Run between 9:15-9:25 if we haven't run today
        if 9 <= hour <= 9 and 15 <= minute <= 25:
            if bot_state.get("last_run_date") != str(date.today()):
                bot_state["last_run_date"] = str(date.today())
                run_daily_trading()


def main():
    port = int(os.getenv("PORT", 10000))
    
    # Start trading bot in background thread
    trader_thread = threading.Thread(target=trading_scheduler, daemon=True)
    trader_thread.start()
    logger.info(f"Trading bot started in background")

    # Start health check server (Render needs this)
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health server on port {port}")
    logger.info(f"Bot is running! Check Telegram for trade alerts.")
    bot_state["status"] = "idle"
    server.serve_forever()


if __name__ == "__main__":
    main()
