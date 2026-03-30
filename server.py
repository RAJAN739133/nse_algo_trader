"""
Render Deployment — Runs trading bot + health endpoint.
ALL secrets come from Render Environment Variables (never in code).

Render Start Command: python server.py
"""
import os,sys,json,threading,logging,time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime,date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

bot_state = {"status":"starting","last_run":None,"trades_today":0,"pnl_today":0}

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status":"running","bot":bot_state,"time":datetime.now().isoformat()}).encode())
    def log_message(self,*a): pass

def send_telegram(msg):
    """Send Telegram using env vars."""
    token = os.getenv("TELEGRAM_BOT_TOKEN","")
    chat_id = os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat_id: return
    try:
        import urllib.request
        data = json.dumps({"chat_id":chat_id,"text":msg,"parse_mode":"Markdown"}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",data=data,headers={"Content-Type":"application/json"})
        urllib.request.urlopen(req,timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def run_daily_trading():
    try:
        from paper_trader import run_simulate
        bot_state["status"] = "trading"
        bot_state["last_run"] = datetime.now().isoformat()
        logger.info("Starting daily paper trade...")
        trades = run_simulate(scenario="normal")
        total_pnl = sum(t.get("net_pnl",0) for t in trades) if trades else 0
        wins = sum(1 for t in trades if t.get("net_pnl",0)>0)
        bot_state["trades_today"] = len(trades)
        bot_state["pnl_today"] = total_pnl
        if trades:
            msg = f"*Daily Paper Trade — {date.today()}*\n"
            msg += f"Trades: {len(trades)} | Won: {wins} | Lost: {len(trades)-wins}\n"
            msg += f"Net P&L: Rs {total_pnl:+,.2f}\n\n"
            for t in trades:
                e = "+" if t.get("net_pnl",0)>0 else "-"
                msg += f"{e} {t['symbol']} Rs {t.get('net_pnl',0):+,.2f} ({t.get('reason','')})\n"
        else:
            msg = f"*Daily Paper Trade — {date.today()}*\nNo trades — conditions didn't qualify.\nCapital protected."
        send_telegram(msg)
        logger.info(f"Done: {len(trades)} trades, Rs {total_pnl:+,.2f}")
        bot_state["status"] = "idle"
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        bot_state["status"] = f"error: {e}"
        send_telegram(f"*ERROR*: {e}")

def scheduler():
    # Run immediately on first deploy
    logger.info("First run — testing immediately")
    run_daily_trading()
    last_date = str(date.today())
    while True:
        time.sleep(300)
        now = datetime.now()
        today = str(date.today())
        if now.hour==9 and 15<=now.minute<=25 and last_date!=today:
            last_date = today
            run_daily_trading()

def main():
    port = int(os.getenv("PORT",10000))
    t = threading.Thread(target=scheduler, daemon=True)
    t.start()
    logger.info(f"Bot running on port {port}")
    send_telegram(f"*Bot deployed!*\nServer started at {datetime.now().strftime('%H:%M')}\nFirst paper trade running now...")
    HTTPServer(("0.0.0.0",port), HealthHandler).serve_forever()

if __name__=="__main__": main()
