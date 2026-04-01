"""
Full Day Orchestrator — Manages the complete trading lifecycle.

Usage:
  python scheduler.py install         # Install cron jobs
  python scheduler.py uninstall       # Remove cron jobs
  python scheduler.py status          # Check if scheduler is active
  python scheduler.py run-now         # Start trading NOW
  python scheduler.py run-day         # Full day lifecycle
  python scheduler.py post-market     # Run post-market analysis
  python scheduler.py download-data   # Download enriched data
  python scheduler.py retrain         # Force retrain ML model
"""
import os, sys, subprocess, time, logging
from pathlib import Path
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python3"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

CRON_MARKER = "# nse_algo_trader"
CRON_JOBS = f"""
# ── NSE Algo Trader V3 — Full Lifecycle ──
55 8 * * 1-5 cd {PROJECT_DIR} && {VENV_PYTHON} {PROJECT_DIR}/scheduler.py run-day >> {LOG_DIR}/cron_day.log 2>&1 {CRON_MARKER}
0 6 * * 6 cd {PROJECT_DIR} && {VENV_PYTHON} {PROJECT_DIR}/scheduler.py weekly >> {LOG_DIR}/cron_weekly.log 2>&1 {CRON_MARKER}
"""

NSE_HOLIDAYS = {
    # ── Official NSE Trading Holidays 2026 (Ref: NSE/CMTR/71775) ──
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-26",  # Shri Ram Navami
    "2026-03-31",  # Shri Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id (Eid-Ul-Adha)
    "2026-06-26",  # Muharram
    "2026-08-19",  # Janmashtami
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-10-22",  # Diwali Laxmi Pujan (half day — Muhurat Trading)
    "2026-11-05",  # Prakash Gurpurb Sri Guru Nanak Dev
    "2026-12-25",  # Christmas
}

HOLIDAY_NAMES = {
    "2026-01-26": "Republic Day",
    "2026-03-03": "Holi",
    "2026-03-26": "Shri Ram Navami",
    "2026-03-31": "Shri Mahavir Jayanti",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-28": "Bakri Id",
    "2026-06-26": "Muharram",
    "2026-08-19": "Janmashtami",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-10-22": "Diwali Laxmi Pujan",
    "2026-11-05": "Guru Nanak Jayanti",
    "2026-12-25": "Christmas",
}


def check_market_open_live(check_date=None):
    """Check if NSE was/is open by fetching live data from yfinance (API-based, no hardcoded holidays)."""
    d = check_date or date.today()
    try:
        import yfinance as yf
        for sym in ["^NSEI", "RELIANCE.NS", "SBIN.NS"]:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=d.isoformat(), end=(d + timedelta(days=1)).isoformat(), interval="1d")
            if len(df) > 0:
                return True, "Market data found"
        return False, "No market data — holiday"
    except Exception as e:
        logger.warning(f"  Live holiday check failed ({e}), falling back to hardcoded list")
        return str(d) not in NSE_HOLIDAYS, "Fallback to hardcoded"


def is_trading_day(check_date=None):
    """Check if a date is a trading day. Hardcoded holidays are authoritative for 2026."""
    d = check_date or date.today()
    if d.weekday() >= 5:
        return False
    # Hardcoded holiday list is authoritative — always check first
    if str(d) in NSE_HOLIDAYS:
        return False
    # For past dates, optionally verify via yfinance (but holidays already handled above)
    return True


def get_holiday_name(check_date=None):
    return HOLIDAY_NAMES.get(str(check_date or date.today()), "Holiday")


def run_day():
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  FULL DAY LIFECYCLE - {date.today()}")
    logger.info(f"{'=' * 60}")
    if not is_trading_day():
        reason = "Weekend" if date.today().weekday() >= 5 else get_holiday_name()
        logger.info(f"  Not a trading day ({reason}). Exiting.")
        try:
            import yaml
            config = {}
            for p in ["config/config_test.yaml", "config/config_prod.yaml"]:
                if Path(p).exists():
                    with open(p) as f:
                        config = yaml.safe_load(f)
                    break
            from live_paper_v3 import send_telegram
            send_telegram(f"📅 {date.today()} — Market closed ({reason})\n🛌 Bot resting.", config)
        except Exception:
            pass
        return

    logger.info(f"\n  === PHASE 1: PRE-MARKET ===")
    pre_market()

    logger.info(f"\n  === PHASE 2: TRADING (V3 Adaptive) ===")
    now = datetime.now()
    market_start = now.replace(hour=9, minute=5, second=0)
    if now < market_start:
        wait = (market_start - now).total_seconds()
        logger.info(f"  Waiting {wait/60:.0f} min for 09:05 AM...")
        time.sleep(max(0, wait))

    logger.info("  Starting live_paper_v3.py...")
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(PROJECT_DIR / "live_paper_v3.py")],
            cwd=str(PROJECT_DIR), timeout=7 * 3600)
        logger.info(f"  V3 trader exited with code: {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning("  V3 trader timed out")
    except Exception as e:
        logger.error(f"  V3 trader error: {e}")

    logger.info(f"\n  === PHASE 3: POST-MARKET ===")
    now = datetime.now()
    post_market_time = now.replace(hour=15, minute=30, second=0)
    if now < post_market_time:
        wait = (post_market_time - now).total_seconds()
        logger.info(f"  Waiting {wait/60:.0f} min for post-market...")
        time.sleep(max(0, wait))
    post_market()
    logger.info(f"\n  DAY COMPLETE - {date.today()}")


def pre_market():
    logger.info("  Checking data freshness...")
    try:
        from data.data_loader import DataLoader
        from config.symbols import NIFTY_50
        loader = DataLoader()
        loader.load_backtest_data(NIFTY_50[:10], target_date=date.today().isoformat())
        logger.info("  Data cache warmed up")
    except Exception as e:
        logger.warning(f"  Data refresh error: {e}")
    tuned = Path("config/auto_tuned.yaml")
    if tuned.exists():
        logger.info("  Auto-tuned params found")
    else:
        logger.info("  Using default strategy params")


def post_market():
    logger.info("  Running post-market analysis...")
    try:
        optimizer_path = PROJECT_DIR / "auto_optimizer.py"
        if optimizer_path.exists():
            subprocess.run([str(VENV_PYTHON), str(optimizer_path)],
                           cwd=str(PROJECT_DIR), timeout=600, capture_output=True, text=True)
    except Exception as e:
        logger.error(f"  Auto-optimizer error: {e}")

    model_path = PROJECT_DIR / "models" / "stock_predictor.pkl"
    should_retrain = False
    if not model_path.exists():
        should_retrain = True
    else:
        age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
        if age.days >= 7:
            should_retrain = True
    if should_retrain:
        retrain()
    send_daily_report()


def retrain():
    logger.info("  Retraining ML model...")
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", "data.train_pipeline", "train"],
            cwd=str(PROJECT_DIR), timeout=600, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("  ML model retrained successfully")
        else:
            logger.error(f"  Retrain failed: {result.stderr[:200]}")
    except Exception as e:
        logger.error(f"  Retrain error: {e}")


def send_daily_report():
    try:
        import yaml
        config = {}
        for p in ["config/config_test.yaml", "config/config_prod.yaml"]:
            if Path(p).exists():
                with open(p) as f:
                    config = yaml.safe_load(f)
                break
        results_dir = PROJECT_DIR / "results"
        today_csv = results_dir / f"live_v3_{date.today()}.csv"
        if today_csv.exists():
            import pandas as pd
            trades = pd.read_csv(today_csv)
            total_trades = len(trades)
            wins = len(trades[trades["net_pnl"] > 0])
            total_pnl = trades["net_pnl"].sum()
            wr = wins / total_trades * 100 if total_trades > 0 else 0
            msg = f"📊 Daily Report — {date.today()}\n📋 Trades: {total_trades} | Win rate: {wr:.0f}%\n💰 P&L: ₹{total_pnl:+,.0f}"
        else:
            msg = f"📊 Daily Report — {date.today()}\n📭 No trades today"
        from live_paper_v3 import send_telegram
        send_telegram(msg, config)
        logger.info("  Daily report sent")
    except Exception as e:
        logger.warning(f"  Daily report error: {e}")


def run_weekly():
    logger.info(f"\n  WEEKLY MAINTENANCE - {date.today()}")
    download_data()
    retrain()
    logger.info("  Weekly maintenance complete")


def download_data():
    try:
        enricher = PROJECT_DIR / "data" / "data_enricher.py"
        if enricher.exists():
            subprocess.run([str(VENV_PYTHON), "-m", "data.data_enricher", "download-all"],
                           cwd=str(PROJECT_DIR), timeout=3600)
        else:
            logger.info("  No data_enricher.py — using yfinance loader")
            from data.data_loader import DataLoader
            from config.symbols import get_universe
            loader = DataLoader()
            symbols = get_universe("nifty250")
            loader.load_backtest_data(symbols, target_date=date.today().isoformat())
            logger.info(f"  Downloaded data for {len(symbols)} symbols")
    except Exception as e:
        logger.error(f"  Download error: {e}")


def install():
    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        existing = ""
    lines = [l for l in existing.split("\n") if CRON_MARKER not in l and "NSE Algo Trader" not in l]
    new_crontab = "\n".join(lines).strip() + "\n" + CRON_JOBS
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE)
    proc.communicate(new_crontab.encode())
    print(f"""
  ✅ Scheduler installed!
  Daily (Mon-Fri 8:55 AM): Full lifecycle → V3 trade → optimize
  Weekly (Sat 6 AM): Data download + ML retrain
  Holiday detection active. Logs: {LOG_DIR}/
  Commands: status | run-now | run-day | post-market | download-data | retrain | uninstall
    """)


def uninstall():
    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        print("  No crontab found.")
        return
    lines = [l for l in existing.split("\n") if CRON_MARKER not in l and "NSE Algo Trader" not in l]
    cleaned = "\n".join(l for l in lines if l.strip()).strip() + "\n"
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE)
    proc.communicate(cleaned.encode())
    print("  ✅ Scheduler removed.")


def status():
    today = date.today()
    trading = "TRADING DAY" if is_trading_day() else f"🚫 {get_holiday_name()}"
    print(f"\n  📅 Today: {today} ({trading})")
    d = today + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    print(f"  📅 Next trading day: {d}")
    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
        jobs = [l for l in existing.split("\n") if CRON_MARKER in l]
        if jobs:
            print(f"\n  ✅ Scheduler ACTIVE ({len(jobs)} cron jobs)")
        else:
            print("\n  ❌ Scheduler NOT active. Run: python scheduler.py install")
    except subprocess.CalledProcessError:
        print("\n  ❌ No crontab. Run: python scheduler.py install")
    angel_cfg = Path("config/angel_config.yaml")
    if angel_cfg.exists():
        import yaml
        with open(angel_cfg) as f:
            ac = yaml.safe_load(f) or {}
        api_key = ac.get("angel_one", {}).get("api_key", "")
        if api_key and api_key != "YOUR_API_KEY":
            print(f"  ⚡ Angel One: CONFIGURED")
        else:
            print(f"  📡 Angel One: NOT configured (yfinance fallback)")
    else:
        print(f"  📡 Angel One: config not found (yfinance fallback)")
    model_path = PROJECT_DIR / "models" / "stock_predictor.pkl"
    if model_path.exists():
        age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
        stale = " ⚠️ STALE" if age.days > 7 else ""
        print(f"  🧠 ML Model: {age.days}d old{stale}")
    else:
        print("  🧠 ML Model: NOT FOUND")
    results_dir = PROJECT_DIR / "results"
    if results_dir.exists():
        csvs = sorted(results_dir.glob("live_v*.csv"))
        if csvs:
            print(f"  📁 Latest results: {csvs[-1].name}")


def run_now():
    if not is_trading_day():
        reason = "Weekend" if date.today().weekday() >= 5 else get_holiday_name()
        print(f"\n  🚫 Not a trading day: {reason}")
        print(f"  Force: python live_paper_v3.py --data-source yfinance")
        return
    print("  Starting V3 live paper trading...")
    os.execvp(str(VENV_PYTHON), [str(VENV_PYTHON), str(PROJECT_DIR / "live_paper_v3.py")])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    commands = {
        "install": install, "uninstall": uninstall, "status": status,
        "run-now": run_now, "run-day": run_day, "post-market": post_market,
        "download-data": download_data, "retrain": retrain, "weekly": run_weekly,
    }
    if cmd in commands:
        commands[cmd]()
    else:
        print(f"  Unknown: {cmd}. Available: {', '.join(commands.keys())}")
