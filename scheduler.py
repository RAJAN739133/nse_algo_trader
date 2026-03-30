"""
Auto Scheduler — Runs the algo automatically every trading day.

Sets up a cron job (macOS/Linux) that:
  - Mon-Fri at 9:05 AM: Starts live_paper.py
  - Saturday at 6:00 AM: Retrains ML model with latest data
  - Skips weekends and market holidays

Usage:
  python scheduler.py install     # Install the cron jobs
  python scheduler.py uninstall   # Remove the cron jobs
  python scheduler.py status      # Check if scheduler is active
  python scheduler.py run-now     # Run live paper trading right now
"""
import os, sys, subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python3"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Cron job definitions ──
CRON_MARKER = "# nse_algo_trader"

CRON_JOBS = f"""
# ── NSE Algo Trader — Auto Schedule ──
# Mon-Fri 9:05 AM: Run live paper trading
5 9 * * 1-5 cd {PROJECT_DIR} && {VENV_PYTHON} {PROJECT_DIR}/live_paper.py >> {LOG_DIR}/cron_live.log 2>&1 {CRON_MARKER}

# Saturday 6:00 AM: Retrain ML model with latest data
0 6 * * 6 cd {PROJECT_DIR} && {VENV_PYTHON} -m data.train_pipeline train >> {LOG_DIR}/cron_train.log 2>&1 {CRON_MARKER}
"""


def install():
    """Install cron jobs."""
    # Get existing crontab
    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        existing = ""

    # Remove old entries
    lines = [l for l in existing.split("\n") if CRON_MARKER not in l and "NSE Algo Trader" not in l]
    new_crontab = "\n".join(lines).strip() + "\n" + CRON_JOBS

    # Install
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE)
    proc.communicate(new_crontab.encode())

    print(f"""
  ✅ Scheduler installed!

  Schedule:
    Mon-Fri 9:05 AM  →  Live paper trading starts automatically
    Saturday 6:00 AM  →  ML model retrained with latest data

  Logs:
    {LOG_DIR}/cron_live.log   — Trading logs
    {LOG_DIR}/cron_train.log  — Model training logs

  Commands:
    python scheduler.py status      — Check if active
    python scheduler.py uninstall   — Remove scheduler
    python scheduler.py run-now     — Run trading right now
    """)


def uninstall():
    """Remove cron jobs."""
    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        print("  No crontab found.")
        return

    lines = [l for l in existing.split("\n") if CRON_MARKER not in l and "NSE Algo Trader" not in l]
    cleaned = "\n".join(l for l in lines if l.strip()).strip() + "\n"

    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE)
    proc.communicate(cleaned.encode())
    print("  ✅ Scheduler removed. No more auto-trading.")


def status():
    """Check if scheduler is active."""
    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
        jobs = [l for l in existing.split("\n") if CRON_MARKER in l]
        if jobs:
            print(f"  ✅ Scheduler is ACTIVE ({len(jobs)} jobs)")
            for j in jobs:
                print(f"    {j.split(CRON_MARKER)[0].strip()}")
        else:
            print("  ❌ Scheduler is NOT active. Run: python scheduler.py install")
    except subprocess.CalledProcessError:
        print("  ❌ No crontab found. Run: python scheduler.py install")

    # Check model age
    model_path = PROJECT_DIR / "models" / "stock_predictor.pkl"
    if model_path.exists():
        import datetime
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"\n  ML Model: {model_path.name} (age: {age.days}d {age.seconds//3600}h)")
        if age.days > 7:
            print("  ⚠️  Model is older than 7 days — retrain: python -m data.train_pipeline train")
    else:
        print(f"\n  ML Model: NOT FOUND — train: python -m data.train_pipeline demo")


def run_now():
    """Run live paper trading immediately."""
    print("  Starting live paper trading...")
    os.execvp(str(VENV_PYTHON), [str(VENV_PYTHON), str(PROJECT_DIR / "live_paper.py")])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "install": install()
    elif cmd == "uninstall": uninstall()
    elif cmd == "status": status()
    elif cmd == "run-now": run_now()
    else: print("Usage: python scheduler.py [install|uninstall|status|run-now]")
