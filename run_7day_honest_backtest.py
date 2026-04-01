#!/usr/bin/env python3
"""
Honest 7-day backtest — runs V3 on last 7 trading days as-if-live.
No lookahead bias: each day uses only data available up to that point.
"""
import os, sys, subprocess, csv
from datetime import date, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = str(PROJECT_DIR / "venv" / "bin" / "python3")

NSE_HOLIDAYS = {
    "2026-01-26", "2026-02-17", "2026-03-10",
    "2026-04-03", "2026-04-14",
}

def get_last_n_trading_days(n=7, from_date=None):
    d = from_date or (date.today() - timedelta(days=1))
    days = []
    while len(days) < n:
        if d.weekday() < 5 and str(d) not in NSE_HOLIDAYS:
            days.append(str(d))
        d -= timedelta(days=1)
    return list(reversed(days))

def run_backtest_day(dt_str):
    """Run V3 backtest for one day, return list of trade dicts."""
    print(f"\n{'='*60}")
    print(f"  BACKTESTING: {dt_str}")
    print(f"{'='*60}")
    result = subprocess.run(
        [VENV_PYTHON, str(PROJECT_DIR / "live_paper_v3.py"), "--backtest", dt_str],
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True, timeout=300,
    )
    # Print key lines from output
    for line in result.stdout.split("\n"):
        if any(kw in line for kw in ["TRADE CLOSED", "Selected", "ORB:", "BUYING", "SELLING", "REJECTED",
                                      "END OF DAY", "Net P&L", "Trades:", "No trades"]):
            print(f"  {line.strip()}")
    if result.returncode != 0:
        print(f"  ⚠️ Exit code: {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"  ERR: {line}")

    # Read results CSV
    csv_path = PROJECT_DIR / "results" / f"live_v3_{dt_str}.csv"
    trades = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["net_pnl"] = float(row["net_pnl"])
                row["qty"] = int(row["qty"])
                trades.append(row)
    return trades

def main():
    days = get_last_n_trading_days(7)
    print(f"\n  Last 7 trading days: {', '.join(days)}")
    print(f"  Running V3 with CURRENT code (all fixes applied)")

    all_trades = []
    day_results = []

    for dt_str in days:
        trades = run_backtest_day(dt_str)
        pnl = sum(t["net_pnl"] for t in trades)
        wins = sum(1 for t in trades if t["net_pnl"] > 0)
        losses = sum(1 for t in trades if t["net_pnl"] <= 0)
        all_trades.extend(trades)
        day_results.append({
            "date": dt_str, "trades": len(trades), "wins": wins,
            "losses": losses, "pnl": pnl,
        })

    # ── SUMMARY ──
    print(f"\n\n{'='*70}")
    print(f"  7-DAY BACKTEST SUMMARY (V3 with fixes)")
    print(f"{'='*70}")
    print(f"  {'Date':<12} {'Trades':>6} {'Won':>4} {'Lost':>5} {'P&L':>12} {'Result':>8}")
    print(f"  {'─'*50}")

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    winning_days = 0
    for dr in day_results:
        emoji = "✅" if dr["pnl"] > 0 else ("❌" if dr["pnl"] < 0 else "➖")
        print(f"  {dr['date']:<12} {dr['trades']:>6} {dr['wins']:>4} {dr['losses']:>5} Rs {dr['pnl']:>+10,.2f} {emoji}")
        total_pnl += dr["pnl"]
        total_trades += dr["trades"]
        total_wins += dr["wins"]
        if dr["pnl"] > 0:
            winning_days += 1

    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<12} {total_trades:>6} {total_wins:>4} {total_trades - total_wins:>5} Rs {total_pnl:>+10,.2f}")
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"\n  Win rate: {wr:.0f}% | Winning days: {winning_days}/7")
    print(f"  Capital: Rs 1,00,000 | Return: {total_pnl/100000*100:+.2f}%")

    # Check for dangerous qty trades
    danger_trades = [t for t in all_trades if t["qty"] > 500]
    if danger_trades:
        print(f"\n  ⚠️ HIGH QTY TRADES (>500 shares):")
        for t in danger_trades:
            print(f"    {t['symbol']} {t['type']} qty={t['qty']} regime={t.get('regime','?')} pnl=Rs {t['net_pnl']:+,.2f}")
    else:
        print(f"\n  ✅ No dangerous qty trades (all ≤500 shares)")

    # Check for TIME_DECAY trades
    td_trades = [t for t in all_trades if t.get("reason") == "TIME_DECAY"]
    if td_trades:
        td_pnl = sum(t["net_pnl"] for t in td_trades)
        print(f"\n  ⏰ TIME_DECAY exits: {len(td_trades)} trades, total Rs {td_pnl:+,.2f}")

if __name__ == "__main__":
    main()
