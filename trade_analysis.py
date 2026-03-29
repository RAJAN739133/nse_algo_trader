"""
Trade Analysis Dashboard — Detailed P&L for every trade.

Shows:
  - Every trade: entry, exit, P&L, costs, reason, strategy
  - Daily/weekly/monthly P&L breakdown
  - Win rate, profit factor, max drawdown
  - Best/worst trades, streaks
  - Strategy comparison (ORB vs VWAP)
  - Stock-wise performance

Usage:
  python trade_analysis.py                    # Analyse all results
  python trade_analysis.py --date 2024-03-15  # Analyse specific day
  python trade_analysis.py --detailed         # Show every single trade
  python trade_analysis.py --export           # Export to CSV report
"""
import os, sys, argparse
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = Path("results")


def load_all_trades():
    """Load all trade results from results/ folder."""
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        print("\n  No trade results found!")
        print("  Run a test first: python paper_trader.py simulate")
        print("  Or backtest:      python run_backtest.py --last 10")
        return pd.DataFrame()
    all_df = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "date" not in df.columns:
                df["date"] = f.stem.split("_")[-1]
            all_df.append(df)
        except Exception:
            pass
    if not all_df:
        return pd.DataFrame()
    combined = pd.concat(all_df, ignore_index=True)
    return combined


def show_trade_details(df):
    """Print every single trade with full details."""
    print(f"\n{'='*80}")
    print(f"  DETAILED TRADE LOG — {len(df)} trades")
    print(f"{'='*80}")
    print(f"  {'#':<4}{'Date':<12}{'Symbol':<12}{'Strategy':<8}{'Entry':>8}{'Exit':>8}{'Qty':>5}{'Gross':>10}{'Costs':>8}{'Net P&L':>10}{'Reason':<12}")
    print(f"  {'-'*95}")

    running_pnl = 0
    for i, (_, t) in enumerate(df.iterrows(), 1):
        net = t.get("net_pnl", 0)
        running_pnl += net
        emoji = "+" if net > 0 else "-"
        print(
            f"  {i:<4}"
            f"{str(t.get('date',''))[:10]:<12}"
            f"{t.get('symbol','?'):<12}"
            f"{t.get('strategy','?'):<8}"
            f"{t.get('entry', t.get('entry_price',0)):>8.2f}"
            f"{t.get('exit', t.get('exit_price',0)):>8.2f}"
            f"{t.get('qty',0):>5}"
            f"{t.get('gross', t.get('gross_pnl',0)):>+10.2f}"
            f"{t.get('costs',0):>8.2f}"
            f"{net:>+10.2f}"
            f"  {t.get('reason', t.get('exit_reason','')):<12}"
        )

    print(f"  {'-'*95}")
    print(f"  {'RUNNING TOTAL':>60}{'':>18}{running_pnl:>+10.2f}")


def show_summary(df):
    """Overall P&L summary."""
    total_pnl = df["net_pnl"].sum()
    total_costs = df["costs"].sum() if "costs" in df.columns else 0
    gross_pnl = df.get("gross", df.get("gross_pnl", pd.Series(dtype=float))).sum()
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]

    print(f"\n{'='*60}")
    print(f"  OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total trades:      {len(df)}")
    print(f"  Winning trades:    {len(wins)} ({len(wins)/len(df)*100:.0f}%)")
    print(f"  Losing trades:     {len(losses)} ({len(losses)/len(df)*100:.0f}%)")
    print(f"  ─────────────────────────────────")
    print(f"  Gross P&L:         Rs {gross_pnl:>+10,.2f}")
    print(f"  Total costs:       Rs {total_costs:>10,.2f}")
    print(f"  Net P&L:           Rs {total_pnl:>+10,.2f}")
    print(f"  ─────────────────────────────────")

    if len(wins) > 0:
        avg_win = wins["net_pnl"].mean()
        max_win = wins["net_pnl"].max()
        print(f"  Avg win:           Rs {avg_win:>+10,.2f}")
        print(f"  Best trade:        Rs {max_win:>+10,.2f}")
    if len(losses) > 0:
        avg_loss = losses["net_pnl"].mean()
        max_loss = losses["net_pnl"].min()
        print(f"  Avg loss:          Rs {avg_loss:>+10,.2f}")
        print(f"  Worst trade:       Rs {max_loss:>+10,.2f}")

    if len(wins) > 0 and len(losses) > 0 and losses["net_pnl"].sum() != 0:
        pf = abs(wins["net_pnl"].sum() / losses["net_pnl"].sum())
        print(f"  Profit factor:     {pf:.2f}x")

    # Expectancy
    if len(df) > 0:
        expectancy = total_pnl / len(df)
        print(f"  Expectancy:        Rs {expectancy:>+10,.2f} per trade")

    # Max drawdown
    cumulative = df["net_pnl"].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_dd = drawdown.min()
    print(f"  Max drawdown:      Rs {max_dd:>+10,.2f}")

    # Streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for pnl in df["net_pnl"]:
        if pnl > 0:
            streak = max(1, streak + 1) if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = min(-1, streak - 1) if streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(streak))
    print(f"  Win streak:        {max_win_streak} trades")
    print(f"  Loss streak:       {max_loss_streak} trades")


def show_by_strategy(df):
    """Performance breakdown by strategy."""
    if "strategy" not in df.columns:
        return
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE BY STRATEGY")
    print(f"{'='*60}")
    print(f"  {'Strategy':<15}{'Trades':>7}{'Wins':>6}{'Win%':>7}{'Net P&L':>12}{'Avg':>10}")
    print(f"  {'-'*57}")
    for strat, g in df.groupby("strategy"):
        w = len(g[g["net_pnl"] > 0])
        print(f"  {strat:<15}{len(g):>7}{w:>6}{w/len(g)*100:>6.0f}%{g['net_pnl'].sum():>+12,.2f}{g['net_pnl'].mean():>+10,.2f}")


def show_by_stock(df):
    """Performance breakdown by stock."""
    if "symbol" not in df.columns:
        return
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE BY STOCK")
    print(f"{'='*60}")
    print(f"  {'Stock':<15}{'Trades':>7}{'Wins':>6}{'Win%':>7}{'Net P&L':>12}{'Avg':>10}")
    print(f"  {'-'*57}")
    by_stock = df.groupby("symbol").agg(
        trades=("net_pnl", "count"),
        wins=("net_pnl", lambda x: (x > 0).sum()),
        total=("net_pnl", "sum"),
        avg=("net_pnl", "mean"),
    ).sort_values("total", ascending=False)
    for sym, r in by_stock.iterrows():
        wr = r["wins"] / r["trades"] * 100 if r["trades"] > 0 else 0
        print(f"  {sym:<15}{r['trades']:>7}{r['wins']:>6}{wr:>6.0f}%{r['total']:>+12,.2f}{r['avg']:>+10,.2f}")


def show_by_exit_reason(df):
    """How trades ended."""
    col = "reason" if "reason" in df.columns else ("exit_reason" if "exit_reason" in df.columns else None)
    if not col:
        return
    print(f"\n{'='*60}")
    print(f"  EXIT REASON BREAKDOWN")
    print(f"{'='*60}")
    print(f"  {'Reason':<18}{'Count':>7}{'Avg P&L':>12}{'Total':>12}")
    print(f"  {'-'*49}")
    for reason, g in df.groupby(col):
        print(f"  {reason:<18}{len(g):>7}{g['net_pnl'].mean():>+12,.2f}{g['net_pnl'].sum():>+12,.2f}")


def show_daily(df):
    """Daily P&L if dates available."""
    if "date" not in df.columns:
        return
    print(f"\n{'='*60}")
    print(f"  DAILY P&L")
    print(f"{'='*60}")
    daily = df.groupby("date").agg(
        trades=("net_pnl", "count"),
        pnl=("net_pnl", "sum"),
        costs=("costs", "sum") if "costs" in df.columns else ("net_pnl", lambda x: 0),
    )
    print(f"  {'Date':<14}{'Trades':>7}{'P&L':>12}{'Costs':>10}{'Cumulative':>12}")
    print(f"  {'-'*55}")
    cum = 0
    for d, r in daily.iterrows():
        cum += r["pnl"]
        emoji = "+" if r["pnl"] > 0 else ("-" if r["pnl"] < 0 else " ")
        print(f"  {str(d)[:10]:<14}{r['trades']:>7}{r['pnl']:>+12,.2f}{r['costs']:>10,.2f}{cum:>+12,.2f}")


def export_report(df):
    """Export detailed report to CSV."""
    outpath = f"results/analysis_report_{date.today()}.csv"
    df.to_csv(outpath, index=False)
    print(f"\n  Report exported to: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Trade Analysis Dashboard")
    parser.add_argument("--date", help="Analyse specific date")
    parser.add_argument("--detailed", action="store_true", help="Show every trade")
    parser.add_argument("--export", action="store_true", help="Export CSV report")
    args = parser.parse_args()

    df = load_all_trades()
    if df.empty:
        return

    # Normalize column names
    if "net_pnl" not in df.columns and "pnl" in df.columns:
        df["net_pnl"] = df["pnl"]
    if "net_pnl" not in df.columns:
        print("  No P&L data found in results.")
        return

    if args.date:
        df = df[df["date"].astype(str).str.contains(args.date)]
        if df.empty:
            print(f"  No trades found for {args.date}")
            return

    # Always show detailed log
    show_trade_details(df)
    show_summary(df)
    show_by_strategy(df)
    show_by_stock(df)
    show_by_exit_reason(df)
    show_daily(df)

    if args.export:
        export_report(df)

    # Monthly projection
    if len(df) > 0:
        avg_daily = df.groupby("date")["net_pnl"].sum().mean() if "date" in df.columns else df["net_pnl"].sum()
        print(f"\n{'='*60}")
        print(f"  PROJECTION (based on current performance)")
        print(f"{'='*60}")
        print(f"  Avg daily P&L:     Rs {avg_daily:+,.2f}")
        print(f"  Monthly (22 days): Rs {avg_daily*22:+,.2f}")
        print(f"  Yearly:            Rs {avg_daily*252:+,.2f}")
        print(f"  Monthly ROI:       {avg_daily*22/100000*100:+.2f}% (on Rs 1L)")


if __name__ == "__main__":
    main()
