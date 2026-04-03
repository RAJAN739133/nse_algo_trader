#!/usr/bin/env python3
"""
FAIR BACKTEST — No Lookahead Bias
═══════════════════════════════════

This script runs a proper backtest for the past week without cheating:

1. For each day, we ONLY use data available UP TO that day's open
2. Stock selection uses PREVIOUS day's data (not same-day)
3. Intraday signals use ONLY candles seen so far (not future candles)
4. No peeking at end-of-day results for entry decisions

Usage:
    python backtest_week.py
    python backtest_week.py --days 5
    python backtest_week.py --start 2026-03-27 --end 2026-04-02
"""

import os
import sys
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_trading_days(start_date, end_date):
    """Get trading days (Mon-Fri) between dates."""
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon=0, Fri=4
            days.append(current)
        current += timedelta(days=1)
    return days


def run_single_day_backtest(target_date):
    """
    Run backtest for a single day with NO lookahead bias.
    
    Key anti-cheating measures:
    1. Stock selection uses data from BEFORE target_date
    2. Intraday candles are processed sequentially
    3. No future data is used for any decision
    """
    from live_paper_v3 import run
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  BACKTEST: {target_date} ({target_date.strftime('%A')})")
    logger.info(f"{'='*60}")
    
    try:
        # Run the backtest for this specific date
        # The live_paper_v3.run() function handles backtest mode properly
        run(backtest_date=target_date.isoformat())
        return True
    except Exception as e:
        logger.error(f"  Backtest failed for {target_date}: {e}")
        return False


def collect_results(dates):
    """Collect and summarize results from all backtest days."""
    results_dir = Path("results")
    all_trades = []
    
    for d in dates:
        csv_path = results_dir / f"live_v3_{d}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df["date"] = str(d)
                all_trades.append(df)
                logger.info(f"  {d}: {len(df)} trades, P&L: Rs {df['net_pnl'].sum():+,.2f}")
            except Exception as e:
                logger.warning(f"  {d}: Could not read results - {e}")
        else:
            logger.warning(f"  {d}: No results file found")
    
    if not all_trades:
        return None
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined


def print_summary(df):
    """Print comprehensive backtest summary."""
    if df is None or df.empty:
        logger.info("\n  No trades to summarize.")
        return
    
    total_trades = len(df)
    wins = len(df[df["net_pnl"] > 0])
    losses = len(df[df["net_pnl"] <= 0])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = df["net_pnl"].sum()
    total_gross = df["gross"].sum() if "gross" in df.columns else 0
    total_costs = df["costs"].sum() if "costs" in df.columns else 0
    
    avg_win = df[df["net_pnl"] > 0]["net_pnl"].mean() if wins > 0 else 0
    avg_loss = df[df["net_pnl"] <= 0]["net_pnl"].mean() if losses > 0 else 0
    
    # By direction
    longs = df[df["direction"] == "LONG"]
    shorts = df[df["direction"] == "SHORT"]
    
    # By strategy
    strategy_stats = df.groupby("type").agg({
        "net_pnl": ["count", "sum", lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    strategy_stats.columns = ["trades", "pnl", "win_rate"]
    
    # By day
    daily_stats = df.groupby("date").agg({
        "net_pnl": ["count", "sum"]
    }).round(2)
    daily_stats.columns = ["trades", "pnl"]
    
    print("\n" + "="*70)
    print("  BACKTEST SUMMARY — PAST WEEK (NO LOOKAHEAD BIAS)")
    print("="*70)
    
    print(f"\n  OVERALL PERFORMANCE")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total Trades:     {total_trades}")
    print(f"  Wins:             {wins} ({win_rate:.1f}%)")
    print(f"  Losses:           {losses}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Gross P&L:        Rs {total_gross:+,.2f}")
    print(f"  Costs:            Rs {total_costs:,.2f}")
    print(f"  Net P&L:          Rs {total_pnl:+,.2f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Avg Win:          Rs {avg_win:+,.2f}")
    print(f"  Avg Loss:         Rs {avg_loss:+,.2f}")
    if avg_loss != 0:
        print(f"  Win/Loss Ratio:   {abs(avg_win/avg_loss):.2f}")
    
    print(f"\n  BY DIRECTION")
    print(f"  ─────────────────────────────────────────")
    if len(longs) > 0:
        long_wr = len(longs[longs["net_pnl"] > 0]) / len(longs) * 100
        print(f"  LONG:  {len(longs)} trades, Rs {longs['net_pnl'].sum():+,.2f}, WR: {long_wr:.1f}%")
    if len(shorts) > 0:
        short_wr = len(shorts[shorts["net_pnl"] > 0]) / len(shorts) * 100
        print(f"  SHORT: {len(shorts)} trades, Rs {shorts['net_pnl'].sum():+,.2f}, WR: {short_wr:.1f}%")
    
    print(f"\n  BY STRATEGY")
    print(f"  ─────────────────────────────────────────")
    for strategy, row in strategy_stats.iterrows():
        print(f"  {strategy:<25} {int(row['trades']):>3} trades, Rs {row['pnl']:>+10,.2f}, WR: {row['win_rate']:.0f}%")
    
    print(f"\n  BY DAY")
    print(f"  ─────────────────────────────────────────")
    for day, row in daily_stats.iterrows():
        print(f"  {day}  {int(row['trades']):>3} trades, Rs {row['pnl']:>+10,.2f}")
    
    print("\n" + "="*70)
    
    # Save combined results
    combined_path = Path("results/backtest_week_combined.csv")
    df.to_csv(combined_path, index=False)
    print(f"\n  Combined results saved to: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Fair backtest for past week")
    parser.add_argument("--days", type=int, default=5, help="Number of trading days to backtest")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # Determine date range
    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        # Default: past N trading days (excluding today)
        today = date.today()
        trading_days = []
        current = today - timedelta(days=1)
        while len(trading_days) < args.days:
            if current.weekday() < 5:  # Mon-Fri
                trading_days.append(current)
            current -= timedelta(days=1)
        trading_days.reverse()
        start_date = trading_days[0]
        end_date = trading_days[-1]
    
    # Get all trading days in range
    dates = get_trading_days(start_date, end_date)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  FAIR BACKTEST — NO LOOKAHEAD BIAS")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Trading days: {len(dates)}")
    logger.info(f"{'='*60}")
    
    # Run backtest for each day
    for d in dates:
        run_single_day_backtest(d)
    
    # Collect and summarize results
    logger.info(f"\n{'='*60}")
    logger.info(f"  COLLECTING RESULTS")
    logger.info(f"{'='*60}")
    
    results = collect_results(dates)
    print_summary(results)


if __name__ == "__main__":
    main()
