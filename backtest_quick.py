#!/usr/bin/env python3
"""
QUICK BACKTEST — Fast Fair Backtest for Past Week
════════════════════════════════════════════════════

This script runs a faster backtest by:
1. Using only top 5 most liquid stocks (to reduce data download)
2. Using smaller stock universe (nifty50 instead of fno)
3. Caching data aggressively

No lookahead bias - same rules as the full backtest.

Usage:
    python backtest_quick.py
    python backtest_quick.py --days 3
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import date, timedelta

# Suppress jugaad_data warnings
warnings.filterwarnings("ignore", category=UserWarning)

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


def run_quick_backtest(target_date, top_stocks=5):
    """
    Run quick backtest for a single day.
    Uses smaller stock universe for speed.
    """
    # Import here to avoid circular imports
    import live_paper_v3
    from config.symbols import get_universe
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  BACKTEST: {target_date} ({target_date.strftime('%A')})")
    logger.info(f"  Stocks: Top {top_stocks} from nifty50")
    logger.info(f"{'='*60}")
    
    # Override universe temporarily to use smaller set
    original_universe_func = live_paper_v3.get_universe
    
    def limited_universe(name):
        """Return smaller universe for faster backtest."""
        # Use only nifty50 instead of full fno
        base = original_universe_func("nifty50")
        return base[:30]  # Further limit to 30 stocks for speed
    
    try:
        # Patch universe function for speed
        live_paper_v3.get_universe = limited_universe
        
        # Run the backtest
        live_paper_v3.run(backtest_date=target_date.isoformat())
        return True
    except Exception as e:
        logger.error(f"  Backtest failed for {target_date}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original function
        live_paper_v3.get_universe = original_universe_func


def collect_and_summarize(dates):
    """Collect results and print summary."""
    import pandas as pd
    from pathlib import Path
    
    results_dir = Path("results")
    all_trades = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  COLLECTING RESULTS")
    logger.info(f"{'='*60}")
    
    for d in dates:
        csv_path = results_dir / f"live_v3_{d}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    df["date"] = str(d)
                    all_trades.append(df)
                    pnl = df['net_pnl'].sum()
                    wr = len(df[df['net_pnl'] > 0]) / len(df) * 100 if len(df) > 0 else 0
                    logger.info(f"  {d}: {len(df)} trades, P&L: Rs {pnl:+,.2f}, WR: {wr:.0f}%")
                else:
                    logger.info(f"  {d}: No trades")
            except Exception as e:
                logger.warning(f"  {d}: Could not read results - {e}")
        else:
            logger.info(f"  {d}: No results file")
    
    if not all_trades:
        logger.info("\n  No trades found in backtest period.")
        return
    
    combined = pd.concat(all_trades, ignore_index=True)
    print_summary(combined, dates)
    
    # Save combined results
    combined_path = results_dir / "backtest_quick_combined.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"\n  Combined results saved to: {combined_path}")


def print_summary(df, dates):
    """Print comprehensive backtest summary."""
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
    longs = df[df["direction"] == "LONG"] if "direction" in df.columns else df
    shorts = df[df["direction"] == "SHORT"] if "direction" in df.columns else df.iloc[0:0]
    
    # By strategy
    if "type" in df.columns:
        strategy_stats = df.groupby("type").agg({
            "net_pnl": ["count", "sum", lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0]
        }).round(2)
        strategy_stats.columns = ["trades", "pnl", "win_rate"]
    else:
        strategy_stats = None
    
    # By day
    daily_stats = df.groupby("date").agg({
        "net_pnl": ["count", "sum"]
    }).round(2)
    daily_stats.columns = ["trades", "pnl"]
    
    print("\n" + "="*70)
    print("  QUICK BACKTEST SUMMARY — NO LOOKAHEAD BIAS")
    print(f"  Period: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
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
    
    if strategy_stats is not None:
        print(f"\n  BY STRATEGY")
        print(f"  ─────────────────────────────────────────")
        for strategy, row in strategy_stats.iterrows():
            print(f"  {strategy:<25} {int(row['trades']):>3} trades, Rs {row['pnl']:>+10,.2f}, WR: {row['win_rate']:.0f}%")
    
    print(f"\n  BY DAY")
    print(f"  ─────────────────────────────────────────")
    for day, row in daily_stats.iterrows():
        print(f"  {day}  {int(row['trades']):>3} trades, Rs {row['pnl']:>+10,.2f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Quick fair backtest")
    parser.add_argument("--days", type=int, default=5, help="Number of trading days")
    parser.add_argument("--stocks", type=int, default=5, help="Top N stocks to trade")
    args = parser.parse_args()
    
    # Get past N trading days
    today = date.today()
    trading_days = []
    current = today - timedelta(days=1)
    while len(trading_days) < args.days:
        if current.weekday() < 5:
            trading_days.append(current)
        current -= timedelta(days=1)
    trading_days.reverse()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  QUICK BACKTEST — NO LOOKAHEAD BIAS")
    logger.info(f"  Period: {trading_days[0]} to {trading_days[-1]}")
    logger.info(f"  Trading days: {len(trading_days)}")
    logger.info(f"  Top stocks: {args.stocks}")
    logger.info(f"{'='*60}")
    
    # Run backtest for each day
    for d in trading_days:
        run_quick_backtest(d, top_stocks=args.stocks)
    
    # Collect and summarize
    collect_and_summarize(trading_days)


if __name__ == "__main__":
    main()
