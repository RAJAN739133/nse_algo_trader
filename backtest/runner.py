"""
Backtest runner — the main entry point for running backtests.
Generates synthetic NSE intraday data and produces performance reports.

Usage:
    python -m backtest.runner
"""

import sys
import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.orb import ORBStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from backtest.engine import BacktestEngine, BacktestConfig
from utils.indicators import add_all_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    symbol: str = "RELIANCE",
    days: int = 120,
    candle_minutes: int = 5,
    base_price: float = 1250.0,
    daily_volatility: float = 0.015,
) -> pd.DataFrame:
    """
    Generate realistic synthetic NSE intraday data.
    Mimics real market behaviour: opening gaps, volume spikes at open/close,
    lunch-hour lull, and trending/mean-reverting regimes.
    """
    rows = []
    price = base_price
    candles_per_day = int((6 * 60 + 15) / candle_minutes)  # 9:15 to 15:30

    for day in range(days):
        current_date = datetime(2025, 1, 2) + timedelta(days=day)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Overnight gap (-1% to +1%)
        gap = np.random.normal(0, 0.005)
        price *= 1 + gap

        # Day regime: trending (40%) or mean-reverting (60%)
        is_trending = np.random.random() < 0.4
        trend_dir = np.random.choice([-1, 1])

        for candle in range(candles_per_day):
            minute = candle * candle_minutes
            hour = 9 + (minute + 15) // 60
            mins = (minute + 15) % 60

            if hour > 15 or (hour == 15 and mins > 30):
                break

            candle_time = current_date.replace(hour=hour, minute=mins)

            # Volatility varies by time of day
            if hour < 10:  # opening hour — high vol
                vol_mult = 2.0
            elif 11 <= hour < 14:  # lunch — low vol
                vol_mult = 0.5
            elif hour >= 14:  # closing — medium-high vol
                vol_mult = 1.5
            else:
                vol_mult = 1.0

            candle_vol = daily_volatility * vol_mult / np.sqrt(candles_per_day)

            # Price movement
            if is_trending:
                drift = trend_dir * candle_vol * 0.3
            else:
                # Mean revert toward day's VWAP (approximated as opening price)
                drift = -0.1 * (price - base_price * (1 + gap)) / price

            returns = drift + np.random.normal(0, candle_vol)
            new_price = price * (1 + returns)

            # Generate OHLC
            intra_high = max(price, new_price) * (1 + abs(np.random.normal(0, candle_vol * 0.3)))
            intra_low = min(price, new_price) * (1 - abs(np.random.normal(0, candle_vol * 0.3)))

            # Volume: higher at open and close
            base_volume = 500_000
            if hour < 10:
                volume = int(base_volume * np.random.uniform(2, 4))
            elif 11 <= hour < 14:
                volume = int(base_volume * np.random.uniform(0.3, 0.8))
            elif hour >= 14:
                volume = int(base_volume * np.random.uniform(1.5, 3))
            else:
                volume = int(base_volume * np.random.uniform(0.8, 1.5))

            rows.append({
                "datetime": candle_time,
                "symbol": symbol,
                "open": round(price, 2),
                "high": round(intra_high, 2),
                "low": round(intra_low, 2),
                "close": round(new_price, 2),
                "volume": volume,
            })

            price = new_price

    df = pd.DataFrame(rows)
    return df


def print_report(result, config):
    """Print a formatted backtest report."""
    summary = result.summary(config.capital)

    print("\n" + "=" * 60)
    print("  BACKTEST REPORT")
    print("=" * 60)
    print(f"  Initial Capital:   ₹{config.capital:,.0f}")
    print(f"  Final Capital:     ₹{config.capital + result.total_pnl:,.0f}")
    print("-" * 60)

    for key, val in summary.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<20} {val}")

    print("-" * 60)

    if result.trades:
        print("\n  LAST 10 TRADES:")
        print(f"  {'Side':<6} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'Reason':<20}")
        print("  " + "-" * 56)
        for t in result.trades[-10:]:
            pnl_str = f"₹{t.pnl:,.0f}"
            marker = "+" if t.pnl > 0 else ""
            print(
                f"  {t.side:<6} ₹{t.entry_price:>8,.2f} ₹{t.exit_price:>8,.2f} "
                f"{marker}{pnl_str:>9} {t.exit_reason:<20}"
            )

    print("=" * 60 + "\n")


def main():
    print("Generating synthetic NSE intraday data...")
    data = generate_synthetic_data(
        symbol="RELIANCE",
        days=120,
        candle_minutes=5,
        base_price=1250.0,
    )
    print(f"Generated {len(data)} candles over {data['datetime'].dt.date.nunique()} trading days")

    # Add indicators
    print("Computing indicators...")
    data = add_all_indicators(data)

    # Configure backtest
    config = BacktestConfig(
        capital=100_000,
        risk_per_trade=0.01,
        max_trades_per_day=2,
        daily_loss_limit=0.03,
        slippage_pct=0.0005,
    )

    engine = BacktestEngine(config)

    # --- Run ORB Strategy ---
    print("\nRunning ORB strategy backtest...")
    orb = ORBStrategy({
        "enabled": True,
        "orb_period_minutes": 15,
        "volume_multiplier": 1.5,
        "atr_stop_multiplier": 1.5,
        "partial_exit_rr": 1.0,
    })

    orb_result = engine.run(orb, data)
    print_report(orb_result, config)

    # --- Run VWAP Strategy ---
    print("Running VWAP Reversion strategy backtest...")
    engine_vwap = BacktestEngine(config)  # fresh engine
    vwap = VWAPReversionStrategy({
        "enabled": True,
        "entry_band": 1.0,
        "stop_band": 2.0,
        "target_band": 0.0,
        "min_volume_percentile": 60,
    })

    vwap_result = engine_vwap.run(vwap, data)
    print_report(vwap_result, config)

    # --- Combined summary ---
    print("=" * 60)
    print("  COMBINED STRATEGY SUMMARY")
    print("=" * 60)
    all_trades = orb_result.trades + vwap_result.trades
    total_pnl = sum(t.pnl for t in all_trades)
    wins = sum(1 for t in all_trades if t.pnl > 0)
    total = len(all_trades)
    print(f"  Total trades:      {total}")
    print(f"  Combined P&L:      ₹{total_pnl:,.2f}")
    print(f"  Combined win rate: {wins/total*100:.1f}%" if total > 0 else "  No trades")
    print(f"  Combined return:   {total_pnl/config.capital*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
