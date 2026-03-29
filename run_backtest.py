"""
Backtest on any past date — test your algo on real historical data.

Usage:
  python run_backtest.py --date 2024-03-15                # Test one specific day
  python run_backtest.py --from 2024-03-01 --to 2024-03-31  # Test a date range
  python run_backtest.py --date 2020-03-23                # Test on COVID crash day!
  python run_backtest.py --date 2024-06-04                # Test on election result day!
  python run_backtest.py --last 30                        # Test last 30 trading days

The backtest:
  1. Loads REAL historical data (from yfinance or your configured source)
  2. Feeds it to the ML model + scoring engine
  3. Simulates trades exactly like paper_trader.py
  4. Saves results to results/ folder
"""
import os, sys, argparse, logging
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.symbols import DEFAULT_UNIVERSE
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/backtest_{date.today()}.log")])
logger = logging.getLogger(__name__)


def run_single_day(target_date, symbols, loader, model=None, features=None):
    """Backtest one specific trading day using real data."""
    logger.info(f"\n{'='*60}\n  BACKTEST: {target_date}\n{'='*60}")

    # Load data UP TO target_date (no future leakage)
    df = loader.load_backtest_data(symbols, target_date=target_date)
    if df.empty:
        logger.warning(f"  No data available for {target_date}")
        return []

    # Add features for each stock
    featured = []
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) < 50:
            continue
        sdf = add_features(sdf)
        featured.append(sdf)

    if not featured:
        logger.warning("  Not enough data for any stock")
        return []

    all_featured = pd.concat(featured, ignore_index=True)

    # Score stocks on the day BEFORE target (model decides morning before market)
    avail_feats = [c for c in FEATURE_COLS if c in all_featured.columns]
    scores = score_stocks(all_featured, model, avail_feats if model else None)

    # Show scores
    logger.info(f"\n  Stock scores for {target_date}:")
    logger.info(f"  {'Symbol':<12}{'Score':>6}{'RSI':>6}{'Strategy':<8}")
    for _, r in scores.head(10).iterrows():
        star = " <-- TRADE" if r["score"] >= 60 else ""
        logger.info(f"  {r['symbol']:<12}{r['score']:>5.0f}{r['rsi']:>6.1f} {r['strategy']:<8}{star}")

    picks = scores[scores["score"] >= 60].head(3)
    if picks.empty:
        logger.info(f"\n  No stocks scored >= 60. Algo sits out today.")
        return []

    # Simulate trades for picked stocks
    cost_model = ZerodhaCostModel()
    cfg_path = Path("config/config_test.yaml")
    if not cfg_path.exists():
        cfg_path = Path("config/config_example.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    trades = []
    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        sdf = all_featured[all_featured["symbol"] == sym]
        if len(sdf) < 2:
            continue

        # Get the target day's data
        last_day = sdf.iloc[-1]
        prev_day = sdf.iloc[-2]
        entry = last_day["open"]
        close = last_day["close"]
        high = last_day["high"]
        low = last_day["low"]
        atr = last_day.get("atr_14", abs(high - low))

        # Simulate ORB-style trade
        sl = entry - atr * 1.5
        target = entry + atr * 2.0

        # What would have happened?
        if high >= target:
            exit_price = target
            reason = "TARGET"
        elif low <= sl:
            exit_price = sl
            reason = "STOP_LOSS"
        else:
            exit_price = close  # square off at close
            reason = "SQUARE_OFF"

        risk = entry - sl
        qty = max(1, int(config["capital"]["total"] * config["capital"]["risk_per_trade"] / max(risk, 1)))
        gross = (exit_price - entry) * qty
        costs = cost_model.calculate(entry * qty, exit_price * qty).total
        net = gross - costs

        trade = {
            "date": str(target_date), "symbol": sym, "strategy": pick["strategy"],
            "score": pick["score"], "entry": round(entry, 2), "exit": round(exit_price, 2),
            "sl": round(sl, 2), "target": round(target, 2), "qty": qty,
            "gross_pnl": round(gross, 2), "costs": round(costs, 2),
            "net_pnl": round(net, 2), "reason": reason,
        }
        trades.append(trade)
        emoji = "WIN " if net > 0 else "LOSS"
        logger.info(f"\n  {emoji} {sym} @ {entry:.2f} -> {exit_price:.2f} | Rs {net:+,.2f} | {reason}")

    return trades


def main():
    parser = argparse.ArgumentParser(description="Backtest on real historical data")
    parser.add_argument("--date", help="Test specific date (YYYY-MM-DD)")
    parser.add_argument("--from", dest="from_date", help="Start date for range")
    parser.add_argument("--to", dest="to_date", help="End date for range")
    parser.add_argument("--last", type=int, help="Test last N trading days")
    parser.add_argument("--stocks", nargs="+", default=None)
    parser.add_argument("--source", default=None, help="Override data source")
    args = parser.parse_args()

    symbols = args.stocks or DEFAULT_UNIVERSE[:10]
    loader = DataLoader()

    # Override source if specified
    if args.source:
        loader.config["backtest_source"] = args.source

    logger.info(f"  Data source: {loader.config.get('backtest_source')}")
    logger.info(f"  Stocks: {', '.join(symbols[:5])}...")

    # Load ML model if available
    model, features = None, None
    model_path = Path("models/stock_predictor.pkl")
    if model_path.exists():
        import pickle
        with open(model_path, "rb") as f:
            d = pickle.load(f)
        model, features = d["model"], d["features"]
        logger.info("  ML model loaded")

    # Determine dates to test
    dates = []
    if args.date:
        dates = [args.date]
    elif args.from_date and args.to_date:
        d = pd.bdate_range(args.from_date, args.to_date)
        dates = [x.strftime("%Y-%m-%d") for x in d]
    elif args.last:
        d = pd.bdate_range(end=date.today(), periods=args.last)
        dates = [x.strftime("%Y-%m-%d") for x in d]
    else:
        dates = [date.today().isoformat()]

    logger.info(f"  Testing {len(dates)} trading days")

    # Run backtests
    all_trades = []
    for d in dates:
        trades = run_single_day(d, symbols, loader, model, features)
        all_trades.extend(trades)

    # Summary
    if all_trades:
        df = pd.DataFrame(all_trades)
        total = df["net_pnl"].sum()
        wins = len(df[df["net_pnl"] > 0])
        logger.info(f"\n{'='*60}\n  BACKTEST SUMMARY\n{'='*60}")
        logger.info(f"  Days: {len(dates)} | Trades: {len(df)} | Wins: {wins} | Losses: {len(df)-wins}")
        logger.info(f"  Net P&L: Rs {total:+,.2f}")
        logger.info(f"  Win rate: {wins/len(df)*100:.0f}%")
        logger.info(f"  Avg trade: Rs {df['net_pnl'].mean():+,.2f}")
        logger.info(f"  Best: Rs {df['net_pnl'].max():+,.2f} | Worst: Rs {df['net_pnl'].min():+,.2f}")

        outfile = f"results/backtest_{dates[0]}_to_{dates[-1]}.csv"
        df.to_csv(outfile, index=False)
        logger.info(f"\n  Saved: {outfile}")
    else:
        logger.info("\n  No trades taken across all test days.")


if __name__ == "__main__":
    main()
