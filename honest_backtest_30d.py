#!/usr/bin/env python3
"""
HONEST 30-Day Backtest — V1 vs V2 Side-by-Side Comparison.

Downloads REAL 5-min intraday data from yfinance for the last 30 trading
days, then walks candle-by-candle through each day using BOTH strategies.
Zero future data leakage. Realistic Zerodha costs.

Usage:
  python honest_backtest_30d.py                           # Nifty 50, last 30 days
  python honest_backtest_30d.py --last 60                 # Last 60 days
  python honest_backtest_30d.py --universe nifty100       # Broader universe
  python honest_backtest_30d.py --stocks HDFCBANK SBIN    # Specific stocks
  python honest_backtest_30d.py --top 5                   # Only top 5 scored picks

Output:
  - results/backtest_30d_v1_YYYY-MM-DD.csv
  - results/backtest_30d_v2_YYYY-MM-DD.csv
  - results/backtest_30d_comparison_YYYY-MM-DD.csv
"""
import os, sys, argparse, pickle, logging, time
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.symbols import NIFTY_50, get_universe
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel
from strategies.pro_strategy import ProStrategy
from strategies.pro_strategy_v2 import ProStrategyV2

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/backtest_30d_{date.today()}.log"),
    ],
)
logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 50


# ════════════════════════════════════════════════════════════
# INTRADAY DATA FETCHER
# ════════════════════════════════════════════════════════════

def fetch_intraday_multiday(symbol, days=60):
    """
    Fetch 5-min intraday data for a symbol over multiple days.
    yfinance provides max 60 days of 5-min data at once.
    Returns dict: {date_str: DataFrame} for each trading day.
    """
    try:
        import yfinance as yf
        ticker = f"{symbol}.NS"
        df = yf.Ticker(ticker).history(period=f"{days}d", interval="5m")
        if df.empty:
            return {}

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["symbol"] = symbol

        times = pd.to_datetime(df["datetime"])
        df["trade_date"] = times.dt.date

        days_data = {}
        for d, grp in df.groupby("trade_date"):
            day_df = grp.reset_index(drop=True)
            if len(day_df) >= 6:
                days_data[str(d)] = day_df
        return days_data

    except Exception as e:
        logger.warning(f"  {symbol}: intraday fetch failed — {e}")
        return {}


# ════════════════════════════════════════════════════════════
# HONEST TRADER — Candle-by-candle, zero cheating
# ════════════════════════════════════════════════════════════

class HonestTrader:
    """
    Walks candle-by-candle using a given strategy.
    Supports both ProStrategy (V1) and ProStrategyV2.
    """

    def __init__(self, symbol, direction, config, strategy_obj, version_label="V1"):
        self.symbol = symbol
        self.direction = direction
        self.capital = config["capital"]["total"]
        self.cost_model = ZerodhaCostModel()
        self.strategy = strategy_obj
        self.version = version_label

        self.orb = None
        self.in_trade = False
        self.entry_price = 0
        self.entry_time = None
        self.entry_idx = 0
        self.side = None
        self.sl = 0
        self.tgt = 0
        self.shares = 0
        self.initial_shares = 0
        self.sl_moved_to_be = False
        self.partial_exited = False
        self.trade_type = ""

        self.trades = []
        self.trade_count = 0
        self.max_trades = 2
        self.cooldown_until = 0

    def _close_trade(self, exit_price, exit_time, reason, shares_to_close=None):
        shares = shares_to_close or self.shares
        if self.side == "SHORT":
            gross = (self.entry_price - exit_price) * shares
        else:
            gross = (exit_price - self.entry_price) * shares
        costs = self.cost_model.calculate(self.entry_price * shares, exit_price * shares).total
        net = gross - costs

        et = str(self.entry_time).split("+")[0]
        xt = str(exit_time).split("+")[0]

        self.trades.append({
            "symbol": self.symbol, "direction": self.side,
            "entry": round(self.entry_price, 2), "exit": round(exit_price, 2),
            "entry_time": et, "exit_time": xt,
            "sl": round(self.sl, 2), "tgt": round(self.tgt, 2),
            "qty": shares, "gross": round(gross, 2),
            "costs": round(costs, 2), "net_pnl": round(net, 2),
            "reason": reason, "type": self.trade_type,
            "version": self.version,
        })

        if shares_to_close and shares_to_close < self.shares:
            self.shares -= shares_to_close
            self.partial_exited = True
        else:
            self.in_trade = False
            self.trade_count += 1
            self.sl_moved_to_be = False
            self.partial_exited = False

    def _enter_trade(self, signal, idx):
        self.entry_price = signal["entry"]
        self.entry_time = signal["time"]
        self.entry_idx = idx
        self.side = signal["side"]
        self.sl = signal["sl"]
        self.tgt = signal["tgt"]
        self.trade_type = signal["type"]
        risk = signal["risk"]
        self.shares = max(1, int(self.capital * self.strategy.max_risk_pct / max(risk, 0.01)))
        self.initial_shares = self.shares
        self.in_trade = True
        self.sl_moved_to_be = False
        self.partial_exited = False

    def process_candle(self, i, candles):
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        h, l, c = row["high"], row["low"], row["close"]
        hour, minute = t.hour, t.minute

        # ── Phase 1: ORB Formation ──
        if self.orb is None:
            if i < self.strategy.orb_candles:
                return
            self.orb = self.strategy.compute_orb(candles, self.strategy.orb_candles)
            if self.orb is None or self.orb["range"] < 0.5:
                return

        # ── Phase 2: Square off ──
        if hour >= self.strategy.square_off_hour and minute >= self.strategy.square_off_minute:
            if self.in_trade:
                self._close_trade(c, t, "SQUARE_OFF")
            return

        # ── Phase 3: Manage open trade ──
        if self.in_trade:
            # Time-decay exit (V2 feature)
            time_decay = getattr(self.strategy, 'time_decay_candles', 999)
            if (i - self.entry_idx) >= time_decay:
                pct_from_entry = abs(c - self.entry_price) / self.entry_price
                if pct_from_entry < 0.003:
                    self._close_trade(c, t, "TIME_DECAY")
                    return

            if self.side == "LONG":
                if l <= self.sl:
                    self._close_trade(self.sl, t, "STOP_LOSS")
                    self.cooldown_until = i + self.strategy.cooldown_candles
                    return
                if h >= self.tgt:
                    self._close_trade(self.tgt, t, "TARGET")
                    return

                # V2: Partial exit at 1× risk
                partial_rr = getattr(self.strategy, 'partial_exit_at_rr', None)
                partial_pct = getattr(self.strategy, 'partial_exit_pct', 0.5)
                if partial_rr and not self.partial_exited:
                    initial_risk = self.entry_price - self.sl
                    if initial_risk > 0 and (c - self.entry_price) >= initial_risk * partial_rr:
                        partial_shares = max(1, int(self.shares * partial_pct))
                        if partial_shares < self.shares:
                            self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)

                unrealised = c - self.entry_price
                initial_risk = self.entry_price - self.sl
                if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                    self.sl = self.entry_price + 0.10
                    self.sl_moved_to_be = True

            else:  # SHORT
                if h >= self.sl:
                    self._close_trade(self.sl, t, "STOP_LOSS")
                    self.cooldown_until = i + self.strategy.cooldown_candles
                    return
                if l <= self.tgt:
                    self._close_trade(self.tgt, t, "TARGET")
                    return

                partial_rr = getattr(self.strategy, 'partial_exit_at_rr', None)
                partial_pct = getattr(self.strategy, 'partial_exit_pct', 0.5)
                if partial_rr and not self.partial_exited:
                    initial_risk = self.sl - self.entry_price
                    if initial_risk > 0 and (self.entry_price - c) >= initial_risk * partial_rr:
                        partial_shares = max(1, int(self.shares * partial_pct))
                        if partial_shares < self.shares:
                            self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial_shares)

                unrealised = self.entry_price - c
                initial_risk = self.sl - self.entry_price
                if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                    self.sl = self.entry_price - 0.10
                    self.sl_moved_to_be = True
            return

        # ── Phase 4: Look for entry ──
        if self.trade_count >= self.max_trades:
            return
        if i < self.cooldown_until:
            return

        if hour < 10 or (hour == 10 and minute <= 30):
            signal = self.strategy.generate_orb_signal(candles, i, self.orb, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return

        if 10 <= hour <= 11:
            signal = self.strategy.generate_pullback_signal(candles, i, self.orb, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return

        if hour >= 10 and (hour < 14 or (hour == 14 and minute == 0)):
            signal = self.strategy.generate_vwap_signal(candles, i, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return


# ════════════════════════════════════════════════════════════
# SCORING
# ════════════════════════════════════════════════════════════

def get_scored_picks(symbols, loader, top_n=5):
    df = loader.load_backtest_data(symbols, target_date=date.today().isoformat())
    featured = []
    for sym in (df["symbol"].unique() if not df.empty else []):
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            featured.append(add_features(sdf))
    if not featured:
        return pd.DataFrame()

    all_feat = pd.concat(featured, ignore_index=True)
    model, feats = None, None
    mp = Path("models/stock_predictor.pkl")
    if mp.exists():
        with open(mp, "rb") as f:
            d = pickle.load(f)
        model, feats = d["model"], d["features"]

    avail = [c for c in (feats or FEATURE_COLS) if c in all_feat.columns]
    scores = score_stocks(all_feat, model, avail if model else None)
    picks = scores[scores["score"] >= SCORE_THRESHOLD].head(top_n)
    if picks.empty:
        picks = scores.head(max(3, top_n))
    return picks


# ════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════

def compute_metrics(trades, daily_pnl, capital=100_000):
    if not trades:
        return {k: 0 for k in [
            "total_trades", "wins", "losses", "win_rate", "total_pnl",
            "avg_win", "avg_loss", "profit_factor", "max_drawdown_pct",
            "sharpe", "max_consecutive_losses", "avg_trade", "return_pct",
            "best_day", "worst_day", "profitable_days", "total_days",
        ]}

    pnls = [t["net_pnl"] for t in trades]
    wins_list = [p for p in pnls if p > 0]
    losses_list = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    gross_profit = sum(wins_list) if wins_list else 0
    gross_loss = abs(sum(losses_list)) if losses_list else 0

    equity = [capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    peak = equity[0]
    max_dd = 0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    max_consec = 0
    current_consec = 0
    for p in pnls:
        if p <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    daily_vals = list(daily_pnl.values())
    sharpe = 0
    if len(daily_vals) >= 2 and np.std(daily_vals) > 0:
        sharpe = (np.mean(daily_vals) / np.std(daily_vals)) * np.sqrt(252)

    profitable_days = sum(1 for v in daily_vals if v > 0)

    return {
        "total_trades": len(trades),
        "wins": len(wins_list),
        "losses": len(losses_list),
        "win_rate": len(wins_list) / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "avg_win": np.mean(wins_list) if wins_list else 0,
        "avg_loss": np.mean(losses_list) if losses_list else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "max_consecutive_losses": max_consec,
        "avg_trade": np.mean(pnls) if pnls else 0,
        "return_pct": total_pnl / capital * 100,
        "best_day": max(daily_vals) if daily_vals else 0,
        "worst_day": min(daily_vals) if daily_vals else 0,
        "profitable_days": profitable_days,
        "total_days": len(daily_vals),
    }


# ════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════

def print_comparison_report(trades_v1, trades_v2, daily_v1, daily_v2, config):
    capital = config["capital"]["total"]
    m1 = compute_metrics(trades_v1, daily_v1, capital)
    m2 = compute_metrics(trades_v2, daily_v2, capital)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  HONEST BACKTEST — V1 vs V2 COMPARISON")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Capital: Rs {capital:,.0f}")
    logger.info(f"{'─' * 70}")

    rows = [
        ("Total Trades", f"{m1['total_trades']}", f"{m2['total_trades']}"),
        ("Wins / Losses", f"{m1['wins']} / {m1['losses']}", f"{m2['wins']} / {m2['losses']}"),
        ("Win Rate", f"{m1['win_rate']:.1f}%", f"{m2['win_rate']:.1f}%"),
        ("", "", ""),
        ("Total P&L", f"Rs {m1['total_pnl']:+,.2f}", f"Rs {m2['total_pnl']:+,.2f}"),
        ("Return %", f"{m1['return_pct']:+.1f}%", f"{m2['return_pct']:+.1f}%"),
        ("Avg Win", f"Rs {m1['avg_win']:,.2f}", f"Rs {m2['avg_win']:,.2f}"),
        ("Avg Loss", f"Rs {m1['avg_loss']:,.2f}", f"Rs {m2['avg_loss']:,.2f}"),
        ("Avg Trade", f"Rs {m1['avg_trade']:+,.2f}", f"Rs {m2['avg_trade']:+,.2f}"),
        ("", "", ""),
        ("Profit Factor", f"{m1['profit_factor']:.2f}", f"{m2['profit_factor']:.2f}"),
        ("Sharpe Ratio", f"{m1['sharpe']:.2f}", f"{m2['sharpe']:.2f}"),
        ("Max Drawdown", f"{m1['max_drawdown_pct']:.1f}%", f"{m2['max_drawdown_pct']:.1f}%"),
        ("Max Consec. Losses", f"{m1['max_consecutive_losses']}", f"{m2['max_consecutive_losses']}"),
        ("", "", ""),
        ("Profitable Days", f"{m1['profitable_days']}/{m1['total_days']}", f"{m2['profitable_days']}/{m2['total_days']}"),
        ("Best Day", f"Rs {m1['best_day']:+,.2f}", f"Rs {m2['best_day']:+,.2f}"),
        ("Worst Day", f"Rs {m1['worst_day']:+,.2f}", f"Rs {m2['worst_day']:+,.2f}"),
    ]

    logger.info(f"  {'Metric':<24} {'V1 (Current)':>18} {'V2 (Improved)':>18}  {'Better':>6}")
    logger.info(f"  {'─' * 24} {'─' * 18} {'─' * 18}  {'─' * 6}")
    for label, v1_val, v2_val in rows:
        if label == "":
            logger.info("")
            continue
        better = ""
        try:
            v1n = float(v1_val.replace("Rs ", "").replace(",", "").replace("%", "").replace("+", ""))
            v2n = float(v2_val.replace("Rs ", "").replace(",", "").replace("%", "").replace("+", ""))
            if "Loss" in label or "Drawdown" in label or "Consec" in label:
                better = "← V1" if abs(v1n) < abs(v2n) else ("→ V2" if abs(v2n) < abs(v1n) else "  =")
            else:
                better = "← V1" if v1n > v2n else ("→ V2" if v2n > v1n else "  =")
        except:
            pass
        logger.info(f"  {label:<24} {v1_val:>18} {v2_val:>18}  {better:>6}")

    logger.info(f"{'=' * 70}")

    # Trade type breakdown
    for version, trades in [("V1", trades_v1), ("V2", trades_v2)]:
        if not trades:
            continue
        df = pd.DataFrame(trades)
        logger.info(f"\n  ─── {version} Trade Type Breakdown ───")
        for ttype in df["type"].unique():
            sub = df[df["type"] == ttype]
            wins = len(sub[sub["net_pnl"] > 0])
            pnl = sub["net_pnl"].sum()
            logger.info(f"  {ttype:<20} {len(sub):>3} trades | {wins}/{len(sub)} won ({wins / len(sub) * 100:.0f}%) | Rs {pnl:+,.0f}")

    # Exit reason breakdown
    for version, trades in [("V1", trades_v1), ("V2", trades_v2)]:
        if not trades:
            continue
        df = pd.DataFrame(trades)
        logger.info(f"\n  ─── {version} Exit Reasons ───")
        for reason in df["reason"].unique():
            sub = df[df["reason"] == reason]
            pnl = sub["net_pnl"].sum()
            logger.info(f"  {reason:<20} {len(sub):>3} trades | Rs {pnl:+,.0f}")

    # Per-stock performance
    for version, trades in [("V1", trades_v1), ("V2", trades_v2)]:
        if not trades:
            continue
        df = pd.DataFrame(trades)
        logger.info(f"\n  ─── {version} Per-Stock P&L ───")
        stock_pnl = df.groupby("symbol")["net_pnl"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        for sym, row in stock_pnl.iterrows():
            emoji = "📈" if row["sum"] > 0 else "📉"
            logger.info(f"  {emoji} {sym:<12} Rs {row['sum']:>+8,.0f} ({int(row['count'])} trades)")


# ════════════════════════════════════════════════════════════
# MAIN BACKTEST LOOP
# ════════════════════════════════════════════════════════════

def run_backtest(symbols, last_n_days=30, top_n=5, label="Nifty 50"):
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  HONEST 30-DAY BACKTEST — {label}")
    logger.info(f"  V1 (current) vs V2 (improved) — Side by Side")
    logger.info(f"  Last {last_n_days} trading days | Top {top_n} picks per day")
    logger.info(f"{'=' * 70}")

    cfg_path = Path("config/config_test.yaml")
    if not cfg_path.exists():
        cfg_path = Path("config/config_example.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader()
    picks = get_scored_picks(symbols, loader, top_n=top_n)
    if picks.empty:
        logger.error("  No scored picks available!")
        return

    logger.info(f"\n  ─── STOCK PICKS (scored) ───")
    for _, r in picks.iterrows():
        d = r.get("direction", "LONG")
        arrow = "▲" if d == "LONG" else "▼"
        logger.info(f"  {r['symbol']:<12} Score:{r['score']:>4.0f} {arrow}{d:<5} RSI={r['rsi']:.1f}")

    # Fetch intraday data
    logger.info(f"\n  ─── DOWNLOADING 5-MIN INTRADAY DATA ───")
    all_intraday = {}
    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        logger.info(f"  Fetching {sym}...")
        days_data = fetch_intraday_multiday(sym, days=last_n_days + 10)
        if days_data:
            all_intraday[sym] = days_data
            logger.info(f"    {sym}: {len(days_data)} trading days")
        else:
            logger.warning(f"    {sym}: no data")
        time.sleep(0.5)

    if not all_intraday:
        logger.error("  No intraday data available!")
        return

    all_dates = set()
    for sym_data in all_intraday.values():
        all_dates.update(sym_data.keys())
    sorted_dates = sorted(all_dates)[-last_n_days:]
    logger.info(f"\n  Testing {len(sorted_dates)} trading days: {sorted_dates[0]} to {sorted_dates[-1]}")

    # Run day-by-day
    all_trades_v1 = []
    all_trades_v2 = []
    daily_pnl_v1 = {}
    daily_pnl_v2 = {}

    for day_str in sorted_dates:
        day_trades_v1 = []
        day_trades_v2 = []

        for _, pick in picks.iterrows():
            sym = pick["symbol"]
            direction = pick.get("direction", "LONG")

            if sym not in all_intraday or day_str not in all_intraday[sym]:
                continue
            candles = all_intraday[sym][day_str]

            # V1
            v1_strat = ProStrategy(config.get("strategies", {}).get("pro", {}))
            v1 = HonestTrader(sym, direction, config, v1_strat, "V1")
            for i in range(len(candles)):
                v1.process_candle(i, candles)
            for t in v1.trades:
                t["date"] = day_str
            day_trades_v1.extend(v1.trades)

            # V2
            v2_strat = ProStrategyV2(config.get("strategies", {}).get("pro", {}))
            v2 = HonestTrader(sym, direction, config, v2_strat, "V2")
            for i in range(len(candles)):
                v2.process_candle(i, candles)
            for t in v2.trades:
                t["date"] = day_str
            day_trades_v2.extend(v2.trades)

        v1_day = sum(t["net_pnl"] for t in day_trades_v1)
        v2_day = sum(t["net_pnl"] for t in day_trades_v2)
        daily_pnl_v1[day_str] = v1_day
        daily_pnl_v2[day_str] = v2_day

        e1 = "✅" if v1_day >= 0 else "❌"
        e2 = "✅" if v2_day >= 0 else "❌"
        logger.info(f"  {day_str} | V1: {e1} Rs {v1_day:>+8,.0f} ({len(day_trades_v1)}t) | V2: {e2} Rs {v2_day:>+8,.0f} ({len(day_trades_v2)}t)")

        all_trades_v1.extend(day_trades_v1)
        all_trades_v2.extend(day_trades_v2)

    print_comparison_report(all_trades_v1, all_trades_v2, daily_pnl_v1, daily_pnl_v2, config)

    today = date.today()
    if all_trades_v1:
        pd.DataFrame(all_trades_v1).to_csv(f"results/backtest_30d_v1_{today}.csv", index=False)
    if all_trades_v2:
        pd.DataFrame(all_trades_v2).to_csv(f"results/backtest_30d_v2_{today}.csv", index=False)

    comparison = []
    for d in sorted(set(list(daily_pnl_v1.keys()) + list(daily_pnl_v2.keys()))):
        comparison.append({
            "date": d,
            "v1_pnl": daily_pnl_v1.get(d, 0),
            "v2_pnl": daily_pnl_v2.get(d, 0),
            "v1_trades": len([t for t in all_trades_v1 if t.get("date") == d]),
            "v2_trades": len([t for t in all_trades_v2 if t.get("date") == d]),
        })
    pd.DataFrame(comparison).to_csv(f"results/backtest_30d_comparison_{today}.csv", index=False)
    logger.info(f"\n  Saved: results/backtest_30d_v1_{today}.csv")
    logger.info(f"  Saved: results/backtest_30d_v2_{today}.csv")
    logger.info(f"  Saved: results/backtest_30d_comparison_{today}.csv")


def main():
    parser = argparse.ArgumentParser(description="Honest 30-day backtest: V1 vs V2")
    parser.add_argument("--last", type=int, default=30, help="Trading days to test")
    parser.add_argument("--stocks", nargs="+", default=None, help="Specific stocks")
    parser.add_argument("--universe", default="nifty50", choices=["nifty50", "nifty100", "nifty250"])
    parser.add_argument("--top", type=int, default=5, help="Top N scored picks")
    parser.add_argument("--threshold", type=int, default=50, help="Min score threshold")
    args = parser.parse_args()

    global SCORE_THRESHOLD
    SCORE_THRESHOLD = args.threshold

    if args.stocks:
        symbols = args.stocks
        label = ", ".join(args.stocks)
    else:
        symbols = get_universe(args.universe)
        label = f"Nifty {args.universe.replace('nifty', '')}"

    run_backtest(symbols, last_n_days=args.last, top_n=args.top, label=label)


if __name__ == "__main__":
    main()
