#!/usr/bin/env python3
"""
FULL PIPELINE: Enrich data → Retrain ML model → Run V3 on last 7 trading days.
Each day is treated as a FRESH day — no peeking ahead.

Usage: python enrich_retrain_backtest.py
"""
import os, sys, time, pickle, logging, json, warnings
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config.symbols import NIFTY_50, get_universe
from data.data_loader import DataLoader
from data.train_pipeline import add_features, FEATURE_COLS

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ═══════════════════════════════════════════
# STEP 1: Download NSE bhavcopy enrichment
# ═══════════════════════════════════════════

def download_nse_bhavcopy(symbols, months_back=6):
    """Download delivery %, VWAP, trade count from NSE via jugaad_data."""
    logger.info(f"\n  STEP 1: Downloading NSE bhavcopy for {len(symbols)} stocks ({months_back} months)...")
    from jugaad_data.nse import stock_df
    end_date = date.today()
    start_date = end_date - timedelta(days=months_back * 30)
    
    all_data = []
    for i, sym in enumerate(symbols):
        try:
            df = stock_df(symbol=sym, from_date=start_date, to_date=end_date, series="EQ")
            if len(df) > 0:
                df = df.rename(columns={
                    "DATE": "date", "OPEN": "open", "HIGH": "high", "LOW": "low",
                    "CLOSE": "close", "VOLUME": "volume", "VWAP": "vwap",
                    "NO OF TRADES": "num_trades", "DELIVERY %": "delivery_pct",
                    "SYMBOL": "symbol", "PREV. CLOSE": "prev_close",
                })
                df["date"] = pd.to_datetime(df["date"]).dt.date
                keep_cols = [c for c in ["date","open","high","low","close","volume","vwap",
                             "num_trades","delivery_pct","symbol","prev_close"] if c in df.columns]
                all_data.append(df[keep_cols])
                if (i + 1) % 10 == 0:
                    logger.info(f"    {i+1}/{len(symbols)} done ({sym}: {len(df)} rows)")
            time.sleep(0.3)  # be nice to NSE
        except Exception as e:
            logger.warning(f"    {sym}: {e}")
            continue
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        outpath = Path("data/nse_bhavcopy.csv")
        result.to_csv(outpath, index=False)
        logger.info(f"  Saved {len(result):,} rows for {result['symbol'].nunique()} stocks to {outpath}")
        return result
    return pd.DataFrame()


# ═══════════════════════════════════════════
# STEP 2: Enhanced features with candles + NSE
# ═══════════════════════════════════════════

ENHANCED_FEATURES = FEATURE_COLS + [
    "delivery_pct", "vwap_distance", "num_trades_ratio",
    "pat_doji", "pat_dragonfly", "pat_gravestone",
    "pat_bullish_harami", "pat_bearish_harami",
    "pat_tweezer_top", "pat_tweezer_bottom",
    "pat_three_inside_up", "pat_three_inside_down",
    "gap_pct", "body_to_range", "upper_shadow_pct", "lower_shadow_pct",
    "consecutive_green", "consecutive_red",
]


def add_enhanced_features(df, nse_data=None):
    """add_features() + candlestick patterns + NSE enrichment."""
    df = add_features(df)
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    body = abs(c - o)
    rng = h - l
    
    # ── Extra candlestick patterns ──
    uw = h - pd.concat([c, o], axis=1).max(axis=1)
    lw = pd.concat([c, o], axis=1).min(axis=1) - l
    
    # Doji (already exists but ensure)
    df["pat_doji"] = (body < rng * 0.1).astype(int)
    
    # Dragonfly doji (long lower shadow, no upper shadow)
    df["pat_dragonfly"] = ((lw > rng * 0.6) & (uw < rng * 0.1) & (body < rng * 0.1)).astype(int)
    
    # Gravestone doji (long upper shadow, no lower shadow)
    df["pat_gravestone"] = -((uw > rng * 0.6) & (lw < rng * 0.1) & (body < rng * 0.1)).astype(int)
    
    # Bullish harami
    po, pc = o.shift(1), c.shift(1)
    prev_body = abs(pc - po)
    df["pat_bullish_harami"] = ((pc < po) & (c > o) & (o > pc) & (c < po) & (body < prev_body * 0.5)).astype(int)
    
    # Bearish harami
    df["pat_bearish_harami"] = -((pc > po) & (c < o) & (o < pc) & (c > po) & (body < prev_body * 0.5)).astype(int)
    
    # Tweezer top (same highs, second bearish)
    df["pat_tweezer_top"] = -((abs(h - h.shift(1)) / h < 0.001) & (c.shift(1) > o.shift(1)) & (c < o)).astype(int)
    
    # Tweezer bottom (same lows, second bullish)
    df["pat_tweezer_bottom"] = ((abs(l - l.shift(1)) / l < 0.001) & (c.shift(1) < o.shift(1)) & (c > o)).astype(int)
    
    # Three inside up/down
    df["pat_three_inside_up"] = ((df.get("pat_bullish_harami", 0).shift(1) > 0) & (c > h.shift(1))).astype(int)
    df["pat_three_inside_down"] = -((df.get("pat_bearish_harami", 0).shift(1) < 0) & (c < l.shift(1))).astype(int)
    
    # ── Candle anatomy features ──
    df["gap_pct"] = (o - c.shift(1)) / c.shift(1)
    df["body_to_range"] = np.where(rng > 0, body / rng, 0)
    df["upper_shadow_pct"] = np.where(rng > 0, uw / rng, 0)
    df["lower_shadow_pct"] = np.where(rng > 0, lw / rng, 0)
    
    # Consecutive green/red candles
    green = (c > o).astype(int)
    red = (c < o).astype(int)
    consec_g, consec_r = [0], [0]
    for i in range(1, len(df)):
        consec_g.append(consec_g[-1] + 1 if green.iloc[i] else 0)
        consec_r.append(consec_r[-1] + 1 if red.iloc[i] else 0)
    df["consecutive_green"] = consec_g
    df["consecutive_red"] = consec_r
    
    # ── NSE enrichment (merge by symbol + date) ──
    if nse_data is not None and len(nse_data) > 0:
        sym = df["symbol"].iloc[0] if "symbol" in df.columns else None
        if sym:
            nse_sym = nse_data[nse_data["symbol"] == sym].copy()
            if len(nse_sym) > 0:
                nse_sym["date"] = pd.to_datetime(nse_sym["date"]).dt.strftime("%Y-%m-%d")
                df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                nse_map = nse_sym.set_index("date_str")
                df["delivery_pct"] = df["date_str"].map(nse_map.get("delivery_pct", {})).fillna(0)
                df["vwap_distance"] = 0
                if "vwap" in nse_sym.columns:
                    vwap_map = nse_map["vwap"].to_dict() if "vwap" in nse_map.columns else {}
                    nse_vwap = df["date_str"].map(vwap_map).fillna(0)
                    df["vwap_distance"] = np.where(nse_vwap > 0, (c - nse_vwap) / nse_vwap, 0)
                if "num_trades" in nse_sym.columns:
                    nt_map = nse_map["num_trades"].to_dict() if "num_trades" in nse_map.columns else {}
                    nt = df["date_str"].map(nt_map).fillna(0)
                    avg_nt = nt.rolling(20, min_periods=1).mean()
                    df["num_trades_ratio"] = np.where(avg_nt > 0, nt / avg_nt, 1)
                df.drop(columns=["date_str"], inplace=True, errors="ignore")
    
    # Fill any missing enhanced features with 0
    for col in ENHANCED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    
    return df


# ═══════════════════════════════════════════
# STEP 3: Retrain ML model with enriched data
# ═══════════════════════════════════════════

def retrain_model(symbols, nse_data=None):
    logger.info(f"\n  STEP 3: Retraining ML model with {len(ENHANCED_FEATURES)} features...")
    
    loader = DataLoader()
    df = loader.load_backtest_data(symbols, target_date=date.today().isoformat())
    
    featured = []
    for sym in (df["symbol"].unique() if not df.empty else []):
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            feat = add_enhanced_features(sdf, nse_data)
            featured.append(feat)
    
    if not featured:
        logger.error("  No data for training!")
        return None, None
    
    all_feat = pd.concat(featured, ignore_index=True)
    
    # Create target: next day return > 0 = LONG (1), else SHORT (0)
    all_feat["future_ret"] = all_feat.groupby("symbol")["close"].pct_change(1).shift(-1)
    all_feat["target"] = (all_feat["future_ret"] > 0).astype(int)
    all_feat = all_feat.dropna(subset=["target", "future_ret"])
    
    avail = [c for c in ENHANCED_FEATURES if c in all_feat.columns]
    X = all_feat[avail].fillna(0)
    y = all_feat["target"]
    
    if len(X) < 100:
        logger.error(f"  Not enough data: {len(X)} rows")
        return None, None
    
    # Train/test split (time-based)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    logger.info(f"  Train accuracy: {train_acc:.1%} | Test accuracy: {test_acc:.1%}")
    logger.info(f"  Features: {len(avail)} | Samples: {len(X):,} ({len(X_train):,} train, {len(X_test):,} test)")
    
    # Feature importance top 10
    imp = pd.Series(model.feature_importances_, index=avail).sort_values(ascending=False)
    logger.info(f"\n  Top 10 features:")
    for feat, score in imp.head(10).items():
        logger.info(f"    {feat:<25} {score:.4f}")
    
    # Save
    model_path = Path("models/stock_predictor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": avail, "enhanced": True,
                      "train_acc": train_acc, "test_acc": test_acc,
                      "trained_on": str(date.today()), "n_features": len(avail)}, f)
    logger.info(f"\n  Model saved: {model_path}")
    
    return model, avail


# ═══════════════════════════════════════════
# STEP 4: Find last 7 trading days
# ═══════════════════════════════════════════

def find_last_n_trading_days(n=7):
    """Find last N trading days using yfinance (no hardcoded holidays)."""
    import yfinance as yf
    logger.info(f"\n  STEP 4: Finding last {n} trading days (live API)...")
    
    # Get NIFTY 50 daily data for last month
    df = yf.Ticker("^NSEI").history(period="1mo", interval="1d")
    if df.empty:
        logger.error("  Cannot fetch NIFTY data!")
        return []
    
    # Get unique trading dates (excluding today if market not closed)
    trading_days = sorted([d.date() for d in df.index if d.date() < date.today()])
    last_n = trading_days[-n:] if len(trading_days) >= n else trading_days
    
    for d in last_n:
        logger.info(f"    {d} ({d.strftime('%A')})")
    
    return last_n


# ═══════════════════════════════════════════
# STEP 5: Run V3 on each day — FRESH, no peeking
# ═══════════════════════════════════════════

def run_v3_day(target_date, config):
    """Run V3 on one day — imports fresh, no shared state."""
    from live_paper_v3 import run
    import io, contextlib
    
    # Capture output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        run(backtest_date=target_date.isoformat())
    
    # Read results
    csv_path = Path(f"results/live_v3_{target_date}.csv")
    if csv_path.exists():
        trades = pd.read_csv(csv_path)
        return trades
    return pd.DataFrame()


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    import yaml
    for p in ["config/config_test.yaml", "config/config_prod.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                config = yaml.safe_load(f)
            break
    
    symbols = NIFTY_50
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  FULL PIPELINE: ENRICH → RETRAIN → 7-DAY BACKTEST")
    logger.info(f"{'=' * 60}")
    
    # STEP 1: Download NSE bhavcopy data
    nse_path = Path("data/nse_bhavcopy.csv")
    if nse_path.exists() and (date.today() - date.fromtimestamp(nse_path.stat().st_mtime)).days < 1:
        logger.info(f"  Using cached NSE bhavcopy ({nse_path})")
        nse_data = pd.read_csv(nse_path)
    else:
        nse_data = download_nse_bhavcopy(symbols, months_back=6)
    
    # STEP 2: Count enhanced features
    logger.info(f"\n  STEP 2: Enhanced features: {len(ENHANCED_FEATURES)}")
    logger.info(f"  New candle patterns: doji, dragonfly, gravestone, harami, tweezer, three_inside")
    logger.info(f"  New candle anatomy: gap_pct, body_to_range, shadow_pct, consecutive streaks")
    logger.info(f"  NSE enrichment: delivery_pct, vwap_distance, num_trades_ratio")
    
    # STEP 3: Retrain
    model, feats = retrain_model(symbols, nse_data)
    if model is None:
        logger.error("  Training failed!")
        return
    
    # STEP 4: Find trading days
    trading_days = find_last_n_trading_days(7)
    if not trading_days:
        return
    
    # STEP 5: Run V3 on each day
    logger.info(f"\n  STEP 5: Running V3 on {len(trading_days)} trading days...")
    logger.info(f"  Each day is FRESH — no peeking, no shared state")
    logger.info(f"  Telegram messages will be sent for every trade\n")
    
    all_results = []
    
    for day in trading_days:
        logger.info(f"\n  {'═' * 50}")
        logger.info(f"  DAY: {day} ({day.strftime('%A')})")
        logger.info(f"  {'═' * 50}")
        
        trades = run_v3_day(day, config)
        
        if len(trades) > 0:
            day_pnl = trades["net_pnl"].sum()
            wins = (trades["net_pnl"] > 0).sum()
            losses = len(trades) - wins
            logger.info(f"\n  {day}: {len(trades)} trades | {wins}W/{losses}L | Rs {day_pnl:+,.2f}")
            all_results.append({
                "date": str(day), "day": day.strftime("%A"),
                "trades": len(trades), "wins": int(wins), "losses": int(losses),
                "pnl": round(float(day_pnl), 2),
                "wr": round(float(wins / len(trades) * 100), 1) if len(trades) > 0 else 0,
            })
        else:
            logger.info(f"\n  {day}: No trades")
            all_results.append({
                "date": str(day), "day": day.strftime("%A"),
                "trades": 0, "wins": 0, "losses": 0, "pnl": 0, "wr": 0,
            })
    
    # ═══════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════
    
    total_pnl = sum(r["pnl"] for r in all_results)
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"  7-DAY V3 ADAPTIVE BACKTEST REPORT")
    print(f"  Model: Enhanced ({len(feats)} features incl. candle patterns + NSE data)")
    print(f"{'=' * 70}")
    print(f"\n  {'Date':<14} {'Day':<10} {'Trades':>6} {'Wins':>5} {'WR%':>5} {'P&L':>12}")
    print(f"  {'─' * 55}")
    for r in all_results:
        emoji = "✅" if r["pnl"] > 0 else ("❌" if r["pnl"] < 0 else "➖")
        print(f"  {emoji}{r['date']:<13} {r['day']:<10} {r['trades']:>6} {r['wins']:>5} {r['wr']:>4.0f}% {r['pnl']:>+11,.2f}")
    print(f"  {'─' * 55}")
    print(f"  {'TOTAL':<24} {total_trades:>6} {total_wins:>5} {avg_wr:>4.0f}% {total_pnl:>+11,.2f}")
    print(f"\n  Capital: Rs 1,00,000 | Return: {total_pnl/100000*100:+.2f}%")
    
    # Save summary
    with open("results/7day_backtest_summary.json", "w") as f:
        json.dump({"results": all_results, "total_pnl": total_pnl,
                    "total_trades": total_trades, "win_rate": avg_wr,
                    "model_features": len(feats)}, f, indent=2)
    print(f"\n  Results saved: results/7day_backtest_summary.json")
    
    # Telegram summary
    from live_paper_v3 import send_telegram, load_config
    cfg = load_config()
    tg_lines = [f"📊 7-DAY V3 BACKTEST", f"Model: {len(feats)} features (enhanced)", ""]
    for r in all_results:
        emoji = "✅" if r["pnl"] > 0 else "❌" if r["pnl"] < 0 else "➖"
        tg_lines.append(f"{emoji} {r['date']} | {r['trades']}T {r['wins']}W | Rs {r['pnl']:+,.0f}")
    tg_lines.append(f"\n💰 TOTAL: Rs {total_pnl:+,.0f} | {total_trades}T | WR: {avg_wr:.0f}%")
    send_telegram("\n".join(tg_lines), cfg)


if __name__ == "__main__":
    main()
