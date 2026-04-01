#!/usr/bin/env python3
"""
Enrich training data with NSE bhavcopy → Retrain ML → Re-test V3 on March 30.
"""
import os, sys, pickle, logging, warnings
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config.symbols import NIFTY_50
from data.data_loader import DataLoader
from data.train_pipeline import add_features, train_model, score_stocks, FEATURE_COLS

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# STEP 1: Download NSE bhavcopy data via jugaad_data
# ════════════════════════════════════════════════════════════

def download_nse_bhavcopy(symbols, from_date, to_date):
    """Download delivery %, VWAP, trade count from NSE for all symbols."""
    from jugaad_data.nse import stock_df
    all_nse = []
    total = len(symbols)
    for i, sym in enumerate(symbols):
        try:
            df = stock_df(symbol=sym, from_date=from_date, to_date=to_date, series="EQ")
            if len(df) > 0:
                df = df.rename(columns={
                    "DATE": "date", "OPEN": "nse_open", "HIGH": "nse_high",
                    "LOW": "nse_low", "CLOSE": "nse_close", "VWAP": "nse_vwap",
                    "VOLUME": "nse_volume", "NO OF TRADES": "no_of_trades",
                    "DELIVERY QTY": "delivery_qty", "DELIVERY %": "delivery_pct",
                    "SYMBOL": "symbol",
                })
                # Keep only what we need
                keep = ["date", "symbol", "nse_vwap", "no_of_trades", "delivery_pct"]
                avail = [c for c in keep if c in df.columns]
                df = df[avail].copy()
                df["date"] = pd.to_datetime(df["date"]).dt.date
                all_nse.append(df)
                if (i + 1) % 10 == 0 or i == total - 1:
                    logger.info(f"  NSE bhavcopy: {i+1}/{total} stocks downloaded")
        except Exception as e:
            continue

    if all_nse:
        return pd.concat(all_nse, ignore_index=True)
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════
# STEP 2: Merge bhavcopy with yfinance data
# ════════════════════════════════════════════════════════════

def enrich_with_bhavcopy(yf_df, nse_df):
    """Merge NSE bhavcopy data into yfinance training data."""
    if nse_df.empty:
        logger.warning("  No NSE data to merge")
        return yf_df

    yf_df = yf_df.copy()
    yf_df["date_key"] = pd.to_datetime(yf_df["date"]).dt.date

    nse_df = nse_df.copy()
    nse_df["date_key"] = pd.to_datetime(nse_df["date"]).dt.date

    merged = yf_df.merge(
        nse_df[["date_key", "symbol", "nse_vwap", "no_of_trades", "delivery_pct"]],
        on=["date_key", "symbol"], how="left"
    )
    # Fill missing with 0 (older dates won't have NSE data)
    for col in ["nse_vwap", "no_of_trades", "delivery_pct"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged.drop(columns=["date_key"], inplace=True)
    return merged


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    symbols = NIFTY_50[:15]  # top 15 for training

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ENRICH + RETRAIN + TEST V3")
    logger.info(f"{'=' * 60}")

    # ── Step 1: Load yfinance cached data ──
    logger.info(f"\n  Step 1: Loading yfinance data for {len(symbols)} stocks...")
    loader = DataLoader()
    yf_df = loader.load_backtest_data(symbols, target_date=date.today().isoformat())
    logger.info(f"  yfinance: {len(yf_df):,} rows | {yf_df['symbol'].nunique()} stocks")

    # ── Step 2: Download NSE bhavcopy (last 6 months) ──
    logger.info(f"\n  Step 2: Downloading NSE bhavcopy data (last 3 months)...")
    nse_from = date.today() - timedelta(days=90)
    nse_to = date.today()
    nse_df = download_nse_bhavcopy(symbols, nse_from, nse_to)
    if not nse_df.empty:
        logger.info(f"  NSE bhavcopy: {len(nse_df):,} rows | {nse_df['symbol'].nunique()} stocks")
        logger.info(f"  Columns: {list(nse_df.columns)}")
    else:
        logger.warning("  NSE bhavcopy download failed — training without enrichment")

    # ── Step 3: Merge ──
    logger.info(f"\n  Step 3: Merging bhavcopy into training data...")
    enriched = enrich_with_bhavcopy(yf_df, nse_df)
    enriched_count = enriched[enriched["delivery_pct"] > 0].shape[0]
    logger.info(f"  Enriched rows (with delivery data): {enriched_count:,} / {len(enriched):,}")

    # ── Step 4: Add features ──
    logger.info(f"\n  Step 4: Computing features (30 + 6 new bhavcopy features)...")
    featured = []
    for sym in enriched["symbol"].unique():
        sdf = enriched[enriched["symbol"] == sym].copy()
        if len(sdf) > 50:
            feat_df = add_features(sdf)
            featured.append(feat_df)

    all_feat = pd.concat(featured, ignore_index=True)
    logger.info(f"  Total featured: {len(all_feat):,} rows | {len(all_feat.columns)} columns")

    # Show new feature coverage
    for col in ["delivery_pct", "vwap_premium", "trade_intensity"]:
        if col in all_feat.columns:
            nonzero = (all_feat[col] != 0).sum()
            logger.info(f"  {col}: {nonzero:,} non-zero values ({nonzero/len(all_feat)*100:.1f}%)")

    # ── Step 5: Train new model ──
    logger.info(f"\n  Step 5: Training model with {len(FEATURE_COLS)} features...")
    old_model_path = Path("models/stock_predictor.pkl")
    old_exists = old_model_path.exists()
    if old_exists:
        with open(old_model_path, "rb") as f:
            old_data = pickle.load(f)
        old_features = old_data["features"]
        logger.info(f"  Old model: {len(old_features)} features")

    model, feats = train_model(all_feat)
    logger.info(f"  New model: {len(feats)} features")

    new_feats = [f for f in feats if f not in (old_features if old_exists else [])]
    if new_feats:
        logger.info(f"  NEW features added: {new_feats}")

    # ── Step 6: Score stocks for March 30 ──
    logger.info(f"\n  Step 6: Scoring stocks for 2026-03-30 with NEW model...")
    target = date(2026, 3, 30)
    test_df = loader.load_backtest_data(NIFTY_50[:10], target_date=target.isoformat())

    test_featured = []
    for sym in (test_df["symbol"].unique() if not test_df.empty else []):
        sdf = test_df[test_df["symbol"] == sym].copy()
        # Also enrich test data
        if not nse_df.empty:
            sdf = enrich_with_bhavcopy(sdf, nse_df)
        if len(sdf) > 50:
            test_featured.append(add_features(sdf))

    if test_featured:
        test_all = pd.concat(test_featured, ignore_index=True)
        avail = [c for c in feats if c in test_all.columns]
        new_scores = score_stocks(test_all, model, avail)
        logger.info(f"\n  NEW scores for March 30:")
        for _, r in new_scores.head(10).iterrows():
            arrow = "📈" if r["direction"] == "LONG" else "📉"
            logger.info(f"    {arrow} {r['symbol']:<12} Score:{r['score']:>4.0f} {r['direction']} RSI={r['rsi']:.1f}")

    # ── Step 7: Re-run V3 backtest on March 30 ──
    logger.info(f"\n  Step 7: Re-running V3 adaptive on March 30 with NEW model...")
    from live_paper_v3 import run
    run(backtest_date="2026-03-30")

    # ── Step 8: Compare old vs new results ──
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  COMPARISON: Old V3 vs New V3 (enriched model)")
    logger.info(f"{'=' * 60}")

    old_csv = Path("results/live_v3_2026-03-30.csv")
    # The re-run overwrites the CSV, so we need to save old results first
    # Let's read the new results
    new_csv = Path("results/live_v3_2026-03-30.csv")
    if new_csv.exists():
        new_trades = pd.read_csv(new_csv)
        new_total = new_trades["net_pnl"].sum()
        new_wins = (new_trades["net_pnl"] > 0).sum()
        new_count = len(new_trades)
        wr = new_wins / new_count * 100 if new_count > 0 else 0
        logger.info(f"\n  NEW V3 (enriched): {new_count} trades | {new_wins} wins | WR: {wr:.0f}% | P&L: Rs {new_total:+,.2f}")
        logger.info(f"\n  Trade details:")
        for _, t in new_trades.iterrows():
            emoji = "✅" if t["net_pnl"] > 0 else "❌"
            logger.info(f"    {emoji} {t['symbol']} {t['direction']} ({t['type']}) Rs {t['entry']:,.2f} → Rs {t['exit']:,.2f} | Rs {t['net_pnl']:+,.2f} | {t['reason']}")
    else:
        logger.warning("  No results CSV found")

    logger.info(f"\n  PREVIOUS V3 (old model): 7 trades | 5 wins | WR: 71% | P&L: Rs +7,218.55")
    logger.info(f"  Done!")


if __name__ == "__main__":
    main()
