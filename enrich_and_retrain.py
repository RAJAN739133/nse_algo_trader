#!/usr/bin/env python3
"""
Enrich training data with NSE bhavcopy (delivery %, VWAP, trade count) + retrain ML model.
Downloads from jugaad_data → merges with yfinance cache → trains with 30+ features.
"""
import os, sys, logging, pickle
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config.symbols import NIFTY_50
from data.data_loader import DataLoader
from data.train_pipeline import add_features, FEATURE_COLS

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("data/cache")


def download_bhavcopy_enrichment(symbols, years_back=2):
    """Download delivery %, VWAP, trade count from NSE via jugaad_data for each stock."""
    try:
        from jugaad_data.nse import stock_df
    except ImportError:
        logger.error("jugaad_data not installed. Run: pip install jugaad-data")
        return {}

    end_date = date.today()
    start_date = end_date - timedelta(days=years_back * 365)
    enrichment = {}

    for i, sym in enumerate(symbols):
        logger.info(f"  [{i+1}/{len(symbols)}] Downloading NSE bhavcopy for {sym}...")
        try:
            df = stock_df(symbol=sym, from_date=start_date, to_date=end_date, series="EQ")
            if len(df) > 0:
                df = df.rename(columns={
                    "DATE": "date", "OPEN": "open", "HIGH": "high", "LOW": "low",
                    "CLOSE": "close", "VOLUME": "volume", "VWAP": "nse_vwap",
                    "DELIVERY %": "delivery_pct", "NO OF TRADES": "no_of_trades",
                    "SYMBOL": "symbol",
                })
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                keep_cols = ["date", "nse_vwap", "delivery_pct", "no_of_trades"]
                available = [c for c in keep_cols if c in df.columns]
                enrichment[sym] = df[available].copy()
                logger.info(f"    {sym}: {len(df)} rows with delivery/VWAP data")
            else:
                logger.warning(f"    {sym}: no bhavcopy data")
        except Exception as e:
            logger.warning(f"    {sym}: bhavcopy fetch error — {e}")

    return enrichment


def merge_enrichment_with_cache(symbols, enrichment):
    """Merge bhavcopy enrichment columns into cached yfinance data."""
    loader = DataLoader()
    all_data = []

    for sym in symbols:
        cache_file = CACHE_DIR / f"{sym}_1d_5y.csv"
        if not cache_file.exists():
            try:
                df = loader.load_backtest_data([sym], target_date=date.today().isoformat())
                df = df[df["symbol"] == sym]
            except:
                continue
        else:
            df = pd.read_csv(cache_file)

        if len(df) < 50:
            continue

        # Merge NSE enrichment
        if sym in enrichment:
            nse = enrichment[sym]
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            nse["date"] = pd.to_datetime(nse["date"]).dt.strftime("%Y-%m-%d")
            df = df.merge(nse, on="date", how="left")
            # Forward-fill missing enrichment
            for col in ["nse_vwap", "delivery_pct", "no_of_trades"]:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)

        if "symbol" not in df.columns:
            df["symbol"] = sym

        all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def train_enhanced_model(df):
    """Train model with all 30 features including bhavcopy enrichment."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    # Add features (will now include delivery_pct, vwap_premium, trade_intensity)
    featured = []
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            feat_df = add_features(sdf)
            featured.append(feat_df)

    if not featured:
        logger.error("No data to train on!")
        return None, None

    all_feat = pd.concat(featured, ignore_index=True)

    # Create target: next-day return > 0 = LONG (1), else SHORT (0)
    all_feat["target"] = (all_feat.groupby("symbol")["close"].shift(-1) > all_feat["close"]).astype(int)
    all_feat = all_feat.dropna(subset=["target"])

    # Use all available features from FEATURE_COLS
    avail_feats = [c for c in FEATURE_COLS if c in all_feat.columns]
    logger.info(f"  Training features ({len(avail_feats)}): {avail_feats}")

    X = all_feat[avail_feats].fillna(0)
    y = all_feat["target"]

    # Check feature coverage
    nse_feats = [f for f in avail_feats if f in ["delivery_pct", "delivery_pct_sma5", "delivery_pct_change", "vwap_premium", "trade_intensity", "trade_intensity_sma5"]]
    pattern_feats = [f for f in avail_feats if f.startswith("pat_")]
    logger.info(f"  NSE enrichment features: {len(nse_feats)} — {nse_feats}")
    logger.info(f"  Candlestick pattern features: {len(pattern_feats)} — {pattern_feats}")

    # Non-zero check for enrichment
    for f in nse_feats:
        non_zero = (X[f] != 0).sum()
        logger.info(f"    {f}: {non_zero}/{len(X)} non-zero ({non_zero/len(X)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"\n  Model accuracy: Train={train_acc:.3f} | Test={test_acc:.3f}")

    # Feature importance
    importances = sorted(zip(avail_feats, model.feature_importances_), key=lambda x: -x[1])
    logger.info(f"\n  Top 15 feature importances:")
    for feat, imp in importances[:15]:
        bar = "█" * int(imp * 200)
        logger.info(f"    {feat:<25} {imp:.4f} {bar}")

    # Save model
    model_data = {"model": model, "features": avail_feats, "train_acc": train_acc,
                  "test_acc": test_acc, "n_samples": len(X), "n_features": len(avail_feats),
                  "enriched": len(nse_feats) > 0}
    model_path = MODEL_DIR / "stock_predictor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info(f"\n  Model saved: {model_path} ({len(avail_feats)} features, {len(X):,} samples)")
    return model, avail_feats


def main():
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ENRICH + RETRAIN — NSE Bhavcopy + yfinance")
    logger.info(f"{'=' * 60}")

    symbols = NIFTY_50

    # Step 1: Download bhavcopy enrichment
    logger.info(f"\n  Step 1: Downloading NSE bhavcopy for {len(symbols)} stocks (2 years)...")
    enrichment = download_bhavcopy_enrichment(symbols, years_back=2)
    logger.info(f"  Got enrichment for {len(enrichment)} stocks")

    # Step 2: Merge with cached data
    logger.info(f"\n  Step 2: Merging with yfinance cached data...")
    merged = merge_enrichment_with_cache(symbols, enrichment)
    logger.info(f"  Merged data: {len(merged):,} rows | {merged['symbol'].nunique()} stocks")

    if merged.empty:
        logger.error("  No data to train on!")
        return

    # Step 3: Train enhanced model
    logger.info(f"\n  Step 3: Training enhanced model with bhavcopy features...")
    model, feats = train_enhanced_model(merged)

    if model:
        logger.info(f"\n  ✅ Enhanced model ready with {len(feats)} features!")
        logger.info(f"  Including: delivery %, VWAP premium, trade intensity, 8 candlestick patterns")


if __name__ == "__main__":
    main()
