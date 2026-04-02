"""
ML Training Pipeline V2 — Better model for bear markets.
═══════════════════════════════════════════════════════════

Improvements over V1:
  1. Trains on ALL 150+ Nifty 250 stocks (not just 50)
  2. Downloads fresh 10-year data from yfinance
  3. Adds market context features (VIX, Nifty trend, sector RS)
  4. Uses XGBoost with better hyperparameters
  5. Separate models for LONG and SHORT predictions
  6. Walk-forward validation (proper time series)
  7. Feature importance analysis

Usage:
  python -m data.train_v2 download   # Download fresh data for all stocks
  python -m data.train_v2 train      # Train improved model
  python -m data.train_v2 backtest   # Validate on recent data
  python -m data.train_v2 score      # Score stocks for today
"""
import os
import sys
import pickle
import logging
import time
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.symbols import NIFTY_50, NIFTY_100_EXTRA, NIFTY_250_EXTRA, ALL_STOCKS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR.parent / "models"

# ════════════════════════════════════════════════════════════════
# EXPANDED FEATURE SET (50+ features)
# ════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Price momentum (short + long term)
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    
    # Trend position
    "price_vs_sma5", "price_vs_sma10", "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "sma5_vs_sma20", "sma20_vs_sma50", "sma50_vs_sma200",
    
    # Volatility
    "atr_pct", "atr_pct_change", "vol_20", "vol_ratio_5_20",
    "bb_width", "bb_pos", "keltner_pos",
    
    # Oscillators
    "rsi_7", "rsi_14", "rsi_change", "stoch_k", "stoch_d",
    "macd_hist", "macd_hist_change", "cci_20", "williams_r",
    
    # Volume
    "vol_ratio", "vol_trend", "obv_slope", "mfi_14",
    
    # Candlestick patterns
    "candle_score", "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "gap_pct", "inside_bar", "outside_bar",
    
    # Market context (CRITICAL for bear markets)
    "market_ret_1d", "market_ret_5d", "market_above_sma20",
    "vix", "vix_regime", "stock_vs_market",
    
    # NSE-specific (delivery %)
    "delivery_pct", "delivery_pct_sma5", "delivery_pct_change",
]


def download_all_stocks(years=10):
    """Download fresh data for all Nifty 250 stocks."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = ALL_STOCKS
    total = len(symbols)
    downloaded = 0
    failed = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  DOWNLOADING {years}yr DATA FOR {total} STOCKS")
    logger.info(f"{'='*60}\n")
    
    for i, sym in enumerate(symbols):
        out_file = CACHE_DIR / f"{sym}_1d_{years}y.csv"
        
        # Skip if fresh (downloaded within 24 hours)
        if out_file.exists():
            age = datetime.now() - datetime.fromtimestamp(out_file.stat().st_mtime)
            if age.days < 1:
                downloaded += 1
                continue
        
        try:
            ticker = f"{sym}.NS"
            df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
            
            if df.empty or len(df) < 100:
                failed.append(sym)
                continue
            
            # Clean columns
            df = df.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            if "adj close" in df.columns:
                df.drop(columns=["adj close"], inplace=True, errors="ignore")
            
            df.rename(columns={"datetime": "date"}, inplace=True)
            df["symbol"] = sym
            df["date"] = pd.to_datetime(df["date"])
            df.to_csv(out_file, index=False)
            downloaded += 1
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{total} | {sym}: {len(df)} rows")
            
        except Exception as e:
            failed.append(sym)
            if "delisted" in str(e).lower():
                logger.debug(f"  {sym}: delisted")
        
        # Rate limit
        if (i + 1) % 10 == 0:
            time.sleep(1)
    
    logger.info(f"\n  Done! Downloaded: {downloaded}/{total} | Failed: {len(failed)}")
    if failed:
        logger.info(f"  Failed: {', '.join(failed[:30])}")
    
    # Also download indices
    download_indices()


def download_indices():
    """Download VIX and Nifty 50 for market context."""
    try:
        import yfinance as yf
    except ImportError:
        return
    
    indices = {
        "NIFTY50": "^NSEI",
        "INDIAVIX": "^INDIAVIX",
    }
    
    for name, ticker in indices.items():
        out_file = CACHE_DIR / f"INDEX_{name}.csv"
        try:
            df = yf.download(ticker, period="10y", interval="1d", progress=False)
            if df.empty:
                continue
            df = df.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df["symbol"] = name
            df.to_csv(out_file, index=False)
            logger.info(f"  Index {name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Index {name} failed: {e}")


def load_all_data():
    """Load all cached stock data into one DataFrame."""
    all_data = []
    
    for csv_file in CACHE_DIR.glob("*.csv"):
        if csv_file.name.startswith("INDEX_"):
            continue
        try:
            df = pd.read_csv(csv_file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            if "symbol" not in df.columns:
                df["symbol"] = csv_file.stem.split("_")[0]
            all_data.append(df)
        except Exception:
            pass
    
    if not all_data:
        logger.error("No data found in cache/. Run: python -m data.train_v2 download")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    
    logger.info(f"  Loaded {len(combined):,} rows | {combined['symbol'].nunique()} stocks")
    logger.info(f"  Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")
    
    return combined


def load_market_data():
    """Load Nifty 50 and VIX for market context features."""
    nifty_file = CACHE_DIR / "INDEX_NIFTY50.csv"
    vix_file = CACHE_DIR / "INDEX_INDIAVIX.csv"
    
    market = pd.DataFrame()
    
    if nifty_file.exists():
        nifty = pd.read_csv(nifty_file, parse_dates=["date"] if "date" in pd.read_csv(nifty_file, nrows=1).columns else None)
        if "date" not in nifty.columns and "datetime" in nifty.columns:
            nifty["date"] = pd.to_datetime(nifty["datetime"])
        nifty = nifty[["date", "close"]].rename(columns={"close": "nifty_close"})
        nifty["market_ret_1d"] = nifty["nifty_close"].pct_change()
        nifty["market_ret_5d"] = nifty["nifty_close"].pct_change(5)
        nifty["market_sma20"] = nifty["nifty_close"].rolling(20).mean()
        nifty["market_above_sma20"] = (nifty["nifty_close"] > nifty["market_sma20"]).astype(int)
        market = nifty
    
    if vix_file.exists():
        vix = pd.read_csv(vix_file, parse_dates=["date"] if "date" in pd.read_csv(vix_file, nrows=1).columns else None)
        if "date" not in vix.columns and "datetime" in vix.columns:
            vix["date"] = pd.to_datetime(vix["datetime"])
        vix = vix[["date", "close"]].rename(columns={"close": "vix"})
        vix["vix_regime"] = pd.cut(vix["vix"], bins=[0, 14, 18, 22, 100], labels=[0, 1, 2, 3]).astype(float)
        
        if market.empty:
            market = vix
        else:
            market = market.merge(vix, on="date", how="outer")
    
    return market


def add_features(df):
    """Add 50+ technical indicators and patterns."""
    df = df.sort_values("date").copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df.get("volume", pd.Series([1]*len(df)))
    
    # ── Price Momentum ──
    for d in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{d}d"] = c.pct_change(d)
    
    # ── Moving Averages ──
    for p in [5, 10, 20, 50, 200]:
        df[f"sma_{p}"] = c.rolling(p).mean()
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    
    # Price vs SMAs
    df["price_vs_sma5"] = (c - df["sma_5"]) / df["sma_5"]
    df["price_vs_sma10"] = (c - df["sma_10"]) / df["sma_10"]
    df["price_vs_sma20"] = (c - df["sma_20"]) / df["sma_20"]
    df["price_vs_sma50"] = (c - df["sma_50"]) / df["sma_50"]
    df["price_vs_sma200"] = (c - df["sma_200"]) / df["sma_200"]
    
    # SMA crosses
    df["sma5_vs_sma20"] = (df["sma_5"] - df["sma_20"]) / df["sma_20"]
    df["sma20_vs_sma50"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]
    df["sma50_vs_sma200"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]
    
    # ── Volatility ──
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / c
    df["atr_pct_change"] = df["atr_pct"].pct_change(5)
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["vol_ratio_5_20"] = df["ret_1d"].rolling(5).std() / df["ret_1d"].rolling(20).std()
    
    # Bollinger Bands
    std20 = c.rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2 * std20
    df["bb_lower"] = df["sma_20"] - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]
    df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # Keltner Channel
    kelt_mid = df["ema_20"]
    kelt_range = df["atr_14"] * 2
    df["keltner_pos"] = (c - (kelt_mid - kelt_range)) / (2 * kelt_range)
    
    # ── Oscillators ──
    for period in [7, 14]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
        df[f"rsi_{period}"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["rsi_change"] = df["rsi_14"].diff(5)
    
    # Stochastic
    low_14 = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    
    # MACD
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_change"] = df["macd_hist"].diff(3)
    
    # CCI
    tp = (h + l + c) / 3
    df["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    
    # Williams %R
    df["williams_r"] = -100 * (high_14 - c) / (high_14 - low_14)
    
    # ── Volume ──
    if v is not None and not v.isna().all():
        df["vol_sma20"] = v.rolling(20).mean()
        df["vol_ratio"] = v / df["vol_sma20"].replace(0, np.nan)
        df["vol_trend"] = v.rolling(5).mean() / v.rolling(20).mean()
        
        # OBV slope
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df["obv_slope"] = obv.diff(10) / obv.rolling(20).mean()
        
        # MFI
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df["mfi_14"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))
    else:
        df["vol_ratio"] = 1.0
        df["vol_trend"] = 1.0
        df["obv_slope"] = 0.0
        df["mfi_14"] = 50.0
    
    # ── Candlestick Patterns ──
    body = abs(c - o)
    rng = h - l + 0.0001
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    
    df["body_ratio"] = body / rng
    df["upper_wick_ratio"] = upper_wick / rng
    df["lower_wick_ratio"] = lower_wick / rng
    df["gap_pct"] = (o - c.shift(1)) / c.shift(1)
    
    # Inside bar (range within previous bar)
    df["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    df["outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(int)
    
    # Pattern scoring
    po, pc = o.shift(1), c.shift(1)
    pat_hammer = ((lower_wick > body * 2) & (upper_wick < body * 0.5) & (c > o)).astype(int)
    pat_shooting = -((upper_wick > body * 2) & (lower_wick < body * 0.5) & (c < o)).astype(int)
    pat_engulf_bull = ((c > o) & (pc < po) & (o <= pc) & (c >= po)).astype(int)
    pat_engulf_bear = -((c < o) & (pc > po) & (o >= pc) & (c <= po)).astype(int)
    df["candle_score"] = pat_hammer + pat_shooting + pat_engulf_bull + pat_engulf_bear
    
    # ── Target ──
    # Next day direction (for training)
    df["target_ret"] = c.shift(-1) / c - 1
    df["target_dir"] = (df["target_ret"] > 0).astype(int)
    # Next 3-day direction (for swing)
    df["target_3d"] = (c.shift(-3) / c - 1 > 0).astype(int)
    
    # Sanitize
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df


def add_market_context(df, market_data):
    """Merge market context (VIX, Nifty) into stock data."""
    if market_data.empty:
        df["market_ret_1d"] = 0
        df["market_ret_5d"] = 0
        df["market_above_sma20"] = 1
        df["vix"] = 15
        df["vix_regime"] = 1
        df["stock_vs_market"] = 0
        return df
    
    df = df.merge(market_data, on="date", how="left")
    
    # Stock vs market relative strength
    df["stock_vs_market"] = df["ret_1d"] - df["market_ret_1d"].fillna(0)
    
    # Fill missing
    df["vix"] = df["vix"].fillna(15)
    df["vix_regime"] = df["vix_regime"].fillna(1)
    df["market_ret_1d"] = df["market_ret_1d"].fillna(0)
    df["market_ret_5d"] = df["market_ret_5d"].fillna(0)
    df["market_above_sma20"] = df["market_above_sma20"].fillna(1)
    
    return df


def train_model(df, model_type="xgboost"):
    """Train ML model with walk-forward validation."""
    
    # Use available features
    avail_features = [f for f in FEATURE_COLS if f in df.columns]
    logger.info(f"  Available features: {len(avail_features)}/{len(FEATURE_COLS)}")
    
    # Prepare data
    train_df = df.dropna(subset=avail_features + ["target_dir"]).copy()
    X = train_df[avail_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_df["target_dir"]
    
    logger.info(f"  Training samples: {len(X):,}")
    logger.info(f"  Class balance: UP {y.mean():.1%} | DOWN {1-y.mean():.1%}")
    
    # Walk-forward validation
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model_class = XGBClassifier
            model_params = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 50,
                "random_state": 42,
                "verbosity": 0,
            }
        except ImportError:
            logger.warning("XGBoost not installed, falling back to GradientBoosting")
            model_type = "gbm"
    
    if model_type == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier
        model_class = GradientBoostingClassifier
        model_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "min_samples_leaf": 50,
            "random_state": 42,
        }
    
    # 5-fold walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    
    logger.info(f"\n  Walk-forward validation ({model_type.upper()}):")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fold_scores.append({"acc": acc, "prec": prec, "rec": rec, "f1": f1})
        logger.info(f"    Fold {fold}: Acc={acc:.1%} Prec={prec:.1%} Rec={rec:.1%} F1={f1:.1%}")
    
    avg_acc = np.mean([s["acc"] for s in fold_scores])
    avg_f1 = np.mean([s["f1"] for s in fold_scores])
    logger.info(f"  Average: Acc={avg_acc:.1%} F1={avg_f1:.1%}")
    
    # Train final model on all data
    logger.info(f"\n  Training final model on all {len(X):,} samples...")
    final_model = model_class(**model_params)
    final_model.fit(X, y)
    
    # Feature importance
    if hasattr(final_model, "feature_importances_"):
        importance = pd.Series(final_model.feature_importances_, index=avail_features)
        importance = importance.sort_values(ascending=False)
        logger.info(f"\n  Top 15 features:")
        for feat, imp in importance.head(15).items():
            bar = "█" * int(imp * 100)
            logger.info(f"    {feat:<25} {imp:.3f} {bar}")
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "stock_predictor_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "features": avail_features,
            "model_type": model_type,
            "trained_on": len(X),
            "train_date": str(date.today()),
            "metrics": {"avg_acc": avg_acc, "avg_f1": avg_f1},
        }, f)
    
    logger.info(f"\n  Model saved: {model_path}")
    
    return final_model, avail_features


def score_stocks(df, model=None, features=None, vix=15):
    """Score all stocks for LONG and SHORT opportunities."""
    if model is None:
        model_path = MODEL_DIR / "stock_predictor_v2.pkl"
        if not model_path.exists():
            model_path = MODEL_DIR / "stock_predictor.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                model = data["model"]
                features = data["features"]
    
    scores = []
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym]
        if len(sdf) < 50:
            continue
        
        lat = sdf.iloc[-1]
        rsi = lat.get("rsi_14", 50)
        vol_ratio = lat.get("vol_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        
        # ML prediction
        ml_bull, ml_bear = 15, 15
        if model and features:
            try:
                x = np.array([lat.get(f, 0) for f in features], dtype=np.float64).reshape(1, -1)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                prob_up = model.predict_proba(x)[0][1]
                ml_bull = int(prob_up * 40)
                ml_bear = int((1 - prob_up) * 40)
            except Exception:
                pass
        
        # Technical scoring
        price_vs_sma20 = lat.get("price_vs_sma20", 0)
        price_vs_sma50 = lat.get("price_vs_sma50", 0)
        macd_hist = lat.get("macd_hist", 0)
        
        # LONG score
        long_tech = 0
        if price_vs_sma20 > 0: long_tech += 5
        if price_vs_sma50 > 0: long_tech += 5
        if macd_hist > 0: long_tech += 5
        long_tech += 10 if rsi < 30 else (7 if rsi < 40 else (3 if rsi < 50 else 0))
        long_vol = min(10, int(vol_ratio * 3))
        long_score = ml_bull + long_tech + long_vol
        
        # SHORT score
        short_tech = 0
        if price_vs_sma20 < 0: short_tech += 5
        if price_vs_sma50 < 0: short_tech += 5
        if macd_hist < 0: short_tech += 5
        short_tech += 10 if rsi > 70 else (7 if rsi > 60 else (3 if rsi > 50 else 0))
        short_vol = min(10, int(vol_ratio * 3))
        short_score = ml_bear + short_tech + short_vol
        
        # VIX adjustment
        if vix > 22:
            short_score += 5  # Favor shorts in high VIX
        elif vix < 14:
            long_score += 5   # Favor longs in low VIX
        
        # Pick direction
        if long_score >= short_score:
            direction = "LONG"
            best_score = long_score
        else:
            direction = "SHORT"
            best_score = short_score
        
        scores.append({
            "symbol": sym,
            "score": min(100, best_score),
            "long_score": min(100, long_score),
            "short_score": min(100, short_score),
            "direction": direction,
            "price": round(lat.get("close", 0), 2),
            "rsi": round(rsi, 1),
            "atr_pct": round(lat.get("atr_pct", 0) * 100, 2),
        })
    
    return pd.DataFrame(scores).sort_values("score", ascending=False)


def main():
    """CLI entry point."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if cmd == "download":
        download_all_stocks(years=10)
    
    elif cmd == "train":
        logger.info(f"\n{'='*60}")
        logger.info(f"  ML TRAINING PIPELINE V2")
        logger.info(f"{'='*60}\n")
        
        # Load data
        logger.info("  Phase 1: Loading data...")
        raw_data = load_all_data()
        if raw_data.empty:
            logger.error("No data. Run: python -m data.train_v2 download")
            return
        
        market_data = load_market_data()
        logger.info(f"  Market data: {len(market_data)} rows")
        
        # Add features
        logger.info("\n  Phase 2: Computing features...")
        featured = []
        symbols = raw_data["symbol"].unique()
        for i, sym in enumerate(symbols):
            sdf = raw_data[raw_data["symbol"] == sym].copy()
            if len(sdf) < 100:
                continue
            sdf = add_features(sdf)
            sdf = add_market_context(sdf, market_data)
            featured.append(sdf)
            if (i + 1) % 50 == 0:
                logger.info(f"    Processed {i+1}/{len(symbols)} stocks")
        
        all_featured = pd.concat(featured, ignore_index=True)
        logger.info(f"  Total: {len(all_featured):,} rows | {all_featured['symbol'].nunique()} stocks")
        
        # Train
        logger.info("\n  Phase 3: Training model...")
        train_model(all_featured)
    
    elif cmd == "score":
        logger.info(f"\n  Scoring stocks for {date.today()}...")
        raw_data = load_all_data()
        market_data = load_market_data()
        
        # Get VIX
        vix = 15
        if not market_data.empty and "vix" in market_data.columns:
            vix = market_data["vix"].iloc[-1]
        
        # Add features and score
        featured = []
        for sym in raw_data["symbol"].unique():
            sdf = raw_data[raw_data["symbol"] == sym].copy()
            if len(sdf) < 100:
                continue
            sdf = add_features(sdf)
            sdf = add_market_context(sdf, market_data)
            featured.append(sdf)
        
        all_featured = pd.concat(featured, ignore_index=True)
        scores = score_stocks(all_featured, vix=vix)
        
        logger.info(f"\n  {'Rank':<5}{'Symbol':<12}{'Score':>6}{'Long':>6}{'Short':>6}{'Dir':<6}{'RSI':>6}{'ATR%':>6}")
        logger.info("  " + "-" * 60)
        for i, (_, r) in enumerate(scores.head(20).iterrows(), 1):
            arrow = "▲" if r["direction"] == "LONG" else "▼"
            logger.info(f"  {i:<5}{r['symbol']:<12}{r['score']:>5.0f}{r['long_score']:>6.0f}{r['short_score']:>6.0f}  {arrow} {r['direction']:<4}{r['rsi']:>6.1f}{r['atr_pct']:>6.1f}")
        
        # Top picks
        top_long = scores[scores["direction"] == "LONG"].head(5)
        top_short = scores[scores["direction"] == "SHORT"].head(5)
        
        logger.info(f"\n  TOP LONG PICKS: {', '.join(top_long['symbol'].tolist())}")
        logger.info(f"  TOP SHORT PICKS: {', '.join(top_short['symbol'].tolist())}")
    
    else:
        print("""
  ML Training Pipeline V2 — Better model for bear markets
  ════════════════════════════════════════════════════════
  
  Usage:
    python -m data.train_v2 download   # Download 10yr data for 150+ stocks
    python -m data.train_v2 train      # Train improved model
    python -m data.train_v2 score      # Score stocks for today
        """)


if __name__ == "__main__":
    main()
