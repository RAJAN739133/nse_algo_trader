"""
Data Enricher — Bulk historical data download for ML training.
═══════════════════════════════════════════════════════════════

Downloads 10+ years of OHLCV data from multiple free sources:
  1. yfinance — up to 25 years for NSE stocks
  2. Angel One historical API — if configured
  3. Kaggle datasets — manual download guide + auto-load

Also downloads auxiliary data:
  - India VIX history (for volatility regime features)
  - Nifty 50 index (for market breadth features)
  - Sector indices (for relative strength)

Usage:
    python -m data.data_enricher download-all       # Download everything
    python -m data.data_enricher download-extended   # 10yr yfinance for Nifty 250
    python -m data.data_enricher download-indices    # VIX + Nifty + Sector indices
    python -m data.data_enricher download-kaggle     # Instructions for Kaggle
    python -m data.data_enricher status              # Check what's available
"""
import os
import sys
import logging
import time
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.symbols import NIFTY_50, NIFTY_100_EXTRA, NIFTY_250_EXTRA, ALL_STOCKS

logger = logging.getLogger(__name__)
CACHE_DIR = Path("data/cache")
ENRICHED_DIR = Path("data/enriched")
KAGGLE_DIR = Path("data/kaggle")

# Sector indices for relative strength
SECTOR_INDICES = {
    "NIFTY_50": "^NSEI",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_ENERGY": "^CNXENERGY",
    "INDIA_VIX": "^INDIAVIX",
}


def download_extended(symbols=None, years=10, batch_size=10, delay=2):
    """
    Download extended history from yfinance.
    10 years of daily OHLCV for all Nifty 250 stocks.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return

    symbols = symbols or ALL_STOCKS
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    total = len(symbols)
    downloaded = 0
    failed = []

    logger.info(f"\n  Downloading {years}yr history for {total} stocks...")
    logger.info(f"  Output: {ENRICHED_DIR}/")

    for i, sym in enumerate(symbols):
        out_file = ENRICHED_DIR / f"{sym}_{years}y.csv"

        # Skip if already downloaded today
        if out_file.exists():
            cached_age = datetime.now() - datetime.fromtimestamp(out_file.stat().st_mtime)
            if cached_age.days < 1:
                downloaded += 1
                continue

        try:
            ticker = f"{sym}.NS"
            df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
            if df.empty:
                failed.append(sym)
                continue

            df = df.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            if "adj close" in df.columns:
                df.drop(columns=["adj close"], inplace=True)

            col_map = {"date": "date", "datetime": "date"}
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            df["symbol"] = sym
            df["date"] = pd.to_datetime(df["date"])
            df.to_csv(out_file, index=False)
            downloaded += 1

            logger.info(f"  [{i+1}/{total}] {sym}: {len(df)} rows "
                        f"({df['date'].min().date()} → {df['date'].max().date()})")

        except Exception as e:
            failed.append(sym)
            logger.warning(f"  [{i+1}/{total}] {sym}: FAILED — {e}")

        # Rate limit: pause every batch_size
        if (i + 1) % batch_size == 0:
            time.sleep(delay)

    logger.info(f"\n  Done! Downloaded: {downloaded}/{total} | Failed: {len(failed)}")
    if failed:
        logger.info(f"  Failed symbols: {', '.join(failed[:20])}")


def download_indices():
    """Download India VIX, Nifty 50, and sector indices history."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n  Downloading index data...")

    for name, ticker in SECTOR_INDICES.items():
        out_file = ENRICHED_DIR / f"INDEX_{name}.csv"
        try:
            df = yf.download(ticker, period="10y", interval="1d", progress=False)
            if df.empty:
                logger.warning(f"  {name}: no data")
                continue
            df = df.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df["symbol"] = name
            df.to_csv(out_file, index=False)
            logger.info(f"  {name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  {name}: {e}")


def load_enriched_data(symbols=None, include_indices=True):
    """
    Load all enriched data into a single DataFrame.
    Used by training pipeline for a richer ML model.
    """
    symbols = symbols or ALL_STOCKS
    all_data = []

    # Load stock data
    for sym in symbols:
        # Try enriched first (10yr), then fall back to cache (5yr)
        for pattern in [
            ENRICHED_DIR / f"{sym}_10y.csv",
            ENRICHED_DIR / f"{sym}_*y.csv",
            CACHE_DIR / f"{sym}_1d_5y.csv",
        ]:
            files = list(pattern.parent.glob(pattern.name)) if "*" in str(pattern) else ([pattern] if pattern.exists() else [])
            if files:
                df = pd.read_csv(files[0], parse_dates=["date"])
                df["symbol"] = sym
                all_data.append(df)
                break

    # Load Kaggle data if available
    if KAGGLE_DIR.exists():
        for csv_file in KAGGLE_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                for dc in ["date", "datetime", "timestamp"]:
                    if dc in df.columns:
                        df["date"] = pd.to_datetime(df[dc], errors="coerce")
                        break
                if "date" not in df.columns:
                    continue
                if "symbol" not in df.columns:
                    df["symbol"] = csv_file.stem.upper().replace(" ", "").replace("_", "")
                sym = df["symbol"].iloc[0]
                if sym in symbols:
                    all_data.append(df)
                    logger.info(f"  Loaded Kaggle: {csv_file.name} ({len(df)} rows)")
            except Exception:
                pass

    # Load index data for features
    if include_indices:
        for name in SECTOR_INDICES:
            idx_file = ENRICHED_DIR / f"INDEX_{name}.csv"
            if idx_file.exists():
                df = pd.read_csv(idx_file, parse_dates=["date"])
                df["symbol"] = name
                all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Deduplicate (prefer longer history)
    combined = combined.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    return combined


def add_market_context_features(df):
    """
    Add cross-stock features using index data:
      - market_return: Nifty 50 return (market trend)
      - vix_level: India VIX (volatility regime)
      - sector_rs: stock return vs sector return (relative strength)
    """
    # Load Nifty 50 index
    nifty_file = ENRICHED_DIR / "INDEX_NIFTY_50.csv"
    vix_file = ENRICHED_DIR / "INDEX_INDIA_VIX.csv"

    if nifty_file.exists():
        nifty = pd.read_csv(nifty_file, parse_dates=["date"])
        nifty = nifty[["date", "close"]].rename(columns={"close": "nifty_close"})
        nifty["market_ret_1d"] = nifty["nifty_close"].pct_change()
        nifty["market_ret_5d"] = nifty["nifty_close"].pct_change(5)
        nifty["market_sma20"] = nifty["nifty_close"].rolling(20).mean()
        nifty["market_above_sma20"] = (nifty["nifty_close"] > nifty["market_sma20"]).astype(int)
        df = df.merge(nifty[["date", "market_ret_1d", "market_ret_5d", "market_above_sma20"]],
                       on="date", how="left")

    if vix_file.exists():
        vix = pd.read_csv(vix_file, parse_dates=["date"])
        vix = vix[["date", "close"]].rename(columns={"close": "vix"})
        vix["vix_sma10"] = vix["vix"].rolling(10).mean()
        vix["vix_regime"] = pd.cut(vix["vix"], bins=[0, 15, 20, 25, 100],
                                    labels=[0, 1, 2, 3]).astype(float)
        df = df.merge(vix[["date", "vix", "vix_regime"]], on="date", how="left")

    return df


def status():
    """Print what data is available."""
    print(f"\n  DATA ENRICHMENT STATUS")
    print(f"  ═════════════════════")

    for name, folder in [("Enriched (10yr)", ENRICHED_DIR),
                          ("Cache (5yr)", CACHE_DIR),
                          ("Kaggle", KAGGLE_DIR)]:
        if folder.exists():
            csvs = list(folder.glob("*.csv"))
            total_rows = 0
            for f in csvs[:5]:
                try:
                    total_rows += sum(1 for _ in open(f)) - 1
                except:
                    pass
            print(f"  {name}: {len(csvs)} files in {folder}/")
        else:
            print(f"  {name}: not found ({folder}/)")

    # Check indices
    for name in SECTOR_INDICES:
        f = ENRICHED_DIR / f"INDEX_{name}.csv"
        if f.exists():
            rows = sum(1 for _ in open(f)) - 1
            print(f"    ✓ {name}: {rows} rows")


def print_kaggle_guide():
    """Print instructions for downloading Kaggle datasets."""
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  KAGGLE DATA DOWNLOAD GUIDE                              ║
  ╠══════════════════════════════════════════════════════════╣
  ║                                                          ║
  ║  Option 1: Nifty 50 — 25 Years (Recommended)            ║
  ║  ─────────────────────────────────────────────           ║
  ║  URL: kaggle.com/datasets/ashishjangra27/                ║
  ║       nifty-50-25-yrs-data                               ║
  ║  Download → Extract CSVs → Put in data/kaggle/           ║
  ║                                                          ║
  ║  Option 2: NSE All Stocks                                ║
  ║  ─────────────────────────────────────────────           ║
  ║  URL: kaggle.com/datasets/iamsouravbanerjee/             ║
  ║       nifty50-stocks-dataset                             ║
  ║                                                          ║
  ║  Option 3: BSE + NSE Historical                          ║
  ║  ─────────────────────────────────────────────           ║
  ║  URL: kaggle.com/datasets/debashis74017/                 ║
  ║       stock-market-data-nifty-50-stocks                  ║
  ║                                                          ║
  ║  After downloading, run:                                 ║
  ║    python -m data.data_enricher download-all             ║
  ║                                                          ║
  ║  Or via Kaggle CLI:                                      ║
  ║    pip install kaggle                                    ║
  ║    kaggle datasets download -d ashishjangra27/           ║
  ║      nifty-50-25-yrs-data -p data/kaggle/ --unzip       ║
  ╚══════════════════════════════════════════════════════════╝
    """)


def download_all():
    """Download everything — stocks + indices."""
    logger.info("═══ Phase 1: Extended stock history (10yr) ═══")
    download_extended(ALL_STOCKS, years=10, batch_size=10, delay=2)

    logger.info("\n═══ Phase 2: Index data (VIX, Nifty, Sectors) ═══")
    download_indices()

    logger.info("\n═══ Phase 3: Check Kaggle data ═══")
    if KAGGLE_DIR.exists() and list(KAGGLE_DIR.glob("*.csv")):
        logger.info(f"  Kaggle data found: {len(list(KAGGLE_DIR.glob('*.csv')))} files")
    else:
        logger.info("  No Kaggle data found. Optional: download for 25yr history.")
        print_kaggle_guide()

    status()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "download-all":
        download_all()
    elif cmd == "download-extended":
        download_extended()
    elif cmd == "download-indices":
        download_indices()
    elif cmd == "download-kaggle":
        print_kaggle_guide()
    elif cmd == "status":
        status()
    else:
        print("Usage: python -m data.data_enricher [download-all|download-extended|download-indices|download-kaggle|status]")
