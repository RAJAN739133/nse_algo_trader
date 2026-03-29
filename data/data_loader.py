"""
Unified Data Loader — One interface, multiple data sources.

Supports: Kaggle CSVs, yfinance, custom CSVs, Kite Connect, synthetic.
Change data/data_config.yaml to switch sources. No code changes needed.

Usage:
  from data.data_loader import DataLoader
  loader = DataLoader()                          # reads data_config.yaml
  df = loader.load_training_data(["RELIANCE"])    # for ML training
  df = loader.load_live_data("SBIN")              # for paper/live trading
  df = loader.load_backtest_data("TCS", "2024-03-15")  # for past date test
"""
import os, sys, logging
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.symbols import DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent
CONFIG_PATH = DATA_DIR / "data_config.yaml"


class DataLoader:
    """Load market data from any configured source."""

    def __init__(self, config_path=None):
        path = Path(config_path) if config_path else CONFIG_PATH
        if path.exists():
            with open(path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {"training_source": "synthetic", "live_source": "synthetic",
                           "backtest_source": "synthetic", "synthetic": {"years": 10, "num_stocks": 10}}
        # Create cache folder
        Path(self.config.get("yfinance", {}).get("cache_folder", "data/cache")).mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════
    # PUBLIC METHODS — use these in your code
    # ════════════════════════════════════════════════

    def load_training_data(self, symbols=None):
        """Load historical data for ML model training."""
        source = self.config.get("training_source", "synthetic")
        symbols = symbols or DEFAULT_UNIVERSE[:self.config.get("synthetic", {}).get("num_stocks", 10)]
        logger.info(f"Loading training data from: {source} ({len(symbols)} stocks)")
        return self._load(source, symbols)

    def load_live_data(self, symbol):
        """Load latest data for live/paper trading."""
        source = self.config.get("live_source", "yfinance")
        logger.info(f"Loading live data for {symbol} from: {source}")
        df = self._load(source, [symbol])
        return df[df["symbol"] == symbol] if "symbol" in df.columns else df

    def load_backtest_data(self, symbols, target_date=None):
        """Load data for backtesting on a specific date."""
        source = self.config.get("backtest_source", "yfinance")
        symbols = symbols if isinstance(symbols, list) else [symbols]
        logger.info(f"Loading backtest data from: {source} ({len(symbols)} stocks)")
        df = self._load(source, symbols)
        if target_date and not df.empty:
            # Return data up to target_date (so model can't see future)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] <= pd.Timestamp(target_date)]
        return df

    # ════════════════════════════════════════════════
    # DATA SOURCE IMPLEMENTATIONS
    # ════════════════════════════════════════════════

    def _load(self, source, symbols):
        if source == "synthetic":
            return self._load_synthetic(symbols)
        elif source == "yfinance":
            return self._load_yfinance(symbols)
        elif source == "kaggle":
            return self._load_kaggle(symbols)
        elif source == "csv_folder":
            return self._load_csv_folder(symbols)
        elif source == "kite":
            return self._load_kite(symbols)
        else:
            logger.warning(f"Unknown source '{source}', falling back to synthetic")
            return self._load_synthetic(symbols)

    def _load_yfinance(self, symbols):
        """Download from Yahoo Finance (FREE, no API key)."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed! Run: pip install yfinance")
            logger.info("Falling back to synthetic data")
            return self._load_synthetic(symbols)

        years = self.config.get("yfinance", {}).get("history_years", 5)
        interval = self.config.get("yfinance", {}).get("interval", "1d")
        cache_dir = Path(self.config.get("yfinance", {}).get("cache_folder", "data/cache"))
        all_data = []

        for sym in symbols:
            # Check cache first
            cache_file = cache_dir / f"{sym}_{interval}_{years}y.csv"
            if cache_file.exists():
                cached_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime))
                if cached_age.days < 1:  # cache valid for 1 day
                    df = pd.read_csv(cache_file, parse_dates=["date"])
                    all_data.append(df)
                    logger.info(f"  {sym}: loaded from cache ({len(df)} rows)")
                    continue

            # Download fresh
            ticker = f"{sym}.NS"
            try:
                raw = yf.download(ticker, period=f"{years}y", interval=interval, progress=False)
                if len(raw) == 0:
                    logger.warning(f"  {sym}: no data from yfinance")
                    continue
                df = raw.reset_index()
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                if "adj close" in df.columns:
                    df = df.drop(columns=["adj close"])
                # Normalize column names
                col_map = {"date": "date", "datetime": "date", "open": "open",
                           "high": "high", "low": "low", "close": "close", "volume": "volume"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df["symbol"] = sym
                df["date"] = pd.to_datetime(df["date"])
                # Save to cache
                df.to_csv(cache_file, index=False)
                all_data.append(df)
                logger.info(f"  {sym}: downloaded {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
            except Exception as e:
                logger.error(f"  {sym}: yfinance error — {e}")

        return pd.concat(all_data, ignore_index=True) if all_data else self._load_synthetic(symbols)

    def _load_kaggle(self, symbols):
        """Load from Kaggle downloaded CSVs."""
        folder = Path(self.config.get("kaggle", {}).get("folder", "data/kaggle"))
        if not folder.exists():
            logger.error(f"Kaggle folder not found: {folder}")
            logger.info("Download from: kaggle.com/datasets/ashishjangra27/nifty-50-25-yrs-data")
            return self._load_synthetic(symbols)

        all_data = []
        for csv_file in sorted(folder.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                # Try to detect symbol from filename
                fname = csv_file.stem.upper().replace(" ", "").replace("_", "").replace("-", "")
                if "symbol" not in df.columns:
                    df["symbol"] = fname
                # Normalize date
                for dc in ["date", "datetime", "timestamp"]:
                    if dc in df.columns:
                        df["date"] = pd.to_datetime(df[dc], errors="coerce")
                        break
                if "date" not in df.columns:
                    continue
                # Filter to requested symbols (if any match)
                sym = df["symbol"].iloc[0] if "symbol" in df.columns else fname
                if symbols and sym not in symbols:
                    # Try fuzzy match
                    matched = [s for s in symbols if s in fname or fname in s]
                    if not matched:
                        continue
                    df["symbol"] = matched[0]
                required = ["date", "open", "high", "low", "close"]
                if all(c in df.columns for c in required):
                    all_data.append(df[required + ["symbol"] + (["volume"] if "volume" in df.columns else [])])
                    logger.info(f"  Loaded {csv_file.name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"  Error loading {csv_file.name}: {e}")

        return pd.concat(all_data, ignore_index=True) if all_data else self._load_synthetic(symbols)

    def _load_csv_folder(self, symbols):
        """Load from custom CSV folder — any CSV with OHLCV columns."""
        folder = Path(self.config.get("csv_folder", {}).get("path", "data/custom"))
        date_col = self.config.get("csv_folder", {}).get("date_column", "date")
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Custom CSV folder created: {folder} — put your CSVs here")
            return self._load_synthetic(symbols)
        return self._load_kaggle_style(folder, symbols, date_col)

    def _load_kaggle_style(self, folder, symbols, date_col="date"):
        """Generic CSV folder loader."""
        all_data = []
        for f in sorted(folder.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                df.columns = [c.strip().lower() for c in df.columns]
                if date_col in df.columns:
                    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
                if "symbol" not in df.columns:
                    df["symbol"] = f.stem.upper()
                if all(c in df.columns for c in ["date", "open", "high", "low", "close"]):
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error: {f.name}: {e}")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _load_kite(self, symbols):
        """Load from Kite Connect (needs API key + subscription)."""
        try:
            from data.kite_auth import KiteAuth
            auth = KiteAuth()
            kite = auth.interactive_login()
            from data.downloader import KiteDataDownloader
            dl = KiteDataDownloader(kite)
            days = self.config.get("kite", {}).get("intraday_days", 60)
            interval = self.config.get("kite", {}).get("interval", "day")
            all_data = []
            for sym in symbols:
                df = dl.get_historical(sym, days=days, interval=interval)
                if not df.empty:
                    df["symbol"] = sym
                    all_data.append(df)
            return pd.concat(all_data, ignore_index=True) if all_data else self._load_synthetic(symbols)
        except Exception as e:
            logger.error(f"Kite Connect error: {e}")
            return self._load_yfinance(symbols)

    def _load_synthetic(self, symbols):
        """Generate synthetic data (works offline, always available)."""
        from data.train_pipeline import generate_synthetic
        years = self.config.get("synthetic", {}).get("years", 10)
        return generate_synthetic(symbols, years=years)

    # ════════════════════════════════════════════════
    # UTILITY — info about available data
    # ════════════════════════════════════════════════

    def status(self):
        """Print current data configuration and what's available."""
        print(f"\n  DATA CONFIGURATION")
        print(f"  ==================")
        print(f"  Training source:  {self.config.get('training_source')}")
        print(f"  Live source:      {self.config.get('live_source')}")
        print(f"  Backtest source:  {self.config.get('backtest_source')}")

        # Check what's available
        kaggle_dir = Path(self.config.get("kaggle", {}).get("folder", "data/kaggle"))
        cache_dir = Path(self.config.get("yfinance", {}).get("cache_folder", "data/cache"))
        custom_dir = Path(self.config.get("csv_folder", {}).get("path", "data/custom"))

        if kaggle_dir.exists():
            csvs = list(kaggle_dir.glob("*.csv"))
            print(f"\n  Kaggle: {len(csvs)} CSVs in {kaggle_dir}")
        else:
            print(f"\n  Kaggle: not downloaded yet")

        if cache_dir.exists():
            cached = list(cache_dir.glob("*.csv"))
            print(f"  yfinance cache: {len(cached)} stocks cached")
        else:
            print(f"  yfinance cache: empty")

        if custom_dir.exists():
            customs = list(custom_dir.glob("*.csv"))
            print(f"  Custom CSVs: {len(customs)} files in {custom_dir}")

        try:
            import yfinance
            print(f"  yfinance: installed (v{yfinance.__version__})")
        except ImportError:
            print(f"  yfinance: NOT installed (run: pip install yfinance)")

        try:
            import kiteconnect
            print(f"  kiteconnect: installed")
        except ImportError:
            print(f"  kiteconnect: NOT installed (optional, Rs 2000/month)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    loader = DataLoader()
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        loader.status()
    else:
        print("Usage: python -m data.data_loader status")
        loader.status()
