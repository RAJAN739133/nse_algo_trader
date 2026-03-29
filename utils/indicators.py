"""
Technical indicators for NSE intraday trading.
All functions take a pandas DataFrame with OHLCV columns and return
the DataFrame with new indicator columns added.
"""

import pandas as pd
import numpy as np


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — measures volatility. Used for dynamic stop losses."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=period).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index — momentum oscillator (0-100)."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume Weighted Average Price — the institutional benchmark.
    Resets every trading day. Requires 'datetime' column or DatetimeIndex.
    """
    if "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"]).dt.date
    else:
        dates = df.index.date

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    # Cumulative sums that reset each day
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()

    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # VWAP standard deviation bands
    squared_diff = ((typical_price - df["vwap"]) ** 2) * df["volume"]
    cum_sq_diff = squared_diff.groupby(dates).cumsum()
    vwap_var = cum_sq_diff / cum_vol.replace(0, np.nan)
    df["vwap_std"] = np.sqrt(vwap_var)
    df["vwap_upper_1"] = df["vwap"] + df["vwap_std"]
    df["vwap_lower_1"] = df["vwap"] - df["vwap_std"]
    df["vwap_upper_2"] = df["vwap"] + 2 * df["vwap_std"]
    df["vwap_lower_2"] = df["vwap"] - 2 * df["vwap_std"]

    return df


def add_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """Exponential Moving Average."""
    col_name = f"ema_{period}"
    df[col_name] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Supertrend indicator — used for trailing stops.
    Returns 'supertrend' (the stop level) and 'supertrend_dir' (+1 or -1).
    """
    df = add_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2

    upper_band = hl2 + multiplier * df["atr"]
    lower_band = hl2 - multiplier * df["atr"]

    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(period, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = max(
                lower_band.iloc[i],
                supertrend.iloc[i - 1] if direction.iloc[i - 1] == 1 else lower_band.iloc[i],
            )
        else:
            supertrend.iloc[i] = min(
                upper_band.iloc[i],
                supertrend.iloc[i - 1] if direction.iloc[i - 1] == -1 else upper_band.iloc[i],
            )

    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands — volatility-based envelope around moving average."""
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()

    df["bb_mid"] = sma
    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    return df


def add_volume_profile(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Rolling average volume and volume ratio (current / average)."""
    df["vol_avg"] = df["volume"].rolling(window=lookback).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, np.nan)
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators needed for ORB + VWAP strategies."""
    df = add_atr(df, period=14)
    df = add_rsi(df, period=14)
    df = add_vwap(df)
    df = add_ema(df, period=20)
    df = add_supertrend(df, period=10, multiplier=2.0)
    df = add_bollinger_bands(df, period=20)
    df = add_volume_profile(df, lookback=20)
    return df
