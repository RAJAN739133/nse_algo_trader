"""
Curated NSE stock universe for intraday algo trading.
Selected for: high liquidity, tight spreads, F&O availability, and
suitability for ORB + VWAP strategies with ₹1L capital.
"""

# Tier 1: Best for intraday — highest volume, tightest spreads
# These should be your primary focus with ₹1L capital
TIER1_STOCKS = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "TCS",
    "INFY",
    "SBIN",
    "BHARTIARTL",
    "ITC",
    "KOTAKBANK",
    "LT",
]

# Tier 2: Good liquidity, slightly wider spreads
TIER2_STOCKS = [
    "AXISBANK",
    "TATAMOTORS",
    "MARUTI",
    "SUNPHARMA",
    "HCLTECH",
    "WIPRO",
    "BAJFINANCE",
    "TATASTEEL",
    "NTPC",
    "POWERGRID",
]

# Index instruments (for hedging and benchmarking)
INDEX_INSTRUMENTS = [
    "NIFTY 50",
    "NIFTY BANK",
]

# Full universe = Tier 1 + Tier 2
ALL_STOCKS = TIER1_STOCKS + TIER2_STOCKS

# Default: start with Tier 1 only (simpler, more liquid)
DEFAULT_UNIVERSE = TIER1_STOCKS

# NSE exchange string for Kite Connect
EXCHANGE = "NSE"

# Kite instrument tokens are fetched dynamically at runtime
# This mapping is just for human reference
STOCK_SECTORS = {
    "RELIANCE": "Energy",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "TCS": "IT",
    "INFY": "IT",
    "SBIN": "Banking",
    "BHARTIARTL": "Telecom",
    "ITC": "FMCG",
    "KOTAKBANK": "Banking",
    "LT": "Infrastructure",
    "AXISBANK": "Banking",
    "TATAMOTORS": "Auto",
    "MARUTI": "Auto",
    "SUNPHARMA": "Pharma",
    "HCLTECH": "IT",
    "WIPRO": "IT",
    "BAJFINANCE": "NBFC",
    "TATASTEEL": "Metals",
    "NTPC": "Power",
    "POWERGRID": "Power",
}
