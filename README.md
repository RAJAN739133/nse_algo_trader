# NSE Intraday Algo Trader

A complete intraday algorithmic trading system for NSE (India) built for ₹1L capital.

## Strategies
1. **ORB (Opening Range Breakout)** — Trades the first 15-min candle breakout
2. **VWAP Mean Reversion** — Fades extremes from VWAP bands

## Features
- Kite Connect integration (Zerodha)
- ATR-based dynamic stop losses
- Position sizing (1% risk rule)
- Daily loss limit circuit breaker
- Trailing stops with Supertrend
- Full backtesting engine with realistic costs
- Telegram alerts

## Setup

```bash
# 1. Clone and enter the project
cd nse_algo_trader

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API keys
cp config/config_example.yaml config/config.yaml
# Edit config.yaml with your Zerodha API key and secret

# 5. Download historical data
python -m data.downloader

# 6. Run backtest
python -m backtest.runner

# 7. Paper trade (simulation mode)
python main.py --mode paper

# 8. Live trade (use with caution!)
python main.py --mode live
```

## Project Structure
```
nse_algo_trader/
├── config/
│   ├── config_example.yaml   # Template — copy to config.yaml
│   ├── config.yaml           # Your actual config (gitignored)
│   └── symbols.py            # Curated stock universe
├── data/
│   ├── downloader.py         # Kite Connect data fetcher
│   └── store.py              # Local data storage (SQLite)
├── strategies/
│   ├── base.py               # Base strategy class
│   ├── orb.py                # Opening Range Breakout
│   └── vwap_reversion.py     # VWAP mean reversion
├── risk/
│   ├── position_sizer.py     # 1% risk position sizing
│   ├── stop_loss.py          # ATR-based stops + trailing
│   └── circuit_breaker.py    # Daily/weekly loss limits
├── backtest/
│   ├── engine.py             # Backtesting engine
│   ├── costs.py              # Zerodha cost model
│   └── runner.py             # Run backtests, generate reports
├── utils/
│   ├── indicators.py         # Technical indicators (RSI, VWAP, ATR, etc.)
│   └── alerts.py             # Telegram notification helper
├── main.py                   # Entry point for paper/live trading
├── requirements.txt
└── README.md
```

## Risk Disclaimer
This is for educational purposes. Algo trading involves significant risk of loss.
Never trade with money you can't afford to lose.
