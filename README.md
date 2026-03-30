# NSE Intraday Algo Trader (Rajan Stock Bot)

Automated intraday trading system for NSE India with ML-powered stock selection,
ORB + VWAP strategies, 7-layer risk management, Claude AI brain, and Telegram alerts.

## Quick Start (Testing — No Money Required)

```bash
# 1. Install dependencies
pip install pyyaml pandas numpy scikit-learn scipy matplotlib requests yfinance

# 2. Run paper trading simulation (works instantly, no API keys needed)
python paper_trader.py simulate                    # Normal market day
python paper_trader.py simulate --scenario crash   # Test crash protection
python paper_trader.py simulate --scenario rally   # Test rally capture
python paper_trader.py multi-test --days 10        # 10-day backtest
python paper_trader.py scenarios                   # All 5 market scenarios
python paper_trader.py analyse                     # Review all results

# 3. Train ML model (synthetic data, no internet needed)
python -m data.train_pipeline demo

# 4. Backtest on real historical data (needs yfinance)
python run_backtest.py --date 2024-03-15           # Any specific date
python run_backtest.py --last 30                   # Last 30 trading days
python run_backtest.py --date 2020-03-23           # COVID crash test!
```

## Two Environments

| | Testing | Production |
|---|---|---|
| Config | `config/config_test.yaml` | `config/config_prod.yaml` |
| Command | `python paper_trader.py simulate` | `python main.py --mode live` |
| Money | Rs 0 (virtual) | Rs 1L (real) |
| Zerodha | Not needed | API keys required (Rs 2000/month) |
| Claude Brain | Works with API key | Works with API key |
| Where | Your laptop (IntelliJ) | Render / DigitalOcean |

## API Keys — Where to Enter

All credentials go in the config YAML files. These files are gitignored (never pushed to GitHub).

### 1. Claude AI Brain (for smart strategy adaptation)
```yaml
# In config/config_test.yaml AND config/config_prod.yaml
claude:
  enabled: true
  api_key: "sk-ant-api03-YOUR_KEY_HERE"
```
**Get your key:** Go to [console.anthropic.com](https://console.anthropic.com/settings/keys) → sign up → Settings → API Keys → Create Key. Cost: ~Rs 250-330/month.

### 2. Zerodha Kite Connect (for live trading only)
```yaml
zerodha:
  api_key: "your_kite_api_key"
  api_secret: "your_kite_api_secret"
```
**Get your key:** Go to [developers.kite.trade](https://developers.kite.trade) → Create App → Copy API Key + Secret. Cost: Rs 2000/month. Not needed for paper trading.

### 3. Telegram Alerts (trade notifications on your phone)
```bash
python setup_telegram.py    # Interactive wizard — does everything for you
```
Or manually: Search @BotFather on Telegram → /newbot → get token. Search @userinfobot → get chat ID.

### 4. For Render Deployment (environment variables)
```
ANTHROPIC_API_KEY = sk-ant-api03-...
KITE_API_KEY      = your_zerodha_key
KITE_API_SECRET   = your_zerodha_secret
TRADING_MODE      = paper
```

## Data Sources — Configurable

Edit `data/data_config.yaml` to switch data sources. No code changes needed.

```yaml
training_source: "synthetic"    # Options: synthetic, yfinance, kaggle, csv_folder
live_source: "yfinance"         # Options: yfinance, kite, synthetic
backtest_source: "yfinance"     # Options: yfinance, kaggle, csv_folder, synthetic
```

| Source | Cost | Data | Best For |
|--------|------|------|----------|
| synthetic | Free | Generated fake data | Quick testing, no internet |
| yfinance | Free | Real NSE data, ~15min delay | Training, backtesting |
| kaggle | Free | 25 years Nifty 50 daily | ML model training |
| kite | Rs 2000/month | Real-time tick-by-tick | Live trading |
| csv_folder | Free | Any CSVs you have | Custom datasets |

## IntelliJ Run Configurations

Pre-configured — just click the green play button:

| Name | What It Does |
|------|--------------|
| Test - Simulate | Paper trade one normal day |
| Test - Crash Day | Test crash protection (VIX 32) |
| Test - 10 Day Backtest | Multi-day simulation |
| Test - All Scenarios | Normal + volatile + crash + rally + flat |
| Train ML Model | Train stock predictor on synthetic data |
| Backtest - Specific Date | Test on 2024-06-04 (change date in config) |
| Backtest - Last 30 Days | Backtest recent 30 trading days |
| Data - Check Status | Show what data sources are available |
| PROD - Paper Mode | Full system in paper mode |
| Analyse Results | Review all paper trading results |

## Project Structure

```
nse_algo_trader/
  paper_trader.py           # TESTING — paper trade simulator
  run_backtest.py           # TESTING — backtest on any historical date
  main.py                   # PRODUCTION — live trading engine
  setup_telegram.py         # SETUP — Telegram bot wizard

  config/
    config_example.yaml     # Template (committed to git)
    config_test.yaml        # Your test config (gitignored, has your keys)
    config_prod.yaml        # Your prod config (gitignored, has your keys)
    symbols.py              # Stock universe (Nifty 50 + 100)

  data/
    data_config.yaml        # Switch data sources here
    data_loader.py          # Unified loader (yfinance/kaggle/kite/CSV/synthetic)
    train_pipeline.py       # ML model training + stock scoring
    kite_auth.py            # Zerodha login + MockKite for testing
    downloader.py           # Historical data + WebSocket feed

  strategies/
    base.py                 # Signal/Trade/Result classes
    orb.py                  # ORB v1 (basic breakout)
    orb_v2.py               # ORB v2 (gap reversal + pullback filter)
    vwap_reversion.py       # VWAP v1 (basic mean reversion)
    vwap_v2.py              # VWAP v2 (regime filter + RSI + EMA alignment)
    pullback_filter.py      # False breakout detector
    claude_brain.py         # Claude AI strategy adaptation

  risk/
    position_sizer.py       # 1% risk rule + VIX adjustment
    stop_loss.py            # ATR stops → breakeven → trailing
    circuit_breaker.py      # Daily 3% + weekly 7% loss limits
    kill_switch.py          # Emergency stop (3 methods)

  backtest/
    engine.py               # Event-driven backtester
    costs.py                # Zerodha fee model (all charges)
    runner.py               # Backtest runner

  utils/
    indicators.py           # ATR, RSI, VWAP, EMA, Supertrend, Bollinger
    alerts.py               # Telegram notifications

  models/                   # Trained ML models (auto-created)
  results/                  # Paper trading results (auto-created)
  logs/                     # Daily logs (auto-created)
```

## Strategy Overview

1. **8:50 AM** — Claude Brain analyses VIX, FII flows, news → adjusts risk level
2. **9:00 AM** — Pre-market filters eliminate dangerous stocks (7 checks)
3. **9:15-10:30** — ORB v2: breakouts with gap reversal + pullback filter
4. **Continuous** — Stop loss manager: initial → breakeven → trailing (Supertrend)
5. **14:00-15:10** — VWAP v2: mean reversion on calm, range-bound days only
6. **15:10** — Mandatory square-off, daily summary to Telegram
7. **3:15 PM** — Claude Brain end-of-day analysis: what went right/wrong

## Risk Management (7 Layers)

1. Pre-market filters (VIX, gaps, earnings, FII/DII, news, commodity prices)
2. Position sizing (1% risk rule, VIX-adjusted, FII-adjusted)
3. ATR-based stop loss (adapts to each stock's volatility)
4. 3-stage trailing stop (initial → breakeven → Supertrend)
5. Max 2 trades/day (prevents overtrading)
6. Circuit breaker (3% daily, 7% weekly hard limits)
7. Kill switch (KILL file / Telegram /stop / config flag)

## ML Model

- **Algorithm:** GradientBoosting (scikit-learn, no extra packages needed)
- **Features:** 30+ technical indicators + 12 candlestick patterns
- **Training:** Time-series cross-validation (no future data leakage)
- **Scoring:** Each stock scored 0-100 daily. Algo trades top 2-3 with score ≥ 60
- **Retrain:** Weekly (Saturday) with latest data

## Deployment

### Render (recommended)
1. Push to GitHub
2. Go to render.com → New → Blueprint → Connect your repo
3. Set environment variables in Render dashboard
4. Auto-deploys from `render.yaml`

### DigitalOcean (free with student credits)
- GitHub Student Pack → $200 DigitalOcean credit → 33 months free
- Create Mumbai droplet → deploy algo

### Oracle Cloud (free forever)
- 2 free VMs forever, Mumbai region available

## Command Cheat Sheet

```bash
# ═══ SETUP ═══
pip install pyyaml pandas numpy scikit-learn scipy matplotlib requests yfinance
python setup_telegram.py                    # Connect Telegram

# ═══ TEST ═══
python paper_trader.py simulate             # Quick test
python paper_trader.py scenarios            # All 5 scenarios
python paper_trader.py multi-test --days 10 # 10-day test

# ═══ BACKTEST ON REAL DATA ═══
python run_backtest.py --date 2024-03-15    # Specific date
python run_backtest.py --last 30            # Last 30 days

# ═══ ML MODEL ═══
python -m data.train_pipeline demo          # Train (synthetic)
python -m data.train_pipeline train         # Train (real data)
python -m data.train_pipeline score         # Score stocks

# ═══ DATA STATUS ═══
python -m data.data_loader status           # What data you have

# ═══ PRODUCTION ═══
python main.py --mode paper                 # Safe paper mode
python main.py --mode live                  # REAL MONEY (careful!)
```

## Requirements

- Python 3.10+
- No paid APIs needed for testing
- Zerodha Kite Connect (Rs 2000/month) — only for live trading
- Claude API key (~Rs 250-330/month) — for AI brain (optional but recommended)
- Telegram account — for trade alerts on phone (free)
