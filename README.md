# NSE Intraday Algo Trader

Automated intraday trading bot for NSE India — ML stock selection, 5 adaptive strategies,
Claude AI brain, Angel One real-time data, regime detection, and Telegram alerts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCHEDULER (cron 8:55 AM)                  │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│Claude V2 │ ML Model │News Scan │ VIX Check│ Angel One Login │
│morning   │ scoring  │ RSS feed │ yfinance │ SmartAPI TOTP   │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│              STOCK SELECTION (Top 8 from Nifty 50)           │
│  ML score + Delivery% + ATR fitness + Sector diversification │
├─────────────────────────────────────────────────────────────┤
│              REGIME DETECTOR (per stock, per hour)            │
│  trending_up │ trending_down │ ranging │ volatile │ choppy   │
├──────┬───────┬────────┬──────┬──────────────────────────────┤
│ ORB  │Pullbk │Momentum│ VWAP │ Afternoon Trend              │
│9:20  │10:00  │11:30   │11:00 │ 13:30                        │
├──────┴───────┴────────┴──────┴──────────────────────────────┤
│  Claude Brain V2 — live adjust every 15 min                  │
│  emergency exits │ strategy switches │ news alerts            │
├─────────────────────────────────────────────────────────────┤
│  RISK: 2% per trade │ 3% daily limit │ breakeven trail       │
│  Partial exit at 1x risk │ 45-min time decay │ circuit break │
├─────────────────────────────────────────────────────────────┤
│  Angel One (real-time data) │ yfinance (fallback)            │
│  Paper orders │ Telegram alerts │ CSV results                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/RAJAN739133/nse_algo_trader.git
cd nse_algo_trader
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Test APIs
python test_apis.py                        # Check Claude + Angel One

# 3. Run backtest on any date
python live_paper_v3.py --backtest 2026-04-01

# 4. Start live paper trading (market hours)
python live_paper_v3.py

# 5. Full automated day (scheduler)
python scheduler.py install                # Install cron
python scheduler.py run-now                # Start trading NOW
```

## APIs & Configuration

### Angel One (real-time data + paper orders)
```yaml
# config/config_test.yaml
broker:
  name: "angel_one"
  mode: "paper"
  data_source: "angel_one"
  angel_one:
    api_key: "YOUR_KEY"
    client_code: "YOUR_CLIENT"
    pin: "YOUR_PIN"
    totp_secret: "YOUR_TOTP"
```
Get from: [smartapi.angelone.in](https://smartapi.angelone.in/)

### Claude AI Brain (strategy adaptation)
```yaml
claude:
  enabled: true
  api_key: "sk-ant-api03-YOUR_KEY"
```
Get from: [console.anthropic.com](https://console.anthropic.com/settings/keys) — ~Rs 250-330/month

### Telegram Alerts
```bash
python setup_telegram.py                   # Interactive wizard
```

## ML Model

| Aspect | Detail |
|--------|--------|
| Algorithm | Random Forest (scikit-learn) |
| Features | 25+ indicators: RSI, MACD, Bollinger, ATR, SMA, EMA, OBV, volume ratio, delivery% |
| Training data | 5 years daily OHLCV via yfinance |
| Retrain | Weekly (Saturday) or on-demand |
| Scoring | Each stock scored 0-100 → direction LONG (>50) or SHORT (<50) |
| Composite | ML score + delivery bonus + trade count + ATR volatility fitness |
| Selection | Top 8 stocks, max 2 per sector, volume >5L, ATR 0.8-4% |

## 5 Trading Strategies

| # | Strategy | Time Window | How It Works | Best Regime |
|---|----------|-------------|--------------|-------------|
| 1 | **ORB Breakout** | 9:20-10:30 | First 15min range break with volume | Trending |
| 2 | **Pullback** | 10:00-11:30 | Re-entry at ORB level after pullback | Trending (LONG only) |
| 3 | **Momentum** | 11:30-14:00 | 0.6%+ move in 8 candles + volume 1.2x | Trending |
| 4 | **VWAP Reversion** | 11:00-14:00 | 2.5σ VWAP deviation + RSI extreme | Ranging/Choppy |
| 5 | **Afternoon Trend** | 13:30-14:30 | 5-candle trend + volume 1.3x + VWAP | Trending (60% size) |

All strategies: ML direction lock (no counter-trend) • ATR-based stops • Partial exit at 1x risk • Breakeven trail • 45-min time decay

## Claude Brain V2

Runs **throughout the trading day**, not just morning:

| When | What | API Calls |
|------|------|-----------|
| 8:50 AM | Morning brief: risk level, news sentiment, skip list, preferred stocks | 1 |
| Every 15 min | Live adjustment: emergency exits, strategy switches, stop tightening | ~15 |
| Before entry | Stock analysis: go/no-go with news + ML + candle data | ~5 |
| 3:30 PM | EOD learning: winning patterns, losing patterns, tomorrow adjustments | 1 |

Budget: 30 API calls/day • Cached + rate-limited • Safety limits enforced • Learnings saved to JSON

## Risk Management

1. **Position sizing** — 2% max risk per trade, regime-adjusted (volatile = 50% size)
2. **Stop loss** — ATR-based, minimum 0.5% floor
3. **Breakeven trail** — SL moves to entry after 1x risk profit
4. **Partial exit** — 50% position closed at 1x risk
5. **Time decay** — Flat trades closed after 45 minutes
6. **Daily limit** — 3% max daily loss, circuit breaker
7. **Max trades** — 3-5 per day (Claude-adjusted)
8. **Cost filter** — Skip if expected profit < 2x transaction costs

## Project Structure

```
nse_algo_trader/
├── live_paper_v3.py          # Main V3 adaptive trader (USE THIS)
├── scheduler.py              # Full day lifecycle + cron
├── test_apis.py              # Claude + Angel One API health check
├── test_angel.py             # Angel One detailed test suite
├── main.py                   # Legacy entry point
│
├── strategies/
│   ├── pro_strategy_v2.py    # ORB + pullback + momentum signals
│   ├── claude_brain_v2.py    # AI brain: morning/live/EOD/news
│   ├── claude_brain.py       # V1 brain (morning only)
│   ├── orb_v2.py             # ORB breakout strategy
│   ├── vwap_v2.py            # VWAP reversion strategy
│   └── base.py               # Signal/Trade base classes
│
├── data/
│   ├── angel_broker.py       # Paper/live broker (orders, LTP, positions)
│   ├── angel_auth.py         # Angel One TOTP login + session cache
│   ├── angel_ws.py           # WebSocket real-time data + candle builder
│   ├── angel_symbols.py      # NSE ticker → Angel One token mapper
│   ├── data_loader.py        # Unified loader (yfinance/kaggle/kite/CSV)
│   ├── train_pipeline.py     # ML training + feature engineering + scoring
│   └── data_config.yaml      # Data source configuration
│
├── risk/
│   ├── position_sizer.py     # Risk-based position sizing
│   ├── stop_loss.py          # ATR stops + trailing
│   ├── circuit_breaker.py    # Daily/weekly loss limits
│   └── kill_switch.py        # Emergency stop
│
├── backtest/
│   ├── engine.py             # Event-driven backtest engine
│   ├── costs.py              # Zerodha fee model (STT, brokerage, GST)
│   └── runner.py             # Backtest runner
│
├── config/
│   ├── config_test.yaml      # Test config (Angel One + Claude keys)
│   ├── config_prod.yaml      # Production config
│   ├── angel_config.yaml     # Angel One credentials
│   └── symbols.py            # Nifty 50/100/250 universe
│
├── utils/
│   ├── indicators.py         # ATR, RSI, VWAP, EMA, Bollinger, Supertrend
│   └── alerts.py             # Telegram notifications
│
├── models/                   # stock_predictor.pkl (auto-created)
├── results/                  # Daily trade CSVs (auto-created)
└── logs/                     # Daily log files (auto-created)
```

## Command Cheat Sheet

### Setup
```bash
git clone https://github.com/RAJAN739133/nse_algo_trader.git
cd nse_algo_trader
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python setup_telegram.py                        # Telegram wizard
```

### API Testing
```bash
python test_apis.py                             # Claude + Angel One health check
python test_angel.py                            # Angel One detailed test
python test_angel.py --ws                       # Angel One WebSocket test (60s)
python test_angel.py --ws --ws-duration 120     # WebSocket test (2 min)
```

### Live Paper Trading
```bash
python live_paper_v3.py                         # Auto-select best 8 stocks
python live_paper_v3.py --universe nifty50      # From Nifty 50
python live_paper_v3.py --universe nifty100     # From Nifty 100
python live_paper_v3.py --stocks HDFCBANK SBIN  # Specific stocks only
```

### Backtesting
```bash
python live_paper_v3.py --backtest 2026-04-01   # Replay a specific day
python live_paper_v3.py --backtest 2026-03-30   # Another day
python honest_backtest_30d.py                   # Last 30 days comparison
python run_backtest.py --last 30                # Legacy 30-day backtest
python run_backtest.py --date 2020-03-23        # COVID crash test
```

### ML Model
```bash
python -m data.train_pipeline train             # Train on real data (yfinance)
python -m data.train_pipeline demo              # Train on synthetic data
python -m data.train_pipeline score             # Score all stocks now
```

### Scheduler (Automated Daily)
```bash
python scheduler.py status                      # Check trading day + cron status
python scheduler.py install                     # Install cron jobs
python scheduler.py uninstall                   # Remove cron jobs
python scheduler.py run-now                     # Start trading NOW
python scheduler.py run-day                     # Full day lifecycle
python scheduler.py post-market                 # Run post-market analysis
python scheduler.py retrain                     # Force retrain ML model
python scheduler.py download-data               # Download all stock data
```

### Data
```bash
python -m data.data_loader status               # Show data sources + cache
python -m data.data_enricher download-all       # Download enriched NSE data
```

### Analysis
```bash
python trade_analysis.py                        # Analyze past results
python compare_strategies.py                    # Compare strategy performance
python auto_optimizer.py                        # Auto-tune strategy params
```

### Old Runners (still work)
```bash
python paper_trader.py simulate                 # V1 paper trader
python paper_trader.py scenarios                # All 5 market scenarios
python quick_test_today.py                      # Quick single-stock test
python start_bot.py                             # V1 full-day bot
python start_bot.py --once                      # V1 single-day run
```

## Data Sources

| Source | Cost | Speed | Best For |
|--------|------|-------|----------|
| Angel One SmartAPI | Free | Real-time tick | Live trading, LTP, candles |
| yfinance | Free | 15min delay | Training, backtesting, fallback |
| jugaad_data | Free | EOD | Delivery %, VWAP, trade count |
| Google News RSS | Free | Live | Market sentiment for Claude Brain |

## Daily Schedule (Automated via Cron)

| Time | Phase | What Happens |
|------|-------|--------------|
| 8:55 AM | Pre-market | Scheduler wakes up, checks holiday |
| 9:05 AM | Selection | ML scores 50 stocks, Claude reads news, picks top 8 |
| 9:15 AM | Connect | Angel One login, data streaming starts |
| 9:20 AM | Trading | Market open, ORB range forms, first signals |
| 9:20-10:30 | ORB | Breakout trades on high-confidence picks |
| 10:00-11:30 | Pullback | Re-entry on pullbacks to ORB levels |
| 11:30-14:00 | Momentum | Trend continuation trades |
| 11:00-14:00 | VWAP | Mean reversion in ranging/choppy regime |
| 13:30-14:30 | Afternoon | Late trend-following with strong signals |
| Every 15 min | Claude V2 | AI adjusts: exits, strategy switches, news |
| 15:10 PM | Square off | All positions closed, no overnight risk |
| 15:30 PM | Post-market | Claude EOD analysis, auto-optimizer, report |
| Saturday 6 AM | Weekly | Data download + ML model retrain |

## Recent Results

### April 1, 2026 (Backtest)
```
Trades: 7 | Won: 3 | WR: 43%
✅ CIPLA SHORT (ORB)    Rs 1,222 → Rs 1,197  +Rs 154
✅ CIPLA SHORT (ORB)    Rs 1,222 → Rs 1,195  +Rs 144
✅ HDFCLIFE SHORT (ORB) Rs 584 → Rs 583      +Rs 22
❌ SBIN SHORT (VWAP)    Rs 1,017 → Rs 1,024  -Rs 104
❌ RELIANCE SHORT       Rs 1,366 → Rs 1,369  -Rs 45
❌ HDFCLIFE SHORT       Rs 572 → Rs 572      -Rs 29
❌ TCS SHORT            Rs 2,411 → Rs 2,408  -Rs 1
Net P&L: Rs +141.63
```

## Tech Stack

- **Language:** Python 3.13
- **ML:** scikit-learn (Random Forest), pandas, numpy
- **Broker:** Angel One SmartAPI (real-time data, paper/live orders)
- **Data:** yfinance (free), jugaad_data (NSE enrichment)
- **AI:** Claude Sonnet 4 API (strategy adaptation, news analysis)
- **Alerts:** Telegram Bot API
- **IDE:** PyCharm / IntelliJ with run configurations
- **Deploy:** Render / DigitalOcean / Oracle Cloud (free tier)
- **CI/CD:** GitHub + cron scheduler

## IntelliJ / PyCharm Run Configs

| Name | What It Does |
|------|-------------|
| Test - All APIs | Check Claude + Angel One connectivity |
| Test - Angel One | Detailed Angel One test suite |
| Test - Simulate | Paper trade simulation |
| Test - Crash Day | Crash scenario (VIX 32) |
| Test - 10 Day Backtest | Multi-day backtest |
| Test - All Scenarios | Normal + volatile + crash + rally + flat |
| Train ML Model | Train stock predictor |
| Analyse Results | Review past results |
| PROD - Paper Mode | Full system paper trading |

## Requirements

```
Python 3.10+
smartapi-python    # Angel One (free)
yfinance           # Market data (free)
scikit-learn       # ML model
pandas, numpy      # Data processing
requests           # API calls
pyotp              # Angel One TOTP auth
pyyaml             # Config files
```

No paid APIs needed for backtesting. Angel One is free. Claude API ~Rs 250-330/month (optional but recommended).
