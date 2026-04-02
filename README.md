# NSE Intraday Algo Trader

Automated intraday trading bot for NSE India — ML stock selection, 5 adaptive strategies,
Claude AI brain, Angel One real-time data, **dynamic market trend detection**, regime detection, and Telegram alerts.

## Key Features (V3)

- **Dynamic Market Trend Detection** — Analyzes Nifty 50, VIX, SMAs at market open to determine BULLISH/BEARISH/NEUTRAL bias
- **Adaptive LONG/SHORT Trading** — Automatically adjusts trade direction limits based on market trend
- **Angel One Integration** — Real-time 5-min candles via SmartAPI (no Zerodha dependency)
- **ML Confidence Filter** — Only takes trades with 12%+ model confidence
- **Enhanced Telegram Alerts** — Trade entry/exit with timestamps, P&L, holding time, EOD summary with reasons
- **Regime Detection** — Per-stock regime (trending/ranging/volatile/choppy) with strategy adaptation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCHEDULER (cron 8:55 AM)                  │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│Claude V2 │ ML Model │News Scan │ VIX Check│ Angel One Login │
│morning   │ scoring  │ RSS feed │ yfinance │ SmartAPI TOTP   │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│         DYNAMIC MARKET TREND DETECTION (9:05 AM)            │
│  Nifty momentum + VIX + Gap + SMAs → BULLISH/BEARISH/NEUTRAL│
├─────────────────────────────────────────────────────────────┤
│              STOCK SELECTION (Top 8-10 from Nifty 100)      │
│  ML score + Delivery% + ATR fitness + Sector diversification │
│  Direction limits: BULLISH → 8L/2S | BEARISH → 2L/8S        │
├─────────────────────────────────────────────────────────────┤
│              REGIME DETECTOR (per stock, per hour)           │
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

# 3. Run live paper trading (market hours)
python live_paper_v3.py

# 4. Full automated day (scheduler)
python scheduler.py install                # Install cron
python scheduler.py run-now                # Start trading NOW
```

## Dynamic Market Trend Detection

At market open (9:05 AM), the system analyzes:

| Factor | Weight | Logic |
|--------|--------|-------|
| Weekly Momentum | ±30 pts | Nifty 50 5-day change (>2% strong, >0.5% mild) |
| Daily Gap | ±20 pts | Gap up/down from previous close |
| Price vs SMAs | ±20 pts | Above/below 5-day and 10-day SMAs |
| VIX Level | ±30 pts | High VIX (>25) = fear, Low VIX (<13) = complacent |

**Trend Classification:**
| Score | Trend | Max LONGS | Max SHORTS |
|-------|-------|-----------|------------|
| ≥30 | BULLISH | 8 | 2 |
| 10-29 | MILD_BULLISH | 6 | 3 |
| -9 to +9 | NEUTRAL | 5 | 5 |
| -10 to -29 | MILD_BEARISH | 3 | 6 |
| ≤-30 | BEARISH | 2 | 8 |

Example Telegram alert at market open:
```
📊 MARKET TREND: BEARISH (90% confidence)
  LONGS: ENABLED (max 2) | SHORTS: ENABLED (max 8)
  Reason: Strong weekly down -4.7% | Below both SMAs | High VIX 25.8
```

## Telegram Notifications

### Trade Entry
```
📉 TRADE ENTRY — SBIN
━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 Time: 13:45:00
💵 SELLING SHORT @ ₹1,003.15
🛑 Stop Loss: ₹1,011.76
🎯 Target: ₹986.73
━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Qty: 14 | Risk: 1.0%
💰 Risk: ₹120 | Reward: ₹230 (R:R 1.9)
🎯 Strategy: VWAP_SHORT
📈 Regime: choppy
📝 Reason: VWAP overbought +2.9σ RSI=84, ML=SHORT
```

### Trade Exit
```
✅ TRADE CLOSED — SBIN
━━━━━━━━━━━━━━━━━━━━━━━━━━
📥 SOLD SHORT: 13:45:00 @ ₹1,003.15
📤 COVERED: 14:10:00 @ ₹995.50
⏱️ Held for: 25 min
━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Qty: 14 | Return: +0.76%
💰 P&L: ₹+107.10 | Day: ₹+107.10
🎯 Strategy: VWAP_SHORT
📈 Regime: choppy | Exit: TARGET
```

### EOD Summary (with no-trade reasons)
```
📊 END OF DAY SUMMARY — 2026-04-02
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Market Trend: BEARISH
📊 VIX: 25.9

📭 NO TRADES TODAY

Possible reasons:
  • High VIX (25.9) - Market too volatile
  • No stocks passed ML confidence filter (>12%)
  • Low liquidity or volatility in scanned stocks
  • Regime unfavorable for selected strategies

🤖 V3: Dynamic trend | Regime detection | ML filter
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
| Confidence | abs(score - 50) / 100 must be ≥12% to trade |
| Composite | ML score + delivery bonus + trade count + ATR volatility fitness |
| Selection | Top 8-10 stocks, max 2 per sector, volume >5L, ATR 0.8-4.5% |

## 5 Trading Strategies

| # | Strategy | Time Window | How It Works | Best Regime |
|---|----------|-------------|--------------|-------------|
| 1 | **ORB Breakout** | 9:20-10:30 | First 15min range break with volume | Trending |
| 2 | **Pullback** | 10:00-11:30 | Re-entry at ORB level after pullback | Trending (LONG only) |
| 3 | **Momentum** | 11:30-14:00 | 0.5%+ move in 8 candles + volume 1.2x | Trending |
| 4 | **VWAP Reversion** | 11:00-14:00 | 2.0σ VWAP deviation + RSI extreme | Ranging/Choppy |
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
2. **Stop loss** — ATR-based, minimum 0.5% floor (disabled by default - time exits work better)
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
│   ├── candle_patterns.py    # Japanese candlestick pattern detection
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
│   ├── live_data_provider.py # Multi-source data provider with failover
│   ├── data_loader.py        # Unified loader (yfinance/kaggle/kite/CSV)
│   ├── train_pipeline.py     # ML training + feature engineering + scoring
│   ├── train_v2.py           # V2 ML training with bear market features
│   └── stock_performance_tracker.py # Rolling performance tracking
│
├── risk/
│   ├── position_sizer.py     # Risk-based position sizing
│   ├── stop_loss.py          # ATR stops + trailing
│   ├── circuit_breaker.py    # Daily/weekly loss limits
│   └── kill_switch.py        # Emergency stop
│
├── backtest/
│   ├── costs.py              # Angel One fee model (STT, brokerage, GST)
│   ├── production_backtest.py# Production-ready backtest
│   ├── improved_strategy.py  # Improved strategy backtest
│   ├── short_only_backtest.py# Short-only strategy test
│   └── walk_forward.py       # Walk-forward validation
│
├── config/
│   ├── config_test.yaml      # Test config (Angel One + Claude keys)
│   ├── config_prod.yaml      # Production config
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
python live_paper_v3.py                         # Auto-select best 8-10 stocks
python live_paper_v3.py --universe nifty50      # From Nifty 50
python live_paper_v3.py --universe nifty100     # From Nifty 100
python live_paper_v3.py --stocks HDFCBANK SBIN  # Specific stocks only
```

### ML Model
```bash
python -m data.train_pipeline train             # Train on real data (yfinance)
python -m data.train_pipeline demo              # Train on synthetic data
python -m data.train_pipeline score             # Score all stocks now
python -m data.train_v2 train                   # Train V2 model (bear market optimized)
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

### Analysis
```bash
python trade_analysis.py                        # Analyze past results
python compare_strategies.py                    # Compare strategy performance
python auto_optimizer.py                        # Auto-tune strategy params
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
| 9:05 AM | **Market Trend** | Detect BULLISH/BEARISH/NEUTRAL, set direction limits |
| 9:05 AM | Selection | ML scores 100 stocks, Claude reads news, picks top 8-10 |
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

### April 2, 2026 (Backtest - BEARISH Market)
```
Market Trend: BEARISH (90% confidence)
VIX: 25.9 | Weekly: -4.7%
Direction: 0 LONGS | 8 SHORTS selected

Trades: 3 | Won: 2 | WR: 67%
✅ ITC SHORT (VWAP)    Rs 292.95 → Rs 290.90  +Rs 43.44
✅ ITC SHORT (VWAP)    Rs 292.95 → Rs 292.30  +Rs 8.75
❌ CIPLA SHORT (VWAP)  Rs 1,185.90 → Rs 1,184.70  -Rs 0.84
Net P&L: Rs +51.35
```

## Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn (Random Forest), pandas, numpy
- **Broker:** Angel One SmartAPI (real-time data, paper/live orders)
- **Data:** yfinance (free), jugaad_data (NSE enrichment)
- **AI:** Claude Sonnet 4 API (strategy adaptation, news analysis)
- **Alerts:** Telegram Bot API
- **IDE:** PyCharm / IntelliJ with run configurations
- **Deploy:** Render / DigitalOcean / Oracle Cloud (free tier)
- **CI/CD:** GitHub + cron scheduler

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

## License

MIT License - Use at your own risk. This is for educational purposes. Always paper trade first!
