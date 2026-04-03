# NSE Algo Trader — Production Guide

## Complete Algorithmic Trading System for NSE India

This is a production-ready automated trading system featuring:
- **ML-based stock selection** (Random Forest with 50+ features)
- **5 adaptive trading strategies** (ORB, VWAP, Momentum, etc.)
- **Claude AI brain** for real-time market analysis
- **Angel One integration** for live/paper trading
- **Comprehensive risk management** (circuit breaker, position limits)
- **Real-time Telegram alerts**
- **Production monitoring dashboard**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCHEDULER                                 │
│  (Orchestrates entire trading day lifecycle)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  CLAUDE BRAIN │    │   ML SCORER   │    │  RISK ENGINE  │
│ (AI Analysis) │    │(Stock Ranking)│    │(Position/Stop)│
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE PAPER TRADER V3                          │
│  • Market trend detection (Nifty/VIX analysis)                   │
│  • 5 Strategy engine (ORB, VWAP, Momentum, Pullback, Afternoon)  │
│  • Dynamic position sizing (ML + Claude driven)                  │
│  • Circuit breaker (2% daily loss limit)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ ANGEL BROKER  │    │   TRADE DB    │    │   TELEGRAM    │
│(Orders/Data)  │    │(Audit Trail)  │    │   (Alerts)    │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Quick Start

### 1. Clone & Setup

```bash
cd /Users/rajananand/Downloads/nse_algo_trader
./deploy.sh dev
```

### 2. Configure API Keys

Edit `config/config_test.yaml`:

```yaml
broker:
  angel_one:
    api_key: "YOUR_API_KEY"
    client_id: "YOUR_CLIENT_ID"
    password: "YOUR_PASSWORD"
    totp_secret: "YOUR_TOTP_SECRET"
  mode: "paper"  # Start with paper trading!

telegram:
  bot_token: "YOUR_BOT_TOKEN"
  chat_ids: [YOUR_CHAT_ID]

claude:
  api_key: "YOUR_ANTHROPIC_KEY"

capital:
  total: 10000  # Rs 10,000 starting
  risk_per_trade: 0.02  # 2% risk
```

### 3. Test Connections

```bash
source .venv/bin/activate
python test_apis.py
```

### 4. Run Paper Trading

```bash
python live_paper_v3.py
```

### 5. Start Dashboard

```bash
streamlit run dashboard/production_dashboard.py --server.port 8501
```

---

## Trading Day Lifecycle

| Time | Event | Description |
|------|-------|-------------|
| 8:55 AM | Wake up | Scheduler starts, checks holiday |
| 9:00 AM | ML scoring | Score 100+ stocks, rank by signal |
| 9:05 AM | Trend detection | Analyze Nifty/VIX → BULLISH/BEARISH |
| 9:05 AM | Claude brief | AI reads news, suggests bias |
| 9:15 AM | Market open | Connect broker, start streaming |
| 9:20 AM | ORB forms | Opening range (15 min) captured |
| 9:20+ | Trading | Execute signals, manage positions |
| Every 5m | Quick scan | Claude checks regime shifts |
| Every 15m | Full scan | Claude deep analysis with news |
| 3:10 PM | Square off | Close all open positions |
| 3:30 PM | EOD report | Send Telegram summary |
| Saturday | Retrain | Weekly ML model update |

---

## Key Components

### 1. Live Paper Trader (`live_paper_v3.py`)

Main trading engine with:
- **Market trend detection**: Uses Nifty intraday data + VIX
- **5 strategies**: ORB, VWAP reversion, Momentum, Pullback, Afternoon
- **Dynamic position sizing**: ML + Claude with safety bounds
- **Circuit breaker**: Stops trading after 2% daily loss
- **Profit booking**: Quick profit at 0.8-1.2%, trailing stops

### 2. Claude Brain V2 (`strategies/claude_brain_v2.py`)

AI-powered analysis:
- **Morning brief** (8:55 AM): Pre-market analysis
- **Quick scan** (every 5 min): Regime check
- **Full scan** (every 15 min): News + sentiment
- **Stock analysis**: Go/no-go for each trade
- **EOD learning**: Post-market review

### 3. Pro Strategy V2 (`strategies/pro_strategy_v2.py`)

Technical analysis:
- Support/resistance levels (K-means clustering)
- Candlestick patterns (20+ patterns)
- Regime detection (trending/ranging/volatile)
- Multi-indicator signals (RSI, MACD, ADX, Supertrend)

### 4. Risk Management

| Control | Setting | Purpose |
|---------|---------|---------|
| Circuit breaker | 2% daily loss | Stop all trading |
| Position size | 10-15% max | Limit single trade risk |
| Max positions | 2 concurrent | Risk diversification |
| Stop loss | 0.5% dynamic | Cut losses fast |
| Quick profit | 0.8-1.2% | Lock in gains |
| LONGs in bearish | DISABLED | Trade with market |

### 5. Trade Database (`core/trade_database.py`)

SQLite/PostgreSQL storage:
- Complete trade audit trail
- Daily/weekly/monthly summaries
- Performance analytics
- Export to CSV/JSON

### 6. System Monitor (`core/system_monitor.py`)

Production monitoring:
- Health checks (broker, DB, internet)
- Auto-restart on failures
- Resource monitoring (CPU, RAM, disk)
- Heartbeat alerts (hourly)
- Kill switch for emergencies

---

## Configuration Reference

### Risk Settings (Safety Bounds)

```yaml
# These bounds cannot be overridden by ML or Claude
quick_profit: [0.5%, 2.0%]   # Exit in profit
stop_loss: [0.3%, 1.5%]      # Cut losses
position_size: [5%, 15%]     # Per trade
circuit_breaker: 2%          # Daily loss limit
```

### Trading Controls

```yaml
strategy_tweaks:
  target_pct: 0.012          # 1.2% target
  stop_pct: 0.005            # 0.5% stop
  quick_profit_pct: 0.008    # 0.8% book profit
  cut_loss_pct: 0.003        # 0.3% cut loser
  time_decay_candles: 16     # 80 min max hold
```

### Direction Controls

```yaml
# Automatically set based on market trend
BULLISH:       enable_longs=True,  max_longs=5, enable_shorts=True,  max_shorts=2
MILD_BEARISH:  enable_longs=False, max_longs=0, enable_shorts=True,  max_shorts=5
BEARISH:       enable_longs=False, max_longs=0, enable_shorts=True,  max_shorts=6
NEUTRAL:       enable_longs=True,  max_longs=3, enable_shorts=True,  max_shorts=3
```

---

## Production Deployment

### Option 1: Manual Run

```bash
# In terminal (or tmux/screen)
source .venv/bin/activate
python scheduler.py
```

### Option 2: Systemd Service (Recommended)

```bash
# Install service
sudo cp nse_algo_trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nse_algo_trader
sudo systemctl start nse_algo_trader

# Check status
sudo systemctl status nse_algo_trader

# View logs
journalctl -u nse_algo_trader -f

# Stop trading
sudo systemctl stop nse_algo_trader
```

### Option 3: Docker (Coming Soon)

```bash
docker-compose up -d
```

---

## Dashboard

Access at: `http://localhost:8501`

Features:
- **Overview**: Today's P&L, open positions, recent trades
- **Positions**: Real-time position tracking
- **Trade History**: Filter by date, symbol, strategy
- **Analytics**: Win rate, profit factor, drawdown
- **System Health**: CPU/RAM, component status
- **Kill Switch**: Emergency trading halt

---

## Emergency Procedures

### Kill Switch Activation

**Via Dashboard:**
1. Go to sidebar → Emergency Controls
2. Enter reason
3. Click "ACTIVATE KILL SWITCH"

**Via Command Line:**
```bash
python -c "from core.system_monitor import KillSwitch; KillSwitch.activate('Manual halt')"
```

**Via Telegram:**
Send `/killswitch` to your bot (if configured)

### Deactivation

```bash
python -c "from core.system_monitor import KillSwitch; KillSwitch.deactivate()"
```

---

## Monitoring & Alerts

### Telegram Notifications

You receive alerts for:
- ✅ Trade entries/exits with P&L
- 📊 Hourly heartbeat with system status
- ⚠️ Circuit breaker triggered
- 🚨 Kill switch activated
- 📈 End-of-day summary

### Log Files

| File | Content |
|------|---------|
| `logs/trading_YYYY-MM-DD.log` | Daily trading log |
| `logs/service.log` | Systemd service output |
| `logs/service_error.log` | Error log |

---

## Backtest Before Live

Always backtest strategy changes:

```bash
# 60-day backtest
python backtest_daily.py --days 60 --capital 10000

# 90-day with specific universe
python backtest_daily.py --days 90 --capital 10000 --universe nifty50
```

---

## Troubleshooting

### Common Issues

**1. Angel One connection failed**
- Check API keys in config
- Verify TOTP secret is correct
- Check if 2FA app is synced

**2. No trades being made**
- Check if market is open
- Verify VIX is not extreme (>25)
- Check if circuit breaker triggered
- Verify capital is sufficient

**3. Telegram not working**
- Test bot: `python setup_telegram.py`
- Verify chat_id is correct
- Check bot token

**4. ML model errors**
- Retrain model: `python -m ml.train_model`
- Check if data files exist

---

## Performance Tips

1. **Start with paper trading** for 2-4 weeks
2. **Begin with Rs 10-25K** until strategy validates
3. **Monitor win rate** — target 55%+ 
4. **Watch drawdown** — max 5% acceptable
5. **Review weekly** — adjust parameters based on results
6. **Retrain ML monthly** — keep model fresh

---

## Support

For issues:
1. Check logs in `logs/` directory
2. Review trade database for patterns
3. Run diagnostics: `python test_apis.py`

---

## License

MIT License — Use at your own risk. Trading involves significant financial risk.
