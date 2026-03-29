# NSE Intraday Algo Trader (Rajan Stock Bot)

Automated intraday trading system for NSE India with ML-powered stock selection,
ORB + VWAP strategies, 7-layer risk management, and Claude AI brain.

## Quick Start (Testing — No Money)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run paper trading simulation
python paper_trader.py simulate                    # Normal market day
python paper_trader.py simulate --scenario crash   # Test crash protection
python paper_trader.py simulate --scenario rally   # Test rally capture
python paper_trader.py multi-test --days 10        # 10-day backtest
python paper_trader.py scenarios                   # All 5 market scenarios
python paper_trader.py analyse                     # Review all results

# 3. Train ML model
python -m data.train_pipeline demo                 # Demo with synthetic data
python -m data.train_pipeline train                # Train on real data
python -m data.train_pipeline score                # Score stocks for today
```

## Two Environments

| | Testing | Production |
|---|---|---|
| Config | `config/config_test.yaml` | `config/config_prod.yaml` |
| Command | `python paper_trader.py simulate` | `python main.py --mode live` |
| Money | Rs 0 (virtual) | Rs 1L (real) |
| Zerodha | Not needed | Real API keys required |
| Where | Your laptop | Render / DigitalOcean |

## IntelliJ Run Configurations

Create these in Run > Edit Configurations > + Python:

| Name | Script | Parameters |
|------|--------|------------|
| Test - Simulate | `paper_trader.py` | `simulate` |
| Test - Crash | `paper_trader.py` | `simulate --scenario crash` |
| Test - 10 Days | `paper_trader.py` | `multi-test --days 10` |
| Test - All Scenarios | `paper_trader.py` | `scenarios` |
| Train ML | `data/train_pipeline.py` | `demo` |
| PROD Paper | `main.py` | `--mode paper` |
| PROD Live | `main.py` | `--mode live` |

Working directory: project root for all.

## Project Structure

```
nse_algo_trader/
  paper_trader.py          # TESTING: Paper trade simulator (start here!)
  main.py                  # PRODUCTION: Live trading engine
  config/
    config_test.yaml       # Test settings (no real keys needed)
    config_prod.yaml       # Production settings (gitignored!)
    config_example.yaml    # Template
    symbols.py             # Stock universe (Nifty 50 + 100)
  data/
    train_pipeline.py      # ML model training + stock scoring
    kite_auth.py           # Zerodha login + session cache
    downloader.py          # Historical data + live WebSocket
  strategies/
    base.py                # Signal/Trade classes
    orb.py                 # ORB v1 (basic)
    orb_v2.py              # ORB v2 (gap reversal + pullback filter)
    vwap_reversion.py      # VWAP v1 (basic)
    vwap_v2.py             # VWAP v2 (regime filter + trend alignment)
    pullback_filter.py     # False breakout detector
    claude_brain.py        # Claude AI strategy adaptation
  risk/
    position_sizer.py      # 1% risk rule + VIX adjustment
    stop_loss.py           # ATR stops + breakeven + trailing
    circuit_breaker.py     # Daily 3% + weekly 7% loss limits
    kill_switch.py         # Emergency stop (file/Telegram/config)
  backtest/
    engine.py              # Event-driven backtester
    costs.py               # Zerodha fee model
    runner.py              # Backtest runner
  utils/
    indicators.py          # Technical indicators
    alerts.py              # Telegram notifications
  results/                 # Paper trading results (auto-created)
  logs/                    # Daily logs (auto-created)
  models/                  # Trained ML models (auto-created)
```

## Deployment on Render

1. Push to GitHub: `git add -A && git commit -m "deploy" && git push`
2. Go to render.com > New > Blueprint
3. Connect your GitHub repo
4. Set environment variables in Render dashboard:
   - `KITE_API_KEY`, `KITE_API_SECRET`
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - `ANTHROPIC_API_KEY` (optional, for Claude Brain)
   - `TRADING_MODE=paper` (change to `live` when ready)
5. Deploy — it auto-starts as a worker process

## Strategy Overview

1. **8:50 AM** — Claude Brain adjusts risk level based on VIX, FII flows
2. **9:00 AM** — Pre-market filters eliminate dangerous stocks
3. **9:15-10:30** — ORB v2 trades breakouts (gap reversal + pullback filter)
4. **Continuous** — Stop loss manager: initial > breakeven > trailing
5. **14:00-15:10** — VWAP v2 on calm, range-bound days only
6. **15:10** — Mandatory square-off, daily summary

## Risk Management (7 Layers)

1. Pre-market filters (VIX, gaps, earnings, FII)
2. Position sizing (1% risk rule, VIX-adjusted)
3. ATR-based stop loss
4. 3-stage trailing stop
5. Max 2 trades/day
6. Circuit breaker (3% daily, 7% weekly)
7. Kill switch (3 methods)

## Requirements

- Python 3.10+
- Zerodha Kite Connect API (Rs 2,000/month) — only for live trading
- No API needed for paper trading / testing
