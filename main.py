#!/usr/bin/env python3
"""
Rajan Stock Bot — NSE Intraday Algo Trader
==========================================
Entry point for both PAPER (test) and LIVE (prod) modes.

Usage:
    python main.py --mode paper        # Test mode (no real money)
    python main.py --mode live         # Production (real money!)
    python main.py --mode backtest     # Run on historical data
    python main.py --mode analyse      # Analyse today's trades after market close
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from config.symbols import DEFAULT_UNIVERSE, ALL_STOCKS, STOCK_SECTORS
from strategies.base import Signal, TradeSignal, TradeResult
from strategies.orb import ORBStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from risk.position_sizer import PositionSizer
from risk.circuit_breaker import CircuitBreaker
from utils.indicators import add_all_indicators

# ============================================================
# Config Loader
# ============================================================

def load_config(mode: str) -> dict:
    """Load config based on mode — paper uses test, live uses prod."""
    config_dir = Path(__file__).parent / "config"
    
    if mode in ("paper", "backtest", "analyse"):
        config_path = config_dir / "config_test.yaml"
    elif mode == "live":
        config_path = config_dir / "config_prod.yaml"
    else:
        config_path = config_dir / "config_test.yaml"
    
    if not config_path.exists():
        # Fall back to example
        config_path = config_dir / "config_example.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Force paper mode for safety unless explicitly live
    if mode != "live":
        config["trading"]["mode"] = "paper"
    
    return config


# ============================================================
# Paper Trading Engine (simulates live market with real data)
# ============================================================

class PaperTrader:
    """
    Simulates trading without real money.
    Uses real-time or historical market data.
    Tracks virtual P&L, positions, and trade journal.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.capital = config["capital"]["total"]
        self.initial_capital = self.capital
        self.positions = {}  # symbol -> position dict
        self.trades = []     # completed trades
        self.journal = []    # all decisions (trade + skip)
        self.day_pnl = 0.0
        self.day_trades = 0
        
        # Initialize strategies
        self.orb = ORBStrategy(config.get("strategies", {}).get("orb", {}))
        self.vwap = VWAPReversionStrategy(config.get("strategies", {}).get("vwap", {}))ReversionStrategy(config.get("strategies", {}).get("vwap", {}))

        # Risk management
        cap_config = config["capital"]
        self.sizer = PositionSizer(
            capital=cap_config["total"],
            risk_per_trade=cap_config.get("risk_per_trade", 0.01),
        )
        self.circuit = CircuitBreaker(
            capital=cap_config["total"],
            daily_loss_limit=cap_config.get("daily_loss_limit", 0.03),
            weekly_loss_limit=cap_config.get("weekly_loss_limit", 0.07),
            max_trades_per_day=cap_config.get("max_trades_per_day", 2),
        )
        
        self.logger = logging.getLogger("PaperTrader")
    
    def run_on_data(self, data: dict[str, pd.DataFrame]) -> dict:
        """
        Run strategies on provided data (historical or live).
        
        Args:
            data: dict of {symbol: DataFrame with OHLCV + indicators}
        
        Returns:
            Summary dict with trades, P&L, etc.
        """
        today = date.today().isoformat()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"  RAJAN STOCK BOT — Paper Trading")
        self.logger.info(f"  Date: {today}")
        self.logger.info(f"  Capital: ₹{self.capital:,.0f}")
        self.logger.info(f"  Stocks: {len(data)}")
        self.logger.info(f"{'='*60}\n")
        
        for symbol, df in data.items():
            if len(df) < 20:
                self.journal.append({
                    "symbol": symbol, "action": "SKIP",
                    "reason": f"Not enough data ({len(df)} bars)",
                    "time": datetime.now().isoformat(),
                })
                continue
            
            # Add indicators if not already present
            if "atr" not in df.columns:
                df = add_all_indicators(df)
            
            self._run_strategy_on_stock(symbol, df)
        
        return self._generate_summary()
    
    def _run_strategy_on_stock(self, symbol: str, df: pd.DataFrame):
        """Run both strategies on a single stock's data."""
        
        # Check circuit breaker
        if self.circuit.should_stop(self.day_pnl, self.initial_capital):
            self.journal.append({
                "symbol": symbol, "action": "SKIP",
                "reason": "Circuit breaker active",
                "time": datetime.now().isoformat(),
            })
            return
        
        # Max trades per day check
        max_trades = self.config["capital"].get("max_trades_per_day", 2)
        if self.day_trades >= max_trades:
            self.journal.append({
                "symbol": symbol, "action": "SKIP",
                "reason": f"Max trades ({max_trades}) reached",
                "time": datetime.now().isoformat(),
            })
            return
        
        # Try each strategy
        for strategy in [self.orb, self.vwap]:
            if not strategy.is_active:
                continue
            
            # Walk through candles (simulate live)
            for i in range(20, len(df)):
                signal = strategy.generate_signal(df, i)
                
                if signal is None:
                    continue
                
                # Calculate position size
                risk = abs(signal.entry_price - signal.stop_loss)
                if risk == 0:
                    continue
                
                max_risk_amount = self.capital * self.config["capital"]["risk_per_trade"]
                shares = int(max_risk_amount / risk)
                
                if shares <= 0:
                    continue
                
                cost = shares * signal.entry_price
                if cost > self.capital * 0.9:  # don't use more than 90% of capital
                    shares = int(self.capital * 0.9 / signal.entry_price)
                
                if shares <= 0:
                    continue
                
                # Simulate the trade
                result = self._simulate_trade(df, i, signal, shares)
                
                if result:
                    self.trades.append(result)
                    self.day_pnl += result.pnl
                    self.capital += result.pnl
                    self.day_trades += 1
                    
                    self.journal.append({
                        "symbol": symbol,
                        "action": f"{signal.signal.value}",
                        "strategy": strategy.name,
                        "entry": result.entry_price,
                        "exit": result.exit_price,
                        "shares": result.shares,
                        "pnl": round(result.pnl, 2),
                        "exit_reason": result.exit_reason,
                        "reason": signal.reason,
                        "time": datetime.now().isoformat(),
                    })
                    
                    emoji = "✅" if result.pnl > 0 else "❌"
                    self.logger.info(
                        f"  {emoji} {symbol} | {strategy.name} | "
                        f"{signal.signal.value} @ ₹{result.entry_price:.2f} → "
                        f"₹{result.exit_price:.2f} | "
                        f"P&L: ₹{result.pnl:+,.2f} | {result.exit_reason}"
                    )
                    
                    break  # one trade per stock
            
            if self.day_trades >= max_trades:
                break
    
    def _simulate_trade(
        self, df: pd.DataFrame, entry_idx: int,
        signal: TradeSignal, shares: int,
    ) -> TradeResult | None:
        """Simulate a trade from entry to exit."""
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        target = signal.target_price
        side = signal.signal.value
        
        # Walk forward from entry
        for i in range(entry_idx + 1, len(df)):
            row = df.iloc[i]
            high, low, close = row["high"], row["low"], row["close"]
            
            # Check stop loss
            if side == "BUY" and low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
            elif side == "SELL" and high >= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
            # Check target
            elif side == "BUY" and high >= target:
                exit_price = target
                exit_reason = "target"
            elif side == "SELL" and low <= target:
                exit_price = target
                exit_reason = "target"
            # Square off at last candle
            elif i == len(df) - 1:
                exit_price = close
                exit_reason = "square_off"
            else:
                continue
            
            # Calculate P&L
            if side == "BUY":
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares
            
            # Deduct costs (Zerodha intraday)
            pnl -= self._calculate_costs(entry_price, exit_price, shares)
            
            return TradeResult(
                symbol=signal.symbol,
                side=side,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                shares=shares,
                pnl=round(pnl, 2),
                pnl_percent=round(pnl / (entry_price * shares) * 100, 2),
                entry_time=str(df.index[entry_idx]) if hasattr(df.index[entry_idx], 'strftime') else str(entry_idx),
                exit_time=str(df.index[i]) if hasattr(df.index[i], 'strftime') else str(i),
                exit_reason=exit_reason,
                strategy_name=signal.strategy_name,
            )
        
        return None
    
    def _calculate_costs(self, entry: float, exit: float, shares: int) -> float:
        """Calculate Zerodha intraday trading costs."""
        turnover = (entry + exit) * shares
        brokerage = min(40, turnover * 0.0003)  # 0.03% or ₹20/side
        stt = turnover * 0.00025 / 2  # sell side only
        exchange = turnover * 0.0000345
        gst = brokerage * 0.18
        sebi = turnover * 0.000001
        stamp = entry * shares * 0.00003
        return brokerage + stt + exchange + gst + sebi + stamp
    
    def _generate_summary(self) -> dict:
        """Generate end-of-day summary."""
        summary = {
            "date": date.today().isoformat(),
            "mode": "PAPER",
            "capital_start": self.initial_capital,
            "capital_end": round(self.capital, 2),
            "day_pnl": round(self.day_pnl, 2),
            "day_pnl_pct": round(self.day_pnl / self.initial_capital * 100, 2),
            "total_trades": len(self.trades),
            "wins": sum(1 for t in self.trades if t.pnl > 0),
            "losses": sum(1 for t in self.trades if t.pnl <= 0),
            "trades": [
                {
                    "symbol": t.symbol, "side": t.side,
                    "entry": t.entry_price, "exit": t.exit_price,
                    "shares": t.shares, "pnl": t.pnl,
                    "strategy": t.strategy_name, "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ],
            "journal": self.journal,
        }
        
        return summary


# ============================================================
# Data Generator (synthetic for testing without Kite Connect)
# ============================================================

def generate_test_data(symbols: list, n_candles: int = 75) -> dict[str, pd.DataFrame]:
    """
    Generate realistic intraday 5-min candles for testing.
    75 candles = full trading day (9:15 to 15:30).
    """
    np.random.seed(int(datetime.now().timestamp()) % 100000)
    data = {}
    
    for symbol in symbols:
        # Random starting price based on typical NSE stock prices
        base_price = np.random.choice([150, 300, 500, 800, 1200, 1500, 2500])
        
        # Generate intraday price path
        returns = np.random.randn(n_candles) * 0.003  # 0.3% per candle
        
        # Add intraday patterns
        # Opening volatility (first 3 candles)
        returns[:3] *= 2.5
        # Lunch hour calm (candles 25-45)
        returns[25:45] *= 0.4
        # Closing action (last 10 candles)
        returns[-10:] *= 1.5
        
        # Add trend bias (slight bullish or bearish)
        trend = np.random.choice([-1, 0, 1]) * 0.001
        returns += trend
        
        close = base_price * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.randn(n_candles)) * 0.002)
        low = close * (1 - np.abs(np.random.randn(n_candles)) * 0.002)
        open_p = np.concatenate([[base_price], close[:-1]]) + np.random.randn(n_candles) * base_price * 0.001
        volume = np.random.lognormal(12, 0.8, n_candles).astype(int)
        
        # First candle has higher volume (opening auction)
        volume[0] *= 3
        volume[-1] *= 2
        
        # Create timestamps (9:15 AM to 3:30 PM, 5-min candles)
        today = date.today()
        base_time = datetime(today.year, today.month, today.day, 9, 15)
        times = [base_time + timedelta(minutes=5 * i) for i in range(n_candles)]
        
        df = pd.DataFrame({
            "datetime": times,
            "open": np.round(open_p, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
            "symbol": symbol,
        })
        df.set_index("datetime", inplace=True)
        
        data[symbol] = df
    
    return data


# ============================================================
# Trade Analyser (post-market analysis)
# ============================================================

def analyse_trades(journal_path: str = None):
    """Analyse today's trades after market close."""
    journal_dir = Path(__file__).parent / "logs"
    
    if journal_path:
        path = Path(journal_path)
    else:
        # Find today's journal
        today_str = date.today().isoformat()
        path = journal_dir / f"journal_{today_str}.json"
    
    if not path.exists():
        print(f"  No journal found at {path}")
        print(f"  Run paper trade first: python main.py --mode paper")
        return
    
    with open(path) as f:
        summary = json.load(f)
    
    print("\n" + "=" * 60)
    print("  POST-MARKET ANALYSIS")
    print("=" * 60)
    print(f"  Date: {summary['date']}")
    print(f"  Mode: {summary['mode']}")
    print(f"  Capital: ₹{summary['capital_start']:,.0f} → ₹{summary['capital_end']:,.0f}")
    print(f"  Day P&L: ₹{summary['day_pnl']:+,.2f} ({summary['day_pnl_pct']:+.2f}%)")
    print(f"  Trades: {summary['total_trades']} ({summary['wins']} wins, {summary['losses']} losses)")
    
    if summary["trades"]:
        print(f"\n  {'Symbol':<12} {'Side':<5} {'Entry':>8} {'Exit':>8} {'P&L':>10} {'Strategy':<10} {'Exit Reason'}")
        print("  " + "─" * 70)
        for t in summary["trades"]:
            emoji = "✅" if t["pnl"] > 0 else "❌"
            print(f"  {t['symbol']:<12} {t['side']:<5} ₹{t['entry']:>7.2f} ₹{t['exit']:>7.2f} ₹{t['pnl']:>+9.2f} {t['strategy']:<10} {t['exit_reason']} {emoji}")
    
    # Decisions journal
    decisions = summary.get("journal", [])
    skips = [d for d in decisions if d.get("action") == "SKIP"]
    if skips:
        print(f"\n  Skipped decisions ({len(skips)}):")
        for s in skips[:10]:
            print(f"    {s['symbol']:<12} — {s['reason']}")
    
    # Recommendations
    print(f"\n  RECOMMENDATIONS:")
    if summary["day_pnl"] > 0:
        print(f"  ✅ Profitable day. System working as expected.")
    else:
        print(f"  ⚠️  Loss day. Check if filters need tightening.")
    
    if summary["total_trades"] == 0:
        print(f"  ℹ️  No trades taken. This might be correct (bad market conditions) or the filters are too strict.")
    
    win_rate = summary["wins"] / max(1, summary["total_trades"]) * 100
    print(f"  📊 Win rate: {win_rate:.0f}%")
    if win_rate < 60 and summary["total_trades"] > 3:
        print(f"  ⚠️  Win rate below 60%. Consider tightening entry criteria.")
    
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Rajan Stock Bot — NSE Algo Trader")
    parser.add_argument("--mode", choices=["paper", "live", "backtest", "analyse"],
                        default="paper", help="Trading mode (default: paper)")
    parser.add_argument("--stocks", nargs="+", default=None,
                        help="Override stock list (e.g., --stocks RELIANCE SBIN)")
    parser.add_argument("--journal", default=None,
                        help="Path to journal file for analyse mode")
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"bot_{date.today().isoformat()}.log"),
        ]
    )
    logger = logging.getLogger()
    
    # ── Analyse mode ──
    if args.mode == "analyse":
        analyse_trades(args.journal)
        return
    
    # ── Load config ──
    config = load_config(args.mode)
    mode = config["trading"]["mode"]
    
    logger.info(f"\n  🤖 Rajan Stock Bot starting...")
    logger.info(f"  Mode: {'🟢 PAPER (test)' if mode == 'paper' else '🔴 LIVE (real money!)'}")
    logger.info(f"  Capital: ₹{config['capital']['total']:,.0f}")
    
    # Safety check for live mode
    if mode == "live":
        logger.warning("\n  ⚠️  LIVE MODE — REAL MONEY WILL BE USED!")
        confirm = input("  Type 'YES' to confirm: ")
        if confirm != "YES":
            logger.info("  Aborted.")
            return
    
    # ── Select stocks ──
    symbols = args.stocks or DEFAULT_UNIVERSE
    logger.info(f"  Stocks: {len(symbols)} ({', '.join(symbols[:5])}...)")
    
    # ── Get data ──
    if mode == "paper":
        logger.info(f"\n  📊 Generating simulated intraday data...")
        data = generate_test_data(symbols)
        logger.info(f"  Generated {sum(len(v) for v in data.values())} candles for {len(data)} stocks")
        
        # Add indicators
        for sym in data:
            data[sym] = add_all_indicators(data[sym])
    
    elif mode == "live":
        logger.info(f"\n  📊 Connecting to Kite Connect for live data...")
        # This requires kiteconnect package and valid API keys
        # Placeholder — replace with actual Kite Connect integration
        logger.error("  Live data connection not configured yet.")
        logger.info("  Set up Kite Connect API keys in config/config_prod.yaml")
        return
    
    # ── Run trading ──
    trader = PaperTrader(config)
    summary = trader.run_on_data(data)
    
    # ── Save journal ──
    journal_path = log_dir / f"journal_{date.today().isoformat()}.json"
    with open(journal_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\n  📝 Journal saved: {journal_path}")
    
    # ── Print summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"  END OF DAY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Capital: ₹{summary['capital_start']:,.0f} → ₹{summary['capital_end']:,.0f}")
    logger.info(f"  Day P&L: ₹{summary['day_pnl']:+,.2f} ({summary['day_pnl_pct']:+.2f}%)")
    logger.info(f"  Trades: {summary['total_trades']} ({summary['wins']}W / {summary['losses']}L)")
    
    if summary["total_trades"] > 0:
        win_rate = summary["wins"] / summary["total_trades"] * 100
        logger.info(f"  Win Rate: {win_rate:.0f}%")
        avg_pnl = summary["day_pnl"] / summary["total_trades"]
        logger.info(f"  Avg P&L/trade: ₹{avg_pnl:+,.2f}")
    
    logger.info(f"\n  To analyse: python main.py --mode analyse")
    logger.info(f"  Journal: {journal_path}")


if __name__ == "__main__":
    main()
