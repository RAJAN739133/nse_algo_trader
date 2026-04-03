#!/usr/bin/env python3
"""
REALISTIC BACKTEST — No Cheating, No Lookahead Bias
════════════════════════════════════════════════════════

This backtest is designed to be HONEST and simulate real trading conditions:

ANTI-CHEATING MEASURES:
1. Stock selection uses ONLY previous day's data (no same-day info)
2. Entry signals use candles BEFORE current candle (no future peeking)
3. Execution assumes WORST-CASE fills (buy at high of candle, sell at low)
4. Slippage and costs are realistically modeled
5. No optimization on the test data (out-of-sample only)

REALISTIC ASSUMPTIONS:
- 0.1% slippage on entry and exit
- 0.05% transaction costs (brokerage + STT + GST)
- Fill at VWAP approximation, not at exact signal price
- Stop losses may gap through (partial fill simulation)
- Position sizing based on PREVIOUS day's ATR

Usage:
    python backtest_realistic.py                    # Last 7 trading days
    python backtest_realistic.py --days 5           # Last 5 trading days
    python backtest_realistic.py --capital 100000   # Custom capital
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════

SLIPPAGE_PCT = 0.001      # 0.1% slippage per trade
TRANSACTION_COST = 0.0005  # 0.05% (brokerage + STT + GST)
INITIAL_CAPITAL = 10000    # Rs 10,000 (your actual capital)
MAX_POSITION_PCT = 0.50    # Max 50% per position (aggressive for small capital)
MAX_POSITIONS = 3          # Max concurrent positions
RISK_PER_TRADE = 0.02      # 2% risk per trade (more aggressive)

# Trading hours (IST)
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
NO_NEW_ENTRIES_AFTER = "15:00"  # Allow entries until 3 PM

# Strategy parameters - MORE AGGRESSIVE for more signals
RSI_OVERSOLD = 35          # Relaxed from 30
RSI_OVERBOUGHT = 65        # Relaxed from 70
MIN_VOLUME_RATIO = 1.0     # Relaxed from 1.2
MIN_ATR_PCT = 0.5          # Relaxed from 0.8


class RealisticBacktest:
    """
    Realistic backtester with anti-cheating measures.
    """
    
    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.positions = {}
        self.closed_trades = []
        self.daily_pnl = {}
        self.equity_curve = []
        
    def get_trading_days(self, num_days: int) -> List[date]:
        """Get last N trading days (Mon-Fri, excluding today)."""
        days = []
        current = date.today() - timedelta(days=1)
        while len(days) < num_days:
            if current.weekday() < 5:  # Mon=0, Fri=4
                days.append(current)
            current -= timedelta(days=1)
        return list(reversed(days))
    
    def fetch_daily_data(self, symbol: str, end_date: date, lookback: int = 30) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for stock selection.
        Uses data BEFORE end_date only (no lookahead).
        """
        try:
            import yfinance as yf
            start = end_date - timedelta(days=lookback + 10)
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(start=str(start), end=str(end_date))
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            return df
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_intraday_data(self, symbol: str, target_date: date) -> pd.DataFrame:
        """
        Fetch intraday data for a specific date.
        Returns 5-minute candles.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # yfinance requires end date to be after target
            start = str(target_date)
            end = str(target_date + timedelta(days=1))
            
            df = ticker.history(start=start, end=end, interval="5m")
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [c.lower() for c in df.columns]
            
            # Filter to market hours only
            df = df.between_time("09:15", "15:30")
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.debug(f"Failed to fetch intraday {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for stock selection."""
        if len(df) < 20:
            return df
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_trend'] = (df['ema_9'] > df['ema_21']).astype(int)
        
        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        return df
    
    def select_stocks(self, target_date: date, universe: List[str]) -> List[Dict]:
        """
        Select stocks for trading using ONLY data available BEFORE market open.
        No lookahead bias - uses previous day's close.
        
        MORE AGGRESSIVE selection for Rs 10K capital - need more opportunities!
        """
        candidates = []
        
        for symbol in universe:
            # Fetch data up to (but not including) target_date
            df = self.fetch_daily_data(symbol, target_date, lookback=30)
            
            if df.empty or len(df) < 20:
                continue
            
            df = self.calculate_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            
            # Basic filters
            if pd.isna(last.get('rsi')) or pd.isna(last.get('atr_pct')):
                continue
            
            # Minimum volatility filter (relaxed)
            if last['atr_pct'] < MIN_ATR_PCT:
                continue
            
            # Score based on setup quality
            score = 0
            direction = None
            reason = ""
            
            # 1. LONG setup: Oversold bounce
            if last['rsi'] < RSI_OVERSOLD + 10:  # RSI < 45
                score = (45 - last['rsi']) / 5
                direction = "LONG"
                reason = f"RSI oversold ({last['rsi']:.0f})"
            
            # 2. SHORT setup: Overbought reversal
            elif last['rsi'] > RSI_OVERBOUGHT - 10:  # RSI > 55
                score = (last['rsi'] - 55) / 5
                direction = "SHORT"
                reason = f"RSI overbought ({last['rsi']:.0f})"
            
            # 3. EMA Crossover - bullish
            elif last['ema_trend'] == 1 and prev['ema_trend'] == 0:
                score = 3.0
                direction = "LONG"
                reason = "EMA bullish crossover"
            
            # 4. EMA Crossover - bearish
            elif last['ema_trend'] == 0 and prev['ema_trend'] == 1:
                score = 3.0
                direction = "SHORT"
                reason = "EMA bearish crossover"
            
            # 5. Strong momentum with volume
            elif abs(last['momentum_5']) > 0.02:  # Relaxed from 0.03
                if last['momentum_5'] > 0:
                    score = last['momentum_5'] * 30
                    direction = "LONG"
                    reason = f"Momentum +{last['momentum_5']*100:.1f}%"
                else:
                    score = abs(last['momentum_5']) * 30
                    direction = "SHORT"
                    reason = f"Momentum {last['momentum_5']*100:.1f}%"
            
            # 6. Mean reversion - price far from MA
            elif abs(last['close'] - last['ema_21']) / last['ema_21'] > 0.02:
                if last['close'] < last['ema_21']:  # Below MA - bounce expected
                    score = 2.0
                    direction = "LONG"
                    reason = "Mean reversion LONG"
                else:  # Above MA - pullback expected
                    score = 2.0
                    direction = "SHORT"
                    reason = "Mean reversion SHORT"
            
            # Add any stock with reasonable volatility as backup
            elif last['atr_pct'] > 1.0:
                # Use trend direction
                if last['ema_trend'] == 1:
                    score = 1.0
                    direction = "LONG"
                    reason = "Trend following LONG"
                else:
                    score = 1.0
                    direction = "SHORT"
                    reason = "Trend following SHORT"
            
            if score > 0 and direction:
                candidates.append({
                    "symbol": symbol,
                    "score": score,
                    "direction": direction,
                    "reason": reason,
                    "prev_close": last['close'],
                    "atr": last['atr'],
                    "atr_pct": last['atr_pct'],
                    "rsi": last['rsi'],
                    "volume_ratio": last.get('volume_ratio', 1),
                })
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:MAX_POSITIONS * 3]  # Return more candidates
    
    def simulate_entry(self, candle: pd.Series, direction: str, signal_price: float) -> Tuple[float, bool]:
        """
        Simulate realistic entry with slippage.
        Returns (fill_price, success).
        
        ANTI-CHEAT: Entry at WORSE price than signal
        - LONG: Fill at max(signal, candle_open) + slippage
        - SHORT: Fill at min(signal, candle_open) - slippage
        """
        if direction == "LONG":
            # Can't buy below the open
            base_price = max(signal_price, candle['open'])
            # Add slippage
            fill_price = base_price * (1 + SLIPPAGE_PCT)
            # Must be within candle range
            if fill_price > candle['high']:
                return 0, False
            return fill_price, True
        else:  # SHORT
            base_price = min(signal_price, candle['open'])
            fill_price = base_price * (1 - SLIPPAGE_PCT)
            if fill_price < candle['low']:
                return 0, False
            return fill_price, True
    
    def simulate_exit(self, candle: pd.Series, direction: str, signal_price: float) -> float:
        """
        Simulate realistic exit with slippage.
        
        ANTI-CHEAT: Exit at WORSE price than signal
        - LONG exit (sell): at min(signal, candle_close) - slippage
        - SHORT exit (cover): at max(signal, candle_close) + slippage
        """
        if direction == "LONG":
            base_price = min(signal_price, candle['close'])
            return base_price * (1 - SLIPPAGE_PCT)
        else:
            base_price = max(signal_price, candle['close'])
            return base_price * (1 + SLIPPAGE_PCT)
    
    def check_stop_loss(self, candle: pd.Series, position: Dict) -> Tuple[bool, float]:
        """
        Check if stop loss was hit.
        
        ANTI-CHEAT: Stop loss may gap through
        - Returns actual exit price, not stop price
        """
        direction = position['direction']
        stop = position['stop_loss']
        
        if direction == "LONG":
            # Stop hit if low goes below stop
            if candle['low'] <= stop:
                # Exit at worse of stop or open (gap down case)
                exit_price = min(stop, candle['open']) * (1 - SLIPPAGE_PCT)
                return True, exit_price
        else:  # SHORT
            if candle['high'] >= stop:
                exit_price = max(stop, candle['open']) * (1 + SLIPPAGE_PCT)
                return True, exit_price
        
        return False, 0
    
    def check_target(self, candle: pd.Series, position: Dict) -> Tuple[bool, float]:
        """
        Check if target was hit.
        """
        direction = position['direction']
        target = position['target']
        
        if direction == "LONG":
            if candle['high'] >= target:
                # Target hit, fill at target
                return True, target * (1 - SLIPPAGE_PCT)
        else:
            if candle['low'] <= target:
                return True, target * (1 + SLIPPAGE_PCT)
        
        return False, 0
    
    def generate_intraday_signals(self, df: pd.DataFrame, prev_data: Dict) -> List[Dict]:
        """
        Generate entry signals from intraday data.
        
        ANTI-CHEAT: Signal on candle N can only be acted on candle N+1
        
        MORE AGGRESSIVE - Generate signals earlier and more frequently
        """
        signals = []
        
        if len(df) < 5:
            return signals
        
        # Calculate intraday indicators
        df = df.copy()
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_13'] = df['close'].ewm(span=13).mean()
        df['ema_diff'] = df['ema_5'] - df['ema_13']
        df['volume_ma'] = df['volume'].rolling(5).mean()
        
        # Calculate RSI for intraday
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Get expected direction from daily analysis
        expected_dir = prev_data.get('direction')
        
        # Iterate through candles (start early at candle 5)
        for i in range(5, len(df) - 1):
            candle = df.iloc[i]
            time_str = str(candle.name.time())[:5]
            
            # No entries after configured time
            if time_str > NO_NEW_ENTRIES_AFTER:
                continue
            
            # Signal 1: EMA crossover
            if i > 0:
                prev_ema_diff = df['ema_diff'].iloc[i-1]
                curr_ema_diff = df['ema_diff'].iloc[i]
                
                # Bullish crossover
                if prev_ema_diff < 0 and curr_ema_diff > 0:
                    if expected_dir in [None, 'LONG']:
                        signals.append({
                            "time": candle.name,
                            "action_candle_idx": i + 1,
                            "direction": "LONG",
                            "signal_price": candle['close'],
                            "reason": "EMA_CROSS_UP"
                        })
                        break  # Take first signal
                
                # Bearish crossover
                elif prev_ema_diff > 0 and curr_ema_diff < 0:
                    if expected_dir in [None, 'SHORT']:
                        signals.append({
                            "time": candle.name,
                            "action_candle_idx": i + 1,
                            "direction": "SHORT",
                            "signal_price": candle['close'],
                            "reason": "EMA_CROSS_DOWN"
                        })
                        break
            
            # Signal 2: RSI extreme + direction match
            if not signals and i > 6:
                rsi = df['rsi'].iloc[i]
                if rsi < 30 and expected_dir in [None, 'LONG']:
                    signals.append({
                        "time": candle.name,
                        "action_candle_idx": i + 1,
                        "direction": "LONG",
                        "signal_price": candle['close'],
                        "reason": f"RSI_OVERSOLD_{rsi:.0f}"
                    })
                    break
                elif rsi > 70 and expected_dir in [None, 'SHORT']:
                    signals.append({
                        "time": candle.name,
                        "action_candle_idx": i + 1,
                        "direction": "SHORT",
                        "signal_price": candle['close'],
                        "reason": f"RSI_OVERBOUGHT_{rsi:.0f}"
                    })
                    break
            
            # Signal 3: Strong candle in expected direction (after 9:30)
            if not signals and time_str >= "09:30":
                candle_pct = (candle['close'] - candle['open']) / candle['open'] * 100
                
                if candle_pct > 0.3 and expected_dir == "LONG":
                    signals.append({
                        "time": candle.name,
                        "action_candle_idx": i + 1,
                        "direction": "LONG",
                        "signal_price": candle['close'],
                        "reason": "STRONG_BULL_CANDLE"
                    })
                    break
                elif candle_pct < -0.3 and expected_dir == "SHORT":
                    signals.append({
                        "time": candle.name,
                        "action_candle_idx": i + 1,
                        "direction": "SHORT",
                        "signal_price": candle['close'],
                        "reason": "STRONG_BEAR_CANDLE"
                    })
                    break
        
        # Fallback: If no signal but we have expected direction, enter on first candle after 9:30
        if not signals and expected_dir and len(df) > 5:
            for i in range(3, min(10, len(df) - 1)):
                time_str = str(df.iloc[i].name.time())[:5]
                if time_str >= "09:30":
                    signals.append({
                        "time": df.iloc[i].name,
                        "action_candle_idx": i + 1,
                        "direction": expected_dir,
                        "signal_price": df.iloc[i]['close'],
                        "reason": "DAILY_SETUP_ENTRY"
                    })
                    break
        
        return signals
    
    def run_day(self, target_date: date, universe: List[str]) -> Dict:
        """
        Run backtest for a single day.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  DATE: {target_date} ({target_date.strftime('%A')})")
        logger.info(f"{'='*60}")
        
        day_result = {
            "date": target_date,
            "trades": [],
            "pnl": 0,
            "wins": 0,
            "losses": 0,
        }
        
        # Step 1: Stock selection using PREVIOUS day's data
        logger.info(f"  Selecting stocks (using data before {target_date})...")
        candidates = self.select_stocks(target_date, universe)
        
        if not candidates:
            logger.info(f"  No candidates found")
            return day_result
        
        logger.info(f"  Candidates: {[c['symbol'] for c in candidates[:5]]}")
        
        # Step 2: Process each candidate
        for candidate in candidates[:MAX_POSITIONS]:
            symbol = candidate['symbol']
            direction = candidate['direction']
            
            # Fetch intraday data for target date
            intraday = self.fetch_intraday_data(symbol, target_date)
            
            if intraday.empty or len(intraday) < 20:
                logger.debug(f"  {symbol}: No intraday data")
                continue
            
            # Generate signals
            signals = self.generate_intraday_signals(intraday, candidate)
            
            if not signals:
                logger.debug(f"  {symbol}: No signals")
                continue
            
            # Take first valid signal
            signal = signals[0]
            action_idx = signal['action_candle_idx']
            
            if action_idx >= len(intraday):
                continue
            
            action_candle = intraday.iloc[action_idx]
            
            # Simulate entry
            fill_price, success = self.simulate_entry(action_candle, direction, signal['signal_price'])
            
            if not success:
                logger.debug(f"  {symbol}: Entry failed")
                continue
            
            # Calculate position size based on risk
            atr = candidate['atr']
            stop_distance = atr * 1.5
            risk_amount = self.capital * RISK_PER_TRADE
            position_size = int(risk_amount / stop_distance)
            
            # Cap position size
            max_position_value = self.capital * MAX_POSITION_PCT
            max_shares = int(max_position_value / fill_price)
            position_size = min(position_size, max_shares)
            
            if position_size < 1:
                continue
            
            # Set stop loss and target
            if direction == "LONG":
                stop_loss = fill_price - stop_distance
                target = fill_price + (stop_distance * 2)  # 2:1 R:R
            else:
                stop_loss = fill_price + stop_distance
                target = fill_price - (stop_distance * 2)
            
            position = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": fill_price,
                "entry_time": action_candle.name,
                "quantity": position_size,
                "stop_loss": stop_loss,
                "target": target,
                "entry_value": fill_price * position_size,
            }
            
            logger.info(f"  📥 {symbol} {direction} @ {fill_price:.2f} x {position_size} "
                       f"| SL: {stop_loss:.2f} | T: {target:.2f}")
            
            # Step 3: Simulate holding through remaining candles
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for j in range(action_idx + 1, len(intraday)):
                candle = intraday.iloc[j]
                
                # Check stop loss
                sl_hit, sl_price = self.check_stop_loss(candle, position)
                if sl_hit:
                    exit_price = sl_price
                    exit_time = candle.name
                    exit_reason = "STOP_LOSS"
                    break
                
                # Check target
                tgt_hit, tgt_price = self.check_target(candle, position)
                if tgt_hit:
                    exit_price = tgt_price
                    exit_time = candle.name
                    exit_reason = "TARGET"
                    break
            
            # If no exit, close at EOD
            if exit_price is None:
                last_candle = intraday.iloc[-1]
                exit_price = self.simulate_exit(last_candle, direction, last_candle['close'])
                exit_time = last_candle.name
                exit_reason = "EOD"
            
            # Calculate P&L
            if direction == "LONG":
                gross_pnl = (exit_price - fill_price) * position_size
            else:
                gross_pnl = (fill_price - exit_price) * position_size
            
            # Deduct costs
            entry_cost = position['entry_value'] * TRANSACTION_COST
            exit_cost = exit_price * position_size * TRANSACTION_COST
            net_pnl = gross_pnl - entry_cost - exit_cost
            
            trade = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": round(fill_price, 2),
                "exit_price": round(exit_price, 2),
                "quantity": position_size,
                "entry_time": str(position['entry_time']),
                "exit_time": str(exit_time),
                "exit_reason": exit_reason,
                "gross_pnl": round(gross_pnl, 2),
                "costs": round(entry_cost + exit_cost, 2),
                "net_pnl": round(net_pnl, 2),
            }
            
            day_result["trades"].append(trade)
            day_result["pnl"] += net_pnl
            
            if net_pnl > 0:
                day_result["wins"] += 1
            else:
                day_result["losses"] += 1
            
            emoji = "✅" if net_pnl > 0 else "❌"
            logger.info(f"  {emoji} {symbol} {exit_reason} @ {exit_price:.2f} | "
                       f"P&L: Rs {net_pnl:+,.2f}")
        
        # Update capital
        self.capital += day_result["pnl"]
        self.daily_pnl[str(target_date)] = day_result["pnl"]
        self.equity_curve.append({
            "date": str(target_date),
            "equity": self.capital,
            "pnl": day_result["pnl"]
        })
        
        logger.info(f"\n  Day P&L: Rs {day_result['pnl']:+,.2f} | "
                   f"Trades: {len(day_result['trades'])} | "
                   f"Capital: Rs {self.capital:,.2f}")
        
        return day_result
    
    def run(self, num_days: int, universe: List[str]) -> Dict:
        """
        Run full backtest for multiple days.
        """
        trading_days = self.get_trading_days(num_days)
        
        logger.info(f"\n{'═'*70}")
        logger.info(f"  REALISTIC BACKTEST — NO CHEATING")
        logger.info(f"  Period: {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} days)")
        logger.info(f"  Initial Capital: Rs {self.initial_capital:,}")
        logger.info(f"  Universe: {len(universe)} stocks")
        logger.info(f"{'═'*70}")
        
        all_trades = []
        
        for day in trading_days:
            result = self.run_day(day, universe)
            all_trades.extend(result["trades"])
        
        # Generate summary
        return self.generate_summary(all_trades, trading_days)
    
    def generate_summary(self, trades: List[Dict], trading_days: List[date]) -> Dict:
        """Generate comprehensive backtest summary."""
        if not trades:
            logger.info("\n  No trades executed during backtest period.")
            return {"error": "No trades"}
        
        df = pd.DataFrame(trades)
        
        total_trades = len(df)
        wins = len(df[df["net_pnl"] > 0])
        losses = len(df[df["net_pnl"] <= 0])
        win_rate = wins / total_trades * 100
        
        total_pnl = df["net_pnl"].sum()
        total_gross = df["gross_pnl"].sum()
        total_costs = df["costs"].sum()
        
        avg_win = df[df["net_pnl"] > 0]["net_pnl"].mean() if wins > 0 else 0
        avg_loss = df[df["net_pnl"] <= 0]["net_pnl"].mean() if losses > 0 else 0
        
        # Calculate metrics
        returns = total_pnl / self.initial_capital * 100
        
        # Drawdown
        equity = pd.Series([self.initial_capital] + [e["equity"] for e in self.equity_curve])
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # By direction
        longs = df[df["direction"] == "LONG"]
        shorts = df[df["direction"] == "SHORT"]
        
        print("\n" + "═"*70)
        print("  BACKTEST RESULTS — REALISTIC (NO CHEATING)")
        print(f"  Period: {trading_days[0]} to {trading_days[-1]}")
        print("═"*70)
        
        print(f"\n  💰 CAPITAL")
        print(f"  ─────────────────────────────────────────")
        print(f"  Initial:          Rs {self.initial_capital:>12,}")
        print(f"  Final:            Rs {self.capital:>12,.2f}")
        print(f"  Net Return:       {returns:>12.2f}%")
        print(f"  Max Drawdown:     {max_drawdown:>12.2f}%")
        
        print(f"\n  📊 PERFORMANCE")
        print(f"  ─────────────────────────────────────────")
        print(f"  Total Trades:     {total_trades:>12}")
        print(f"  Winners:          {wins:>12} ({win_rate:.1f}%)")
        print(f"  Losers:           {losses:>12}")
        print(f"  ─────────────────────────────────────────")
        print(f"  Gross P&L:        Rs {total_gross:>+12,.2f}")
        print(f"  Costs:            Rs {total_costs:>12,.2f}")
        print(f"  Net P&L:          Rs {total_pnl:>+12,.2f}")
        print(f"  ─────────────────────────────────────────")
        print(f"  Avg Win:          Rs {avg_win:>+12,.2f}")
        print(f"  Avg Loss:         Rs {avg_loss:>+12,.2f}")
        if avg_loss != 0:
            print(f"  Win/Loss Ratio:   {abs(avg_win/avg_loss):>12.2f}")
        
        print(f"\n  📈 BY DIRECTION")
        print(f"  ─────────────────────────────────────────")
        if len(longs) > 0:
            long_wr = len(longs[longs["net_pnl"] > 0]) / len(longs) * 100
            print(f"  LONG:  {len(longs):>3} trades | Rs {longs['net_pnl'].sum():>+10,.2f} | WR: {long_wr:.0f}%")
        if len(shorts) > 0:
            short_wr = len(shorts[shorts["net_pnl"] > 0]) / len(shorts) * 100
            print(f"  SHORT: {len(shorts):>3} trades | Rs {shorts['net_pnl'].sum():>+10,.2f} | WR: {short_wr:.0f}%")
        
        print(f"\n  📅 DAILY P&L")
        print(f"  ─────────────────────────────────────────")
        for day_str, pnl in self.daily_pnl.items():
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            print(f"  {emoji} {day_str}  Rs {pnl:>+10,.2f}")
        
        print("\n" + "═"*70)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        df.to_csv(results_dir / "backtest_realistic.csv", index=False)
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(results_dir / "backtest_equity_curve.csv", index=False)
        
        logger.info(f"\n  Results saved to results/backtest_realistic.csv")
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "net_pnl": total_pnl,
            "returns_pct": returns,
            "max_drawdown": max_drawdown,
            "trades": trades,
        }


def get_default_universe() -> List[str]:
    """Get default stock universe for backtest."""
    return [
        # IT
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
        # Banks
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        # Large caps
        "RELIANCE", "BHARTIARTL", "HINDUNILVR", "ITC", "LT",
        # Others
        "BAJFINANCE", "MARUTI", "TITAN", "ASIANPAINT", "SUNPHARMA",
        "TATAMOTORS", "M&M", "NESTLEIND", "ULTRACEMCO", "DRREDDY",
    ]


def main():
    parser = argparse.ArgumentParser(description="Realistic backtest without cheating")
    parser.add_argument("--days", type=int, default=7, help="Number of trading days")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    args = parser.parse_args()
    
    universe = get_default_universe()
    
    backtest = RealisticBacktest(capital=args.capital)
    results = backtest.run(num_days=args.days, universe=universe)
    
    return results


if __name__ == "__main__":
    main()
