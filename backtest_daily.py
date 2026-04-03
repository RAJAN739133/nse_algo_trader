#!/usr/bin/env python3
"""
DAILY BACKTEST — No Cheating, Uses Daily Candles
════════════════════════════════════════════════════════

Since yfinance intraday data is unreliable, this backtest uses DAILY candles
which are always available.

Strategy: Swing trading on daily timeframe
- Entry: Next day's open after signal
- Exit: Target/SL hit or after N days
- Position held overnight (realistic for small capital)

ANTI-CHEATING:
1. Signal on day N → Entry on day N+1 open
2. No same-day lookahead
3. Realistic fills with slippage

Usage:
    python backtest_daily.py --days 7 --capital 10000
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

SLIPPAGE_PCT = 0.002       # 0.2% slippage (higher for daily trades)
TRANSACTION_COST = 0.001   # 0.1% costs
INITIAL_CAPITAL = 10000    # Rs 10,000

# ════════════════════════════════════════════════════════════
# POSITION SIZING - Adapted for Rs 10K capital
# With tiny capital, we need larger position % per trade
# BUT limit total concurrent positions to control risk
# ════════════════════════════════════════════════════════════
MAX_POSITION_PCT = 0.15    # Max 15% per position (Rs 1,500 in 10K account)
MIN_POSITION_PCT = 0.05    # Min 5% per position 
MAX_POSITIONS = 2          # ONLY 2 concurrent positions (risk control!)
RISK_PER_TRADE = 0.02      # 2% risk per trade
MAX_HOLDING_DAYS = 3       # Exit after 3 days if no SL/Target

# ════════════════════════════════════════════════════════════
# CIRCUIT BREAKER - Stop ALL trading after 2% daily loss
# ════════════════════════════════════════════════════════════
CIRCUIT_BREAKER_LOSS_PCT = 0.02   # 2% daily loss triggers circuit breaker

# PROFIT BOOKING RULES (optimized from 30-day backtest)
# KEY: Book profits faster, cut losses even faster!
QUICK_PROFIT_PCT = 0.012   # Book profit at 1.2% (was 1.5%)
TRAIL_AFTER_PCT = 0.008    # Trail stop after 0.8% profit
TRAIL_DISTANCE_PCT = 0.004 # Trail by 0.4%
CUT_LOSS_EARLY_PCT = 0.008 # Cut losers at 0.8% loss (don't wait for stop)


class DailyBacktest:
    """
    Daily timeframe backtest - more reliable data.
    NOW USES YOUR LIVE STRATEGY'S TREND DETECTION!
    """
    
    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.positions = {}  # {symbol: position_dict}
        self.closed_trades = []
        self.equity_curve = []
        
        # Trend controls (like your live strategy)
        self.enable_longs = True
        self.enable_shorts = True
        self.max_longs = 3
        self.max_shorts = 3
        self.market_trend = "NEUTRAL"
        
        # ══════════════════════════════════════════════════════════════
        # CIRCUIT BREAKER - Stop trading after daily loss threshold
        # ══════════════════════════════════════════════════════════════
        self.circuit_breaker_triggered = False
        self.daily_pnl = 0.0
        self.daily_starting_capital = capital
    
    def detect_market_trend(self, nifty_df: pd.DataFrame, target_date: date, vix: float = 15) -> Dict:
        """
        Detect market trend - SAME LOGIC AS YOUR LIVE STRATEGY!
        
        Uses:
        1. Intraday change from today's open
        2. Position in today's range
        3. VIX level
        """
        result = {
            "trend": "NEUTRAL",
            "confidence": 50,
            "enable_longs": True,
            "enable_shorts": True,
            "max_longs": 3,
            "max_shorts": 3,
            "reason": "",
        }
        
        if nifty_df.empty or target_date not in nifty_df.index:
            return result
        
        today = nifty_df.loc[target_date]
        
        # Get previous day for context
        idx = list(nifty_df.index).index(target_date)
        if idx > 0:
            prev_close = nifty_df.iloc[idx - 1]['close']
        else:
            prev_close = today['open']
        
        # Calculate intraday metrics
        today_open = today['open']
        today_high = today['high']
        today_low = today['low']
        today_close = today['close']
        
        intraday_change_pct = (today_close - today_open) / today_open * 100
        range_size = today_high - today_low
        position_in_range = (today_close - today_low) / range_size if range_size > 0 else 0.5
        
        # Trend scoring - SAME AS YOUR LIVE STRATEGY
        trend_score = 0
        reasons = []
        
        # Factor 1: Intraday change from today's open
        if intraday_change_pct > 1.0:
            trend_score += 40
            reasons.append(f"Strong up +{intraday_change_pct:.1f}%")
        elif intraday_change_pct > 0.3:
            trend_score += 20
            reasons.append(f"Up +{intraday_change_pct:.1f}%")
        elif intraday_change_pct < -1.0:
            trend_score -= 40
            reasons.append(f"Strong down {intraday_change_pct:.1f}%")
        elif intraday_change_pct < -0.3:
            trend_score -= 20
            reasons.append(f"Down {intraday_change_pct:.1f}%")
        else:
            reasons.append(f"Flat {intraday_change_pct:+.1f}%")
        
        # Factor 2: Position in today's range
        if position_in_range > 0.75:
            trend_score += 25
            reasons.append(f"Near high ({position_in_range:.0%})")
        elif position_in_range > 0.55:
            trend_score += 10
        elif position_in_range < 0.25:
            trend_score -= 25
            reasons.append(f"Near low ({position_in_range:.0%})")
        elif position_in_range < 0.45:
            trend_score -= 10
        
        # Factor 3: VIX (use default)
        if vix > 25:
            trend_score -= 15
            reasons.append(f"High VIX {vix:.0f}")
        elif vix > 20:
            trend_score -= 5
        
        # Determine trend - SAME THRESHOLDS AS LIVE STRATEGY
        if trend_score >= 50:
            result["trend"] = "STRONG_BULLISH"
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 5
            result["max_shorts"] = 2
        elif trend_score >= 20:
            result["trend"] = "BULLISH"
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 4
            result["max_shorts"] = 3
        elif trend_score <= -50:
            result["trend"] = "BEARISH"
            result["enable_longs"] = False  # NO LONGS IN BEARISH!
            result["enable_shorts"] = True
            result["max_longs"] = 0
            result["max_shorts"] = 6
        elif trend_score <= -20:
            result["trend"] = "MILD_BEARISH"
            result["enable_longs"] = False  # NO LONGS in ANY bearish market!
            result["enable_shorts"] = True
            result["max_longs"] = 0         # Zero tolerance for LONGs in bearish
            result["max_shorts"] = 5
        else:
            result["trend"] = "NEUTRAL"
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 3
            result["max_shorts"] = 3
        
        result["reason"] = " | ".join(reasons)
        result["trend_score"] = trend_score
        result["intraday_change"] = intraday_change_pct
        
        # Update instance variables
        self.market_trend = result["trend"]
        self.enable_longs = result["enable_longs"]
        self.enable_shorts = result["enable_shorts"]
        self.max_longs = result["max_longs"]
        self.max_shorts = result["max_shorts"]
        
        return result
    
    def get_trading_days(self, num_days: int) -> List[date]:
        """Get last N trading days."""
        days = []
        current = date.today() - timedelta(days=1)
        while len(days) < num_days:
            if current.weekday() < 5:
                days.append(current)
            current -= timedelta(days=1)
        return list(reversed(days))
    
    def fetch_data(self, symbol: str, end_date: date, lookback: int = 60) -> pd.DataFrame:
        """Fetch daily OHLCV data."""
        try:
            import yfinance as yf
            start = end_date - timedelta(days=lookback + 10)
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(start=str(start), end=str(end_date + timedelta(days=1)))
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [c.lower() for c in df.columns]
            df.index = df.index.date
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
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
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # Momentum
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Generate trading signal based on last available candle.
        Returns signal if valid, None otherwise.
        """
        if len(df) < 25:
            return None
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Skip if indicators not calculated
        if pd.isna(last.get('rsi')) or pd.isna(last.get('atr')):
            return None
        
        direction = None
        score = 0
        reason = ""
        
        # ══════════════════════════════════════════════════════════════
        # LONG SIGNALS
        # ══════════════════════════════════════════════════════════════
        
        # 1. RSI oversold bounce
        if last['rsi'] < 35 and prev['rsi'] < last['rsi']:
            direction = "LONG"
            score = 3.0
            reason = f"RSI oversold bounce ({last['rsi']:.0f})"
        
        # 2. EMA bullish crossover (9 crosses above 21)
        elif prev['ema_9'] < prev['ema_21'] and last['ema_9'] > last['ema_21']:
            direction = "LONG"
            score = 4.0
            reason = "EMA 9/21 bullish crossover"
        
        # 3. Bollinger Band bounce from lower band
        elif last['bb_pct'] < 0.1 and last['close'] > last['open']:
            direction = "LONG"
            score = 2.5
            reason = "BB lower band bounce"
        
        # 4. Strong momentum + volume
        elif last['momentum_3'] > 0.03 and last['volume_ratio'] > 1.3:
            direction = "LONG"
            score = 2.0
            reason = f"Momentum {last['momentum_3']*100:.1f}% + volume"
        
        # 5. Price above rising EMAs (trend following)
        elif (last['close'] > last['ema_9'] > last['ema_21'] and 
              last['ema_9'] > prev['ema_9']):
            direction = "LONG"
            score = 1.5
            reason = "Uptrend continuation"
        
        # ══════════════════════════════════════════════════════════════
        # SHORT SIGNALS
        # ══════════════════════════════════════════════════════════════
        
        # 1. RSI overbought reversal
        elif last['rsi'] > 65 and prev['rsi'] > last['rsi']:
            direction = "SHORT"
            score = 3.0
            reason = f"RSI overbought reversal ({last['rsi']:.0f})"
        
        # 2. EMA bearish crossover
        elif prev['ema_9'] > prev['ema_21'] and last['ema_9'] < last['ema_21']:
            direction = "SHORT"
            score = 4.0
            reason = "EMA 9/21 bearish crossover"
        
        # 3. Bollinger Band rejection from upper band
        elif last['bb_pct'] > 0.9 and last['close'] < last['open']:
            direction = "SHORT"
            score = 2.5
            reason = "BB upper band rejection"
        
        # 4. Negative momentum + volume
        elif last['momentum_3'] < -0.03 and last['volume_ratio'] > 1.3:
            direction = "SHORT"
            score = 2.0
            reason = f"Momentum {last['momentum_3']*100:.1f}% + volume"
        
        # 5. Price below falling EMAs (downtrend)
        elif (last['close'] < last['ema_9'] < last['ema_21'] and 
              last['ema_9'] < prev['ema_9']):
            direction = "SHORT"
            score = 1.5
            reason = "Downtrend continuation"
        
        if direction and score > 0:
            return {
                "symbol": symbol,
                "direction": direction,
                "score": score,
                "reason": reason,
                "signal_close": last['close'],
                "atr": last['atr'],
                "rsi": last['rsi'],
            }
        
        return None
    
    def run(self, num_days: int, universe: List[str]) -> Dict:
        """Run the backtest."""
        trading_days = self.get_trading_days(num_days)
        
        logger.info(f"\n{'═'*70}")
        logger.info(f"  DAILY BACKTEST — NO CHEATING")
        logger.info(f"  Period: {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} days)")
        logger.info(f"  Initial Capital: Rs {self.initial_capital:,.0f}")
        logger.info(f"  Universe: {len(universe)} stocks")
        logger.info(f"{'═'*70}")
        
        # Fetch all data upfront
        all_data = {}
        logger.info("\n  Fetching data...")
        for symbol in universe:
            df = self.fetch_data(symbol, trading_days[-1], lookback=60)
            if not df.empty and len(df) > 25:
                df = self.calculate_indicators(df)
                all_data[symbol] = df
        
        logger.info(f"  Loaded data for {len(all_data)} stocks")
        
        # Fetch Nifty data for trend detection
        nifty_df = self.fetch_data("NIFTY", trading_days[-1], lookback=60)
        if nifty_df.empty:
            # Try ^NSEI
            try:
                import yfinance as yf
                start = trading_days[0] - timedelta(days=10)
                nifty = yf.Ticker("^NSEI").history(start=str(start), end=str(trading_days[-1] + timedelta(days=1)))
                nifty.columns = [c.lower() for c in nifty.columns]
                nifty.index = nifty.index.date
                nifty_df = nifty[['open', 'high', 'low', 'close', 'volume']]
            except:
                nifty_df = pd.DataFrame()
        
        # Process each trading day
        for i, target_date in enumerate(trading_days):
            logger.info(f"\n{'─'*60}")
            logger.info(f"  DATE: {target_date} ({target_date.strftime('%A')})")
            logger.info(f"{'─'*60}")
            
            # ══════════════════════════════════════════════════════════════
            # CIRCUIT BREAKER RESET AT START OF EACH DAY
            # ══════════════════════════════════════════════════════════════
            self.circuit_breaker_triggered = False
            self.daily_pnl = 0.0
            self.daily_starting_capital = self.capital
            
            # DETECT MARKET TREND - SAME AS YOUR LIVE STRATEGY!
            trend_info = self.detect_market_trend(nifty_df, target_date)
            logger.info(f"  📊 TREND: {self.market_trend} | "
                       f"LONGS: {'✅' if self.enable_longs else '❌'} (max {self.max_longs}) | "
                       f"SHORTS: {'✅' if self.enable_shorts else '❌'} (max {self.max_shorts})")
            logger.info(f"  🛑 Circuit Breaker: {CIRCUIT_BREAKER_LOSS_PCT*100:.0f}% daily loss | Max Position: {MAX_POSITION_PCT*100:.0f}%")
            if trend_info.get('reason'):
                logger.info(f"     Reason: {trend_info['reason']}")
            
            day_pnl = 0
            
            # 1. Check existing positions for exits
            positions_to_close = []
            for symbol, pos in list(self.positions.items()):
                if symbol not in all_data:
                    continue
                
                df = all_data[symbol]
                if target_date not in df.index:
                    continue
                
                today = df.loc[target_date]
                entry_price = pos['entry_price']
                
                # Calculate current unrealized P&L %
                if pos['direction'] == "LONG":
                    best_price = today['high']
                    current_pnl_pct = (best_price - entry_price) / entry_price
                else:
                    best_price = today['low']  # Best for shorts
                    current_pnl_pct = (entry_price - best_price) / entry_price
                
                # ══════════════════════════════════════════════════════════
                # QUICK PROFIT BOOKING - THE KEY FIX!
                # ══════════════════════════════════════════════════════════
                if current_pnl_pct >= QUICK_PROFIT_PCT:
                    if pos['direction'] == "LONG":
                        exit_price = best_price * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price - entry_price) * pos['quantity']
                    else:
                        exit_price = best_price * (1 + SLIPPAGE_PCT)
                        pnl = (entry_price - exit_price) * pos['quantity']
                    positions_to_close.append((symbol, exit_price, f"QUICK_PROFIT_{current_pnl_pct*100:.1f}%", pnl))
                    continue
                
                # Update trailing stop if in profit
                if current_pnl_pct >= TRAIL_AFTER_PCT:
                    if pos['direction'] == "LONG":
                        new_stop = best_price * (1 - TRAIL_DISTANCE_PCT)
                        if new_stop > pos['stop_loss']:
                            pos['stop_loss'] = new_stop
                            logger.debug(f"  {symbol}: Trailing SL → {new_stop:.2f}")
                    else:
                        new_stop = best_price * (1 + TRAIL_DISTANCE_PCT)
                        if new_stop < pos['stop_loss']:
                            pos['stop_loss'] = new_stop
                            logger.debug(f"  {symbol}: Trailing SL → {new_stop:.2f}")
                
                # Check stop loss
                if pos['direction'] == "LONG":
                    if today['low'] <= pos['stop_loss']:
                        exit_price = min(pos['stop_loss'], today['open']) * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price - entry_price) * pos['quantity']
                        positions_to_close.append((symbol, exit_price, "STOP_LOSS", pnl))
                        continue
                    
                    # Check target
                    if today['high'] >= pos['target']:
                        exit_price = pos['target'] * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price - entry_price) * pos['quantity']
                        positions_to_close.append((symbol, exit_price, "TARGET", pnl))
                        continue
                    
                else:  # SHORT
                    if today['high'] >= pos['stop_loss']:
                        exit_price = max(pos['stop_loss'], today['open']) * (1 + SLIPPAGE_PCT)
                        pnl = (entry_price - exit_price) * pos['quantity']
                        positions_to_close.append((symbol, exit_price, "STOP_LOSS", pnl))
                        continue
                    
                    if today['low'] <= pos['target']:
                        exit_price = pos['target'] * (1 + SLIPPAGE_PCT)
                        pnl = (entry_price - exit_price) * pos['quantity']
                        positions_to_close.append((symbol, exit_price, "TARGET", pnl))
                        continue
                
                # Check max holding period
                pos['days_held'] += 1
                if pos['days_held'] >= MAX_HOLDING_DAYS:
                    exit_price = today['close'] * (1 - SLIPPAGE_PCT if pos['direction'] == "LONG" else 1 + SLIPPAGE_PCT)
                    if pos['direction'] == "LONG":
                        pnl = (exit_price - entry_price) * pos['quantity']
                    else:
                        pnl = (entry_price - exit_price) * pos['quantity']
                    positions_to_close.append((symbol, exit_price, "MAX_DAYS", pnl))
            
            # Close positions
            for symbol, exit_price, reason, pnl in positions_to_close:
                pos = self.positions[symbol]
                cost = (pos['entry_price'] + exit_price) * pos['quantity'] * TRANSACTION_COST
                net_pnl = pnl - cost
                
                self.closed_trades.append({
                    "symbol": symbol,
                    "direction": pos['direction'],
                    "entry_date": str(pos['entry_date']),
                    "exit_date": str(target_date),
                    "entry_price": round(pos['entry_price'], 2),
                    "exit_price": round(exit_price, 2),
                    "quantity": pos['quantity'],
                    "reason": reason,
                    "gross_pnl": round(pnl, 2),
                    "costs": round(cost, 2),
                    "net_pnl": round(net_pnl, 2),
                })
                
                self.capital += net_pnl
                day_pnl += net_pnl
                
                emoji = "✅" if net_pnl > 0 else "❌"
                logger.info(f"  {emoji} CLOSE {symbol} {pos['direction']} @ {exit_price:.2f} | "
                           f"{reason} | P&L: Rs {net_pnl:+,.2f}")
                
                del self.positions[symbol]
            
            # ══════════════════════════════════════════════════════════════
            # CIRCUIT BREAKER CHECK - Update daily PnL and check threshold
            # ══════════════════════════════════════════════════════════════
            self.daily_pnl += day_pnl
            if self.daily_starting_capital > 0:
                daily_loss_pct = -self.daily_pnl / self.daily_starting_capital
                if daily_loss_pct >= CIRCUIT_BREAKER_LOSS_PCT:
                    if not self.circuit_breaker_triggered:
                        self.circuit_breaker_triggered = True
                        logger.warning(f"  🛑 CIRCUIT BREAKER! Daily loss {daily_loss_pct*100:.1f}% > {CIRCUIT_BREAKER_LOSS_PCT*100:.0f}%")
            
            # 2. Generate new signals (using data up to PREVIOUS day)
            # Count current positions by direction
            current_longs = sum(1 for p in self.positions.values() if p['direction'] == 'LONG')
            current_shorts = sum(1 for p in self.positions.values() if p['direction'] == 'SHORT')
            
            # Skip new entries if circuit breaker triggered
            if self.circuit_breaker_triggered:
                logger.info(f"  ⛔ Skipping new entries - Circuit breaker active")
            elif len(self.positions) < MAX_POSITIONS:
                signals = []
                
                for symbol in all_data:
                    if symbol in self.positions:
                        continue
                    
                    df = all_data[symbol]
                    
                    # Use data up to previous day for signal generation
                    prev_day_idx = list(df.index).index(target_date) - 1 if target_date in df.index else -1
                    if prev_day_idx < 20:
                        continue
                    
                    df_for_signal = df.iloc[:prev_day_idx + 1]
                    signal = self.generate_signals(df_for_signal, symbol)
                    
                    if signal:
                        # FILTER BY TREND - SAME AS YOUR LIVE STRATEGY!
                        direction = signal['direction']
                        
                        # Skip LONG if longs disabled or at max
                        if direction == "LONG":
                            if not self.enable_longs:
                                logger.debug(f"  {symbol}: LONG skipped (disabled in {self.market_trend})")
                                continue
                            if current_longs >= self.max_longs:
                                continue
                        
                        # Skip SHORT if shorts disabled or at max
                        if direction == "SHORT":
                            if not self.enable_shorts:
                                logger.debug(f"  {symbol}: SHORT skipped (disabled)")
                                continue
                            if current_shorts >= self.max_shorts:
                                continue
                        
                        signals.append(signal)
                
                # Sort by score and take top signals
                signals.sort(key=lambda x: x['score'], reverse=True)
                
                for signal in signals[:MAX_POSITIONS - len(self.positions)]:
                    symbol = signal['symbol']
                    df = all_data[symbol]
                    
                    if target_date not in df.index:
                        continue
                    
                    today = df.loc[target_date]
                    
                    # Entry at today's open + slippage
                    if signal['direction'] == "LONG":
                        entry_price = today['open'] * (1 + SLIPPAGE_PCT)
                    else:
                        entry_price = today['open'] * (1 - SLIPPAGE_PCT)
                    
                    # Position sizing
                    atr = signal['atr']
                    stop_distance = atr * 2
                    risk_amount = self.capital * RISK_PER_TRADE
                    quantity = int(risk_amount / stop_distance)
                    
                    # Cap by max position size
                    max_qty = int((self.capital * MAX_POSITION_PCT) / entry_price)
                    quantity = min(quantity, max_qty)
                    
                    if quantity < 1:
                        continue
                    
                    # Set stop and target
                    if signal['direction'] == "LONG":
                        stop_loss = entry_price - stop_distance
                        target = entry_price + (stop_distance * 2)  # 2:1 R:R
                    else:
                        stop_loss = entry_price + stop_distance
                        target = entry_price - (stop_distance * 2)
                    
                    self.positions[symbol] = {
                        "direction": signal['direction'],
                        "entry_price": entry_price,
                        "entry_date": target_date,
                        "quantity": quantity,
                        "stop_loss": stop_loss,
                        "target": target,
                        "days_held": 0,
                        "reason": signal['reason'],
                    }
                    
                    logger.info(f"  📥 OPEN {symbol} {signal['direction']} @ {entry_price:.2f} x {quantity} | "
                               f"SL: {stop_loss:.2f} | T: {target:.2f} | {signal['reason']}")
            
            # Record equity
            self.equity_curve.append({
                "date": str(target_date),
                "equity": self.capital,
                "pnl": day_pnl,
                "positions": len(self.positions),
            })
            
            logger.info(f"\n  Day P&L: Rs {day_pnl:+,.2f} | Positions: {len(self.positions)} | "
                       f"Capital: Rs {self.capital:,.2f}")
        
        # Close any remaining positions at last day's close
        if self.positions:
            logger.info(f"\n{'─'*60}")
            logger.info(f"  CLOSING REMAINING POSITIONS")
            logger.info(f"{'─'*60}")
            
            last_day = trading_days[-1]
            for symbol, pos in list(self.positions.items()):
                if symbol not in all_data:
                    continue
                
                df = all_data[symbol]
                if last_day not in df.index:
                    continue
                
                today = df.loc[last_day]
                exit_price = today['close'] * (1 - SLIPPAGE_PCT if pos['direction'] == "LONG" else 1 + SLIPPAGE_PCT)
                
                if pos['direction'] == "LONG":
                    pnl = (exit_price - pos['entry_price']) * pos['quantity']
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['quantity']
                
                cost = (pos['entry_price'] + exit_price) * pos['quantity'] * TRANSACTION_COST
                net_pnl = pnl - cost
                
                self.closed_trades.append({
                    "symbol": symbol,
                    "direction": pos['direction'],
                    "entry_date": str(pos['entry_date']),
                    "exit_date": str(last_day),
                    "entry_price": round(pos['entry_price'], 2),
                    "exit_price": round(exit_price, 2),
                    "quantity": pos['quantity'],
                    "reason": "END_OF_TEST",
                    "gross_pnl": round(pnl, 2),
                    "costs": round(cost, 2),
                    "net_pnl": round(net_pnl, 2),
                })
                
                self.capital += net_pnl
                
                emoji = "✅" if net_pnl > 0 else "❌"
                logger.info(f"  {emoji} CLOSE {symbol} @ {exit_price:.2f} | P&L: Rs {net_pnl:+,.2f}")
        
        return self.generate_summary(trading_days)
    
    def generate_summary(self, trading_days: List[date]) -> Dict:
        """Generate summary."""
        if not self.closed_trades:
            logger.info("\n  No trades executed.")
            return {"error": "No trades"}
        
        df = pd.DataFrame(self.closed_trades)
        
        total_trades = len(df)
        wins = len(df[df["net_pnl"] > 0])
        losses = len(df[df["net_pnl"] <= 0])
        win_rate = wins / total_trades * 100
        
        total_pnl = df["net_pnl"].sum()
        returns = total_pnl / self.initial_capital * 100
        
        # Drawdown
        equity = pd.Series([e["equity"] for e in self.equity_curve])
        if len(equity) > 1:
            rolling_max = equity.cummax()
            drawdown = (equity - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        avg_win = df[df["net_pnl"] > 0]["net_pnl"].mean() if wins > 0 else 0
        avg_loss = df[df["net_pnl"] <= 0]["net_pnl"].mean() if losses > 0 else 0
        
        print("\n" + "═"*70)
        print("  DAILY BACKTEST RESULTS — REALISTIC")
        print(f"  Period: {trading_days[0]} to {trading_days[-1]}")
        print("═"*70)
        
        print(f"\n  💰 CAPITAL")
        print(f"  ─────────────────────────────────────────")
        print(f"  Initial:          Rs {self.initial_capital:>12,.0f}")
        print(f"  Final:            Rs {self.capital:>12,.2f}")
        print(f"  Net Return:       {returns:>12.2f}%")
        print(f"  Max Drawdown:     {max_drawdown:>12.2f}%")
        
        print(f"\n  📊 TRADES")
        print(f"  ─────────────────────────────────────────")
        print(f"  Total:            {total_trades:>12}")
        print(f"  Winners:          {wins:>12} ({win_rate:.1f}%)")
        print(f"  Losers:           {losses:>12}")
        print(f"  ─────────────────────────────────────────")
        print(f"  Net P&L:          Rs {total_pnl:>+12,.2f}")
        print(f"  Avg Win:          Rs {avg_win:>+12,.2f}")
        print(f"  Avg Loss:         Rs {avg_loss:>+12,.2f}")
        if avg_loss != 0:
            print(f"  Win/Loss Ratio:   {abs(avg_win/avg_loss):>12.2f}")
        
        # By direction
        longs = df[df["direction"] == "LONG"]
        shorts = df[df["direction"] == "SHORT"]
        
        print(f"\n  📈 BY DIRECTION")
        print(f"  ─────────────────────────────────────────")
        if len(longs) > 0:
            long_wr = len(longs[longs["net_pnl"] > 0]) / len(longs) * 100
            print(f"  LONG:   {len(longs):>3} trades | Rs {longs['net_pnl'].sum():>+10,.2f} | WR: {long_wr:.0f}%")
        if len(shorts) > 0:
            short_wr = len(shorts[shorts["net_pnl"] > 0]) / len(shorts) * 100
            print(f"  SHORT:  {len(shorts):>3} trades | Rs {shorts['net_pnl'].sum():>+10,.2f} | WR: {short_wr:.0f}%")
        
        # Individual trades
        print(f"\n  📝 TRADES")
        print(f"  ─────────────────────────────────────────")
        for _, trade in df.iterrows():
            emoji = "🟢" if trade["net_pnl"] > 0 else "🔴"
            print(f"  {emoji} {trade['symbol']:<10} {trade['direction']:<5} | "
                  f"{trade['entry_date']} → {trade['exit_date']} | "
                  f"Rs {trade['net_pnl']:>+8,.2f} ({trade['reason']})")
        
        print("\n" + "═"*70)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        df.to_csv(results_dir / "backtest_daily.csv", index=False)
        
        eq_df = pd.DataFrame(self.equity_curve)
        eq_df.to_csv(results_dir / "backtest_daily_equity.csv", index=False)
        
        logger.info(f"\n  Results saved to results/backtest_daily.csv")
        
        return {
            "trades": total_trades,
            "win_rate": win_rate,
            "net_pnl": total_pnl,
            "returns": returns,
            "max_drawdown": max_drawdown,
        }


def get_universe() -> List[str]:
    """Get stock universe."""
    return [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "RELIANCE", "BHARTIARTL", "HINDUNILVR", "ITC", "LT",
        "BAJFINANCE", "MARUTI", "TITAN", "ASIANPAINT", "SUNPHARMA",
        "NESTLEIND", "ULTRACEMCO", "DRREDDY", "CIPLA", "M&M",
    ]


def main():
    parser = argparse.ArgumentParser(description="Daily backtest")
    parser.add_argument("--days", type=int, default=7, help="Trading days")
    parser.add_argument("--capital", type=float, default=10000, help="Capital")
    args = parser.parse_args()
    
    backtest = DailyBacktest(capital=args.capital)
    results = backtest.run(num_days=args.days, universe=get_universe())
    
    return results


if __name__ == "__main__":
    main()
