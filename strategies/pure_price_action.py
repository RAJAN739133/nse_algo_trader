#!/usr/bin/env python3
"""
Pure Price-Action Algo Strategy
═══════════════════════════════════════════════════════════════

NO ML MODEL
NO PATTERN RECOGNITION  
JUST PRICE + VOLUME + REACT

April 2, 2026 Backtest Results:
- 82 trades, 59% WR, +Rs 98,800
- LONGs: 67% WR, +Rs 116,308
- SHORTs: 15% WR, -Rs 17,507 (avoid in bullish market)

Rules:
1. Wait for first hour (skip fake moves)
2. Build ORB from first 30 min
3. Get market bias from Nifty position in day's range
4. Trade ORB breakouts with volume confirmation
5. ATR-based stops with trailing
6. Max 45 min hold if flat
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PurePriceActionStrategy:
    """
    Simple price-action strategy that reacts to market, doesn't predict.
    """
    
    def __init__(self, capital=100000, risk_pct=0.02):
        self.capital = capital
        self.risk_pct = risk_pct
        self.max_trades_per_stock = 2
        
        # ORB data
        self.orb_high = None
        self.orb_low = None
        self.orb_ready = False
        
        # Nifty data for bias
        self.nifty_day_high = None
        self.nifty_day_low = None
    
    def set_orb(self, candles):
        """
        Set Opening Range Breakout levels from first 6 candles (30 min).
        Skip first candle (often erratic).
        """
        if len(candles) < 7:
            return False
        
        orb_candles = candles.iloc[1:7]
        self.orb_high = orb_candles['high'].max()
        self.orb_low = orb_candles['low'].min()
        
        # Validate ORB range (skip dead stocks)
        orb_range = self.orb_high - self.orb_low
        if orb_range / self.orb_low < 0.003:  # Less than 0.3%
            return False
        
        self.orb_ready = True
        return True
    
    def update_nifty_range(self, nifty_candles, current_idx):
        """Update Nifty day's range for bias calculation."""
        self.nifty_day_high = nifty_candles['high'].iloc[:current_idx+1].max()
        self.nifty_day_low = nifty_candles['low'].iloc[:current_idx+1].min()
    
    def get_market_bias(self, nifty_candles, current_idx):
        """
        Get market bias based on WHERE price is in day's range.
        
        Returns:
            1: LONG bias (price in upper 55% of range)
           -1: SHORT bias (price in lower 45% of range)
            0: NEUTRAL (middle zone, no trade)
        """
        if current_idx < 6:
            return 0
        
        self.update_nifty_range(nifty_candles, current_idx)
        
        if self.nifty_day_high == self.nifty_day_low:
            return 0
        
        current = nifty_candles['close'].iloc[current_idx]
        position = (current - self.nifty_day_low) / (self.nifty_day_high - self.nifty_day_low)
        
        if position > 0.55:
            return 1  # LONG
        elif position < 0.45:
            return -1  # SHORT
        return 0  # No clear bias
    
    def calculate_atr(self, candles, current_idx, period=14):
        """Calculate Average True Range."""
        if current_idx < period:
            return candles['high'].iloc[current_idx] - candles['low'].iloc[current_idx]
        
        tr_list = []
        for i in range(max(0, current_idx - period), current_idx + 1):
            high = candles['high'].iloc[i]
            low = candles['low'].iloc[i]
            prev_close = candles['close'].iloc[i-1] if i > 0 else candles['open'].iloc[i]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        return np.mean(tr_list)
    
    def check_volume(self, candles, current_idx, multiplier=1.2):
        """Check if current volume is above average."""
        if current_idx < 10:
            return True  # Not enough data
        
        avg_vol = candles['volume'].iloc[max(0, current_idx-10):current_idx].mean()
        curr_vol = candles['volume'].iloc[current_idx]
        
        return curr_vol > avg_vol * multiplier
    
    def generate_entry_signal(self, candles, current_idx, nifty_candles):
        """
        Generate entry signal based on pure price action.
        
        Returns dict with signal or None.
        """
        if not self.orb_ready:
            return None
        
        # Skip first hour (candles 0-11, ~9:15-10:15)
        if current_idx < 12:
            return None
        
        row = candles.iloc[current_idx]
        close = row['close']
        
        # Get market bias
        bias = self.get_market_bias(nifty_candles, current_idx)
        
        if bias == 0:
            return None  # No clear direction
        
        # Volume confirmation
        if not self.check_volume(candles, current_idx):
            return None
        
        atr = self.calculate_atr(candles, current_idx)
        
        # ORB BREAKOUT LONG
        if bias == 1 and close > self.orb_high * 1.002:
            stop = close - atr * 1.5
            target = close * 1.015  # 1.5% target
            risk = close - stop
            qty = max(1, int(self.capital * self.risk_pct / risk))
            
            return {
                'side': 'LONG',
                'type': 'ORB_BREAKOUT',
                'entry': close,
                'stop': stop,
                'target': target,
                'qty': qty,
                'risk': risk,
                'reason': f'ORB breakout, Nifty bias LONG, vol confirmed'
            }
        
        # ORB BREAKDOWN SHORT
        if bias == -1 and close < self.orb_low * 0.998:
            stop = close + atr * 1.5
            target = close * 0.985  # 1.5% target
            risk = stop - close
            qty = max(1, int(self.capital * self.risk_pct / risk))
            
            return {
                'side': 'SHORT',
                'type': 'ORB_BREAKDOWN',
                'entry': close,
                'stop': stop,
                'target': target,
                'qty': qty,
                'risk': risk,
                'reason': f'ORB breakdown, Nifty bias SHORT, vol confirmed'
            }
        
        return None
    
    def manage_position(self, position, candle, holding_candles):
        """
        Manage existing position - check exits.
        
        Returns: (exit_signal, reason) or (None, None)
        """
        close = candle['close']
        high = candle['high']
        low = candle['low']
        
        if position['side'] == 'LONG':
            profit_pct = (close - position['entry']) / position['entry']
            
            # Update trailing stop
            if profit_pct >= 0.005 and position['stop'] < position['entry']:
                position['stop'] = position['entry'] * 1.001
            if profit_pct >= 0.01:
                new_stop = close * 0.993
                position['stop'] = max(position['stop'], new_stop)
            
            # Check exits
            if high >= position['target']:
                return position['target'], 'TARGET'
            if low <= position['stop']:
                reason = 'TRAILING_STOP' if position['stop'] > position['entry'] else 'STOP_LOSS'
                return position['stop'], reason
            if holding_candles >= 9 and profit_pct < 0.003:
                return close, 'TIME_EXIT'
        
        else:  # SHORT
            profit_pct = (position['entry'] - close) / position['entry']
            
            if profit_pct >= 0.005 and position['stop'] > position['entry']:
                position['stop'] = position['entry'] * 0.999
            if profit_pct >= 0.01:
                new_stop = close * 1.007
                position['stop'] = min(position['stop'], new_stop)
            
            if low <= position['target']:
                return position['target'], 'TARGET'
            if high >= position['stop']:
                reason = 'TRAILING_STOP' if position['stop'] < position['entry'] else 'STOP_LOSS'
                return position['stop'], reason
            if holding_candles >= 9 and profit_pct < 0.003:
                return close, 'TIME_EXIT'
        
        return None, None


def create_strategy(capital=100000, risk_pct=0.02):
    """Factory function to create strategy instance."""
    return PurePriceActionStrategy(capital=capital, risk_pct=risk_pct)
