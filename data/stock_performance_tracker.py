"""
Dynamic Stock Performance Tracker
═══════════════════════════════════════════════════════════════

Instead of hardcoded blacklist/whitelist, this module:
1. Tracks rolling performance of each stock over last N days
2. Calculates win rate, average P&L, Sharpe ratio per stock
3. Dynamically scores stocks based on RECENT performance
4. Updates daily before market open

This ensures:
- No hardcoding - adapts to market changes
- Same logic in backtest and production
- Stocks that stop performing get filtered out
- New winners get promoted

Usage:
    tracker = StockPerformanceTracker(lookback_days=60)
    tracker.update_from_trades(trades_df)  # From backtest or live
    
    # Get dynamic scores
    good_stocks = tracker.get_preferred_stocks(min_trades=5, min_wr=0.55)
    bad_stocks = tracker.get_avoid_stocks(min_trades=5, max_wr=0.40)
    score = tracker.get_stock_score("RELIANCE")
"""

import json
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRACKER_FILE = Path("data/stock_performance.json")


@dataclass
class StockStats:
    """Performance statistics for a single stock."""
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 1.0
    sharpe: float = 0.0
    last_updated: str = ""
    
    # Rolling metrics
    recent_trades: int = 0
    recent_wr: float = 0.5
    recent_pnl: float = 0.0
    
    @property
    def score(self) -> float:
        """
        Composite score (0-100) based on multiple factors.
        Higher = better stock to trade.
        """
        if self.total_trades < 3:
            return 50.0  # Neutral for insufficient data
        
        # Win rate component (0-40 points)
        wr_score = max(0, min(40, (self.win_rate - 0.4) * 100))
        
        # Profit factor component (0-30 points)
        pf_score = max(0, min(30, (self.profit_factor - 0.8) * 30))
        
        # Consistency component (0-20 points) - recent vs overall
        if self.recent_trades >= 3:
            consistency = 1 - abs(self.recent_wr - self.win_rate)
            cons_score = consistency * 20
        else:
            cons_score = 10  # Neutral
        
        # Volume of trades (0-10 points) - more data = more confidence
        vol_score = min(10, self.total_trades / 2)
        
        return round(wr_score + pf_score + cons_score + vol_score, 1)


class StockPerformanceTracker:
    """
    Tracks and scores stock performance dynamically.
    Used in both backtest and production for consistency.
    """
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.stats: Dict[str, StockStats] = {}
        self.trades_history: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load existing performance data."""
        if TRACKER_FILE.exists():
            try:
                with open(TRACKER_FILE) as f:
                    data = json.load(f)
                self.stats = {
                    sym: StockStats(**s) for sym, s in data.get("stats", {}).items()
                }
                self.trades_history = data.get("trades", [])
                logger.info(f"Loaded performance data for {len(self.stats)} stocks")
            except Exception as e:
                logger.warning(f"Could not load tracker: {e}")
    
    def _save(self):
        """Save performance data."""
        TRACKER_FILE.parent.mkdir(exist_ok=True)
        data = {
            "stats": {sym: asdict(s) for sym, s in self.stats.items()},
            "trades": self.trades_history[-1000:],  # Keep last 1000 trades
            "updated": datetime.now().isoformat(),
        }
        with open(TRACKER_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    def record_trade(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        trade_date: date,
        confidence: float = 0.5,
    ):
        """Record a single trade result."""
        self.trades_history.append({
            "symbol": symbol,
            "direction": direction,
            "pnl": pnl,
            "date": str(trade_date),
            "confidence": confidence,
            "won": pnl > 0,
        })
        self._update_stats(symbol)
    
    def update_from_trades(self, trades_df: pd.DataFrame):
        """
        Bulk update from trades DataFrame.
        Expected columns: symbol, direction, net_pnl, date
        """
        if trades_df.empty:
            return
        
        for _, row in trades_df.iterrows():
            self.trades_history.append({
                "symbol": row["symbol"],
                "direction": row.get("direction", "LONG"),
                "pnl": row.get("net_pnl", row.get("pnl", 0)),
                "date": str(row.get("date", date.today())),
                "confidence": row.get("confidence", 0.5),
                "won": row.get("net_pnl", row.get("pnl", 0)) > 0,
            })
        
        # Update stats for all affected symbols
        for sym in trades_df["symbol"].unique():
            self._update_stats(sym)
        
        self._save()
        logger.info(f"Updated tracker with {len(trades_df)} trades")
    
    def _update_stats(self, symbol: str):
        """Recalculate stats for a symbol from trade history."""
        # Filter trades for this symbol
        sym_trades = [t for t in self.trades_history if t["symbol"] == symbol]
        
        if not sym_trades:
            return
        
        # All-time metrics
        total = len(sym_trades)
        wins = [t for t in sym_trades if t["won"]]
        losses = [t for t in sym_trades if not t["won"]]
        
        total_pnl = sum(t["pnl"] for t in sym_trades)
        win_pnls = [t["pnl"] for t in wins]
        loss_pnls = [t["pnl"] for t in losses]
        
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        gross_profit = sum(win_pnls) if win_pnls else 0
        gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0
        
        # Recent metrics (last N days)
        cutoff = (date.today() - timedelta(days=self.lookback_days)).isoformat()
        recent = [t for t in sym_trades if t["date"] >= cutoff]
        recent_wins = [t for t in recent if t["won"]]
        
        # Sharpe-like ratio
        pnls = [t["pnl"] for t in sym_trades]
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) > 0 else 0
        
        self.stats[symbol] = StockStats(
            symbol=symbol,
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=round(total_pnl, 2),
            avg_pnl=round(total_pnl / total, 2) if total > 0 else 0,
            win_rate=round(len(wins) / total, 3) if total > 0 else 0.5,
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 2.0,
            sharpe=round(sharpe, 2),
            last_updated=datetime.now().isoformat(),
            recent_trades=len(recent),
            recent_wr=round(len(recent_wins) / len(recent), 3) if recent else 0.5,
            recent_pnl=round(sum(t["pnl"] for t in recent), 2),
        )
    
    def get_stock_score(self, symbol: str) -> float:
        """Get dynamic score for a stock (0-100)."""
        if symbol not in self.stats:
            return 50.0  # Neutral for unknown stocks
        return self.stats[symbol].score
    
    def get_preferred_stocks(
        self,
        min_trades: int = 5,
        min_wr: float = 0.52,
        min_pf: float = 1.0,
        top_n: int = 20,
    ) -> Set[str]:
        """
        Get stocks that have performed well recently.
        Dynamic equivalent of whitelist.
        """
        good = []
        for sym, stats in self.stats.items():
            if stats.total_trades < min_trades:
                continue
            if stats.win_rate < min_wr:
                continue
            if stats.profit_factor < min_pf:
                continue
            good.append((sym, stats.score))
        
        # Sort by score, take top N
        good.sort(key=lambda x: x[1], reverse=True)
        return set(sym for sym, _ in good[:top_n])
    
    def get_avoid_stocks(
        self,
        min_trades: int = 5,
        max_wr: float = 0.42,
        max_pf: float = 0.8,
    ) -> Set[str]:
        """
        Get stocks that have performed poorly.
        Dynamic equivalent of blacklist.
        """
        bad = set()
        for sym, stats in self.stats.items():
            if stats.total_trades < min_trades:
                continue
            if stats.win_rate <= max_wr or stats.profit_factor <= max_pf:
                bad.add(sym)
        return bad
    
    def get_direction_stats(self, symbol: str) -> Tuple[float, float]:
        """
        Get win rate for LONG vs SHORT for a symbol.
        Returns: (long_wr, short_wr)
        """
        sym_trades = [t for t in self.trades_history if t["symbol"] == symbol]
        
        longs = [t for t in sym_trades if t["direction"] == "LONG"]
        shorts = [t for t in sym_trades if t["direction"] == "SHORT"]
        
        long_wr = len([t for t in longs if t["won"]]) / len(longs) if longs else 0.5
        short_wr = len([t for t in shorts if t["won"]]) / len(shorts) if shorts else 0.5
        
        return round(long_wr, 3), round(short_wr, 3)
    
    def should_prefer_short(self, symbol: str, threshold: float = 0.1) -> bool:
        """Check if SHORT direction works better for this stock."""
        long_wr, short_wr = self.get_direction_stats(symbol)
        return short_wr > long_wr + threshold
    
    def adjust_confidence(self, symbol: str, base_confidence: float) -> float:
        """
        Adjust ML confidence based on historical performance.
        Good stocks get boosted, bad stocks get penalized.
        """
        score = self.get_stock_score(symbol)
        
        # Score is 0-100, center at 50
        # Multiply confidence by factor: 0.8 to 1.2
        factor = 0.8 + (score / 100) * 0.4
        
        return base_confidence * factor
    
    def get_summary(self) -> Dict:
        """Get summary of tracked performance."""
        if not self.stats:
            return {"message": "No performance data yet"}
        
        all_scores = [s.score for s in self.stats.values()]
        preferred = self.get_preferred_stocks()
        avoid = self.get_avoid_stocks()
        
        return {
            "total_stocks_tracked": len(self.stats),
            "total_trades": len(self.trades_history),
            "avg_score": round(np.mean(all_scores), 1),
            "preferred_count": len(preferred),
            "avoid_count": len(avoid),
            "preferred_stocks": list(preferred)[:10],
            "avoid_stocks": list(avoid)[:10],
        }
    
    def clear_old_trades(self, days: int = 180):
        """Remove trades older than N days."""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        self.trades_history = [t for t in self.trades_history if t["date"] >= cutoff]
        
        # Recalculate all stats
        for sym in list(self.stats.keys()):
            self._update_stats(sym)
        
        self._save()


# ════════════════════════════════════════════════════════════
# INTEGRATED SCORING FOR BACKTEST & PRODUCTION
# ════════════════════════════════════════════════════════════

def get_dynamic_stock_filter(
    as_of_date: date,
    trades_history: List[Dict],
    lookback_days: int = 60,
) -> Tuple[Set[str], Set[str], Dict[str, float]]:
    """
    Calculate dynamic blacklist/whitelist as of a specific date.
    Used in walk-forward backtest to avoid future leakage.
    
    Returns:
        (avoid_stocks, preferred_stocks, stock_scores)
    """
    # Filter trades to only those before as_of_date
    cutoff_start = (as_of_date - timedelta(days=lookback_days)).isoformat()
    cutoff_end = as_of_date.isoformat()
    
    relevant_trades = [
        t for t in trades_history
        if cutoff_start <= t["date"] < cutoff_end
    ]
    
    if len(relevant_trades) < 10:
        return set(), set(), {}
    
    # Calculate stats per symbol
    stats = {}
    for t in relevant_trades:
        sym = t["symbol"]
        if sym not in stats:
            stats[sym] = {"wins": 0, "losses": 0, "pnl": 0.0}
        
        if t["won"]:
            stats[sym]["wins"] += 1
        else:
            stats[sym]["losses"] += 1
        stats[sym]["pnl"] += t["pnl"]
    
    avoid = set()
    preferred = set()
    scores = {}
    
    for sym, s in stats.items():
        total = s["wins"] + s["losses"]
        if total < 3:
            scores[sym] = 50.0
            continue
        
        wr = s["wins"] / total
        avg_pnl = s["pnl"] / total
        
        # Score: 50 baseline + win_rate contribution + pnl contribution
        score = 50 + (wr - 0.5) * 60 + min(20, max(-20, avg_pnl / 50))
        scores[sym] = round(score, 1)
        
        if wr < 0.42 or avg_pnl < -100:
            avoid.add(sym)
        elif wr > 0.55 and avg_pnl > 50:
            preferred.add(sym)
    
    return avoid, preferred, scores


# Singleton tracker for production use
_tracker: Optional[StockPerformanceTracker] = None

def get_tracker() -> StockPerformanceTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = StockPerformanceTracker()
    return _tracker
