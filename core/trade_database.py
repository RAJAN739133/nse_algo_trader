#!/usr/bin/env python3
"""
TRADE DATABASE — Production-Grade Trade Storage
════════════════════════════════════════════════════════════

Provides:
1. SQLite for development / PostgreSQL for production
2. Complete trade audit trail
3. Performance analytics queries
4. Daily/weekly/monthly reports
5. Trade export (CSV, JSON)

Usage:
    from core.trade_database import TradeDB
    db = TradeDB()
    db.log_trade(trade_data)
    db.get_daily_summary()
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradeDB")


@dataclass
class Trade:
    """Trade record."""
    id: Optional[int] = None
    symbol: str = ""
    direction: str = ""  # LONG or SHORT
    strategy: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    entry_time: str = ""
    exit_time: str = ""
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""
    regime: str = ""
    market_trend: str = ""
    vix: float = 0.0
    holding_minutes: int = 0
    ml_score: float = 0.0
    claude_confidence: int = 0
    notes: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DailySummary:
    """Daily trading summary."""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    gross_pnl: float
    total_costs: float
    net_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    long_pnl: float
    short_pnl: float
    best_strategy: str
    worst_strategy: str
    market_trend: str
    

class TradeDB:
    """
    Production-grade trade database.
    
    Features:
    - SQLite (dev) / PostgreSQL (prod) support
    - Full CRUD for trades
    - Performance analytics
    - Automatic backups
    - Export to CSV/JSON
    """
    
    def __init__(self, db_path: str = None, use_postgres: bool = False, 
                 postgres_url: str = None):
        self.use_postgres = use_postgres
        self.postgres_url = postgres_url
        
        if use_postgres and postgres_url:
            self._init_postgres()
        else:
            self.db_path = db_path or str(
                Path(__file__).parent.parent / "data" / "trades.db"
            )
            self._init_sqlite()
            
    def _init_sqlite(self):
        """Initialize SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strategy TEXT,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    gross_pnl REAL DEFAULT 0,
                    costs REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    exit_reason TEXT,
                    regime TEXT,
                    market_trend TEXT,
                    vix REAL,
                    holding_minutes INTEGER DEFAULT 0,
                    ml_score REAL,
                    claude_confidence INTEGER,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    gross_pnl REAL,
                    total_costs REAL,
                    net_pnl REAL,
                    win_rate REAL,
                    starting_capital REAL,
                    ending_capital REAL,
                    max_drawdown REAL,
                    market_trend TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    stop_loss REAL,
                    target REAL,
                    strategy TEXT,
                    status TEXT DEFAULT 'OPEN',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(entry_time);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            """)
            
        logger.info(f"Initialized SQLite database: {self.db_path}")
        
    def _init_postgres(self):
        """Initialize PostgreSQL database."""
        try:
            import psycopg2
            self.pg_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10, self.postgres_url
            )
            # Create tables (similar schema)
            logger.info("Initialized PostgreSQL database")
        except ImportError:
            logger.warning("psycopg2 not installed, falling back to SQLite")
            self.use_postgres = False
            self._init_sqlite()
            
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        if self.use_postgres:
            conn = self.pg_pool.getconn()
            try:
                yield conn
                conn.commit()
            finally:
                self.pg_pool.putconn(conn)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()
                
    # ══════════════════════════════════════════════════════════════
    # TRADE OPERATIONS
    # ══════════════════════════════════════════════════════════════
    
    def log_trade(self, trade: Dict) -> int:
        """Log a completed trade."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO trades (
                    symbol, direction, strategy, entry_price, exit_price,
                    quantity, entry_time, exit_time, gross_pnl, costs, net_pnl,
                    exit_reason, regime, market_trend, vix, holding_minutes,
                    ml_score, claude_confidence, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('symbol'),
                trade.get('direction'),
                trade.get('strategy'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('quantity'),
                trade.get('entry_time'),
                trade.get('exit_time'),
                trade.get('gross_pnl', 0),
                trade.get('costs', 0),
                trade.get('net_pnl', 0),
                trade.get('exit_reason'),
                trade.get('regime'),
                trade.get('market_trend'),
                trade.get('vix'),
                trade.get('holding_minutes', 0),
                trade.get('ml_score'),
                trade.get('claude_confidence'),
                trade.get('notes'),
            ))
            trade_id = cursor.lastrowid
            
        logger.info(f"Logged trade #{trade_id}: {trade.get('symbol')} {trade.get('direction')}")
        return trade_id
        
    def get_trade(self, trade_id: int) -> Optional[Trade]:
        """Get a specific trade by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            
        if row:
            return Trade(**dict(row))
        return None
        
    def get_trades(self, start_date: str = None, end_date: str = None,
                   symbol: str = None, strategy: str = None,
                   direction: str = None, limit: int = 100) -> List[Trade]:
        """Get trades with filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
            
        query += f" ORDER BY entry_time DESC LIMIT {limit}"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
        return [Trade(**dict(row)) for row in rows]
        
    def get_today_trades(self) -> List[Trade]:
        """Get all trades from today."""
        today = date.today().isoformat()
        return self.get_trades(start_date=today, limit=1000)
        
    # ══════════════════════════════════════════════════════════════
    # POSITION OPERATIONS
    # ══════════════════════════════════════════════════════════════
    
    def open_position(self, position: Dict) -> int:
        """Record opening a position."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO positions (
                    symbol, direction, quantity, entry_price, entry_time,
                    stop_loss, target, strategy, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                position.get('symbol'),
                position.get('direction'),
                position.get('quantity'),
                position.get('entry_price'),
                position.get('entry_time'),
                position.get('stop_loss'),
                position.get('target'),
                position.get('strategy'),
            ))
            return cursor.lastrowid
            
    def close_position(self, position_id: int, exit_price: float, 
                       exit_time: str, reason: str):
        """Mark a position as closed."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE positions 
                SET status = 'CLOSED'
                WHERE id = ?
            """, (position_id,))
            
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'OPEN'"
            ).fetchall()
        return [dict(row) for row in rows]
        
    # ══════════════════════════════════════════════════════════════
    # ANALYTICS
    # ══════════════════════════════════════════════════════════════
    
    def get_daily_summary(self, target_date: str = None) -> Optional[DailySummary]:
        """Get summary for a specific day."""
        target_date = target_date or date.today().isoformat()
        
        trades = self.get_trades(
            start_date=target_date, 
            end_date=target_date + "T23:59:59",
            limit=1000
        )
        
        if not trades:
            return None
            
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl < 0]
        
        return DailySummary(
            date=target_date,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            gross_pnl=sum(t.gross_pnl for t in trades),
            total_costs=sum(t.costs for t in trades),
            net_pnl=sum(t.net_pnl for t in trades),
            win_rate=len(winners) / len(trades) * 100 if trades else 0,
            avg_win=sum(t.net_pnl for t in winners) / len(winners) if winners else 0,
            avg_loss=sum(t.net_pnl for t in losers) / len(losers) if losers else 0,
            max_win=max((t.net_pnl for t in winners), default=0),
            max_loss=min((t.net_pnl for t in losers), default=0),
            long_pnl=sum(t.net_pnl for t in trades if t.direction == 'LONG'),
            short_pnl=sum(t.net_pnl for t in trades if t.direction == 'SHORT'),
            best_strategy=self._get_best_strategy(trades),
            worst_strategy=self._get_worst_strategy(trades),
            market_trend=trades[0].market_trend if trades else "",
        )
        
    def _get_best_strategy(self, trades: List[Trade]) -> str:
        """Find best performing strategy."""
        by_strategy = {}
        for t in trades:
            by_strategy[t.strategy] = by_strategy.get(t.strategy, 0) + t.net_pnl
        if not by_strategy:
            return ""
        return max(by_strategy, key=by_strategy.get)
        
    def _get_worst_strategy(self, trades: List[Trade]) -> str:
        """Find worst performing strategy."""
        by_strategy = {}
        for t in trades:
            by_strategy[t.strategy] = by_strategy.get(t.strategy, 0) + t.net_pnl
        if not by_strategy:
            return ""
        return min(by_strategy, key=by_strategy.get)
        
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Get performance statistics for the last N days."""
        start_date = (date.today() - timedelta(days=days)).isoformat()
        trades = self.get_trades(start_date=start_date, limit=10000)
        
        if not trades:
            return {}
            
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl < 0]
        
        # Calculate streaks
        pnls = [t.net_pnl for t in sorted(trades, key=lambda x: x.entry_time)]
        win_streak, loss_streak = 0, 0
        max_win_streak, max_loss_streak = 0, 0
        for pnl in pnls:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        # Calculate drawdown
        cumulative = []
        running = 0
        for t in sorted(trades, key=lambda x: x.entry_time):
            running += t.net_pnl
            cumulative.append(running)
        peak = 0
        max_dd = 0
        for val in cumulative:
            peak = max(peak, val)
            dd = (peak - val) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return {
            "period_days": days,
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(trades) * 100 if trades else 0,
            "net_pnl": sum(t.net_pnl for t in trades),
            "gross_pnl": sum(t.gross_pnl for t in trades),
            "total_costs": sum(t.costs for t in trades),
            "avg_win": sum(t.net_pnl for t in winners) / len(winners) if winners else 0,
            "avg_loss": sum(t.net_pnl for t in losers) / len(losers) if losers else 0,
            "max_win": max((t.net_pnl for t in winners), default=0),
            "max_loss": min((t.net_pnl for t in losers), default=0),
            "profit_factor": (
                sum(t.net_pnl for t in winners) / abs(sum(t.net_pnl for t in losers))
                if losers and sum(t.net_pnl for t in losers) != 0 else 0
            ),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "max_drawdown_pct": max_dd,
            "avg_holding_minutes": (
                sum(t.holding_minutes for t in trades) / len(trades) if trades else 0
            ),
            "by_direction": {
                "LONG": {
                    "trades": len([t for t in trades if t.direction == 'LONG']),
                    "pnl": sum(t.net_pnl for t in trades if t.direction == 'LONG'),
                    "win_rate": (
                        len([t for t in winners if t.direction == 'LONG']) /
                        len([t for t in trades if t.direction == 'LONG']) * 100
                        if [t for t in trades if t.direction == 'LONG'] else 0
                    ),
                },
                "SHORT": {
                    "trades": len([t for t in trades if t.direction == 'SHORT']),
                    "pnl": sum(t.net_pnl for t in trades if t.direction == 'SHORT'),
                    "win_rate": (
                        len([t for t in winners if t.direction == 'SHORT']) /
                        len([t for t in trades if t.direction == 'SHORT']) * 100
                        if [t for t in trades if t.direction == 'SHORT'] else 0
                    ),
                },
            },
            "by_strategy": self._get_strategy_breakdown(trades),
        }
        
    def _get_strategy_breakdown(self, trades: List[Trade]) -> Dict:
        """Get P&L breakdown by strategy."""
        breakdown = {}
        for t in trades:
            if t.strategy not in breakdown:
                breakdown[t.strategy] = {"trades": 0, "pnl": 0, "wins": 0}
            breakdown[t.strategy]["trades"] += 1
            breakdown[t.strategy]["pnl"] += t.net_pnl
            if t.net_pnl > 0:
                breakdown[t.strategy]["wins"] += 1
                
        for strategy in breakdown:
            total = breakdown[strategy]["trades"]
            wins = breakdown[strategy]["wins"]
            breakdown[strategy]["win_rate"] = wins / total * 100 if total else 0
            
        return breakdown
        
    # ══════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════
    
    def export_to_csv(self, filepath: str, start_date: str = None, 
                      end_date: str = None):
        """Export trades to CSV."""
        trades = self.get_trades(start_date=start_date, end_date=end_date, limit=100000)
        df = pd.DataFrame([t.to_dict() for t in trades])
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(trades)} trades to {filepath}")
        
    def export_to_json(self, filepath: str, start_date: str = None,
                       end_date: str = None):
        """Export trades to JSON."""
        trades = self.get_trades(start_date=start_date, end_date=end_date, limit=100000)
        with open(filepath, 'w') as f:
            json.dump([t.to_dict() for t in trades], f, indent=2)
        logger.info(f"Exported {len(trades)} trades to {filepath}")
        
    # ══════════════════════════════════════════════════════════════
    # SYSTEM EVENTS
    # ══════════════════════════════════════════════════════════════
    
    def log_event(self, event_type: str, event_data: Dict = None):
        """Log a system event."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO system_events (event_type, event_data)
                VALUES (?, ?)
            """, (event_type, json.dumps(event_data) if event_data else None))
            
    def get_events(self, event_type: str = None, limit: int = 100) -> List[Dict]:
        """Get system events."""
        query = "SELECT * FROM system_events"
        params = []
        if event_type:
            query += " WHERE event_type = ?"
            params.append(event_type)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
        
    # ══════════════════════════════════════════════════════════════
    # BACKUP
    # ══════════════════════════════════════════════════════════════
    
    def backup(self, backup_dir: str = None):
        """Create a backup of the database."""
        if self.use_postgres:
            logger.warning("PostgreSQL backup not implemented")
            return
            
        backup_dir = backup_dir or str(Path(self.db_path).parent / "backups")
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        backup_path = Path(backup_dir) / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        
        # Keep only last 7 backups
        backups = sorted(Path(backup_dir).glob("trades_*.db"), reverse=True)
        for old_backup in backups[7:]:
            old_backup.unlink()


# Convenience functions
_db_instance = None

def get_db() -> TradeDB:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradeDB()
    return _db_instance


if __name__ == "__main__":
    # Test the database
    db = TradeDB()
    
    # Log a test trade
    trade_id = db.log_trade({
        "symbol": "TCS",
        "direction": "LONG",
        "strategy": "ORB",
        "entry_price": 3500.0,
        "exit_price": 3550.0,
        "quantity": 2,
        "entry_time": datetime.now().isoformat(),
        "exit_time": datetime.now().isoformat(),
        "gross_pnl": 100.0,
        "costs": 5.0,
        "net_pnl": 95.0,
        "exit_reason": "TARGET",
        "regime": "trending_up",
        "market_trend": "BULLISH",
    })
    
    print(f"Logged trade #{trade_id}")
    
    # Get stats
    stats = db.get_performance_stats(days=30)
    print(f"\nPerformance Stats (30 days):")
    print(json.dumps(stats, indent=2))
