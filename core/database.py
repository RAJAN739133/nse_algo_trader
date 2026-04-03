"""
Database Backend - TimescaleDB/PostgreSQL integration for time-series data.

Features:
- Trade storage and retrieval
- Candle data management
- Performance metrics persistence
- Efficient time-series queries
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

logger = logging.getLogger(__name__)

# Try to import database libraries
try:
    import psycopg2
    from psycopg2.extras import execute_values, RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.info("psycopg2 not installed - using SQLite fallback")

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


@dataclass
class TradeRecord:
    """Trade record for database storage."""
    trade_id: str
    symbol: str
    side: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy: str
    exit_reason: str
    sector: str = ""
    metadata: Dict = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = json.dumps(d.get("metadata") or {})
        return d


@dataclass
class CandleRecord:
    """Candle/OHLCV record."""
    symbol: str
    timestamp: datetime
    timeframe: str  # "1m", "5m", "15m", "1h", "1d"
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class MetricRecord:
    """Performance metric record."""
    metric_name: str
    metric_value: float
    timestamp: datetime
    period: str  # "daily", "weekly", "monthly"
    metadata: Dict = None


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    @abstractmethod
    def create_tables(self):
        pass
    
    @abstractmethod
    def insert_trade(self, trade: TradeRecord):
        pass
    
    @abstractmethod
    def insert_candle(self, candle: CandleRecord):
        pass
    
    @abstractmethod
    def get_trades(
        self,
        start_date: date = None,
        end_date: date = None,
        symbol: str = None,
        strategy: str = None
    ) -> List[TradeRecord]:
        pass
    
    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime = None
    ) -> List[CandleRecord]:
        pass


class SQLiteBackend(DatabaseBackend):
    """
    SQLite database backend - simple file-based storage.
    Good for development and small-scale usage.
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._lock = threading.Lock()
    
    def connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
        logger.info(f"Connected to SQLite: {self.db_path}")
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        with self._lock:
            cursor = self.conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    strategy TEXT,
                    exit_reason TEXT,
                    sector TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Candles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    period TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions table (current state)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    entry_time TEXT NOT NULL,
                    strategy TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    metadata TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles(symbol, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp)")
            
            self.conn.commit()
    
    def insert_trade(self, trade: TradeRecord):
        """Insert a trade record."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trades 
                (trade_id, symbol, side, qty, entry_price, exit_price, 
                 entry_time, exit_time, pnl, pnl_pct, strategy, exit_reason, sector, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id, trade.symbol, trade.side, trade.qty,
                trade.entry_price, trade.exit_price,
                trade.entry_time.isoformat(), trade.exit_time.isoformat(),
                trade.pnl, trade.pnl_pct, trade.strategy, trade.exit_reason,
                trade.sector, json.dumps(trade.metadata or {})
            ))
            self.conn.commit()
    
    def insert_trades_batch(self, trades: List[TradeRecord]):
        """Insert multiple trades efficiently."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO trades 
                (trade_id, symbol, side, qty, entry_price, exit_price, 
                 entry_time, exit_time, pnl, pnl_pct, strategy, exit_reason, sector, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (t.trade_id, t.symbol, t.side, t.qty, t.entry_price, t.exit_price,
                 t.entry_time.isoformat(), t.exit_time.isoformat(),
                 t.pnl, t.pnl_pct, t.strategy, t.exit_reason, t.sector,
                 json.dumps(t.metadata or {}))
                for t in trades
            ])
            self.conn.commit()
    
    def insert_candle(self, candle: CandleRecord):
        """Insert a candle record."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO candles 
                (symbol, timestamp, timeframe, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle.symbol, candle.timestamp.isoformat(), candle.timeframe,
                candle.open, candle.high, candle.low, candle.close, candle.volume
            ))
            self.conn.commit()
    
    def insert_candles_batch(self, candles: List[CandleRecord]):
        """Insert multiple candles efficiently."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO candles 
                (symbol, timestamp, timeframe, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (c.symbol, c.timestamp.isoformat(), c.timeframe,
                 c.open, c.high, c.low, c.close, c.volume)
                for c in candles
            ])
            self.conn.commit()
    
    def insert_metric(self, metric: MetricRecord):
        """Insert a metric record."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (metric_name, metric_value, timestamp, period, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.metric_name, metric.metric_value,
                metric.timestamp.isoformat(), metric.period,
                json.dumps(metric.metadata or {})
            ))
            self.conn.commit()
    
    def get_trades(
        self,
        start_date: date = None,
        end_date: date = None,
        symbol: str = None,
        strategy: str = None
    ) -> List[TradeRecord]:
        """Get trades with filters."""
        with self._lock:
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND date(exit_time) >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date(exit_time) <= ?"
                params.append(end_date.isoformat())
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            query += " ORDER BY exit_time DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                TradeRecord(
                    trade_id=row["trade_id"],
                    symbol=row["symbol"],
                    side=row["side"],
                    qty=row["qty"],
                    entry_price=row["entry_price"],
                    exit_price=row["exit_price"],
                    entry_time=datetime.fromisoformat(row["entry_time"]),
                    exit_time=datetime.fromisoformat(row["exit_time"]),
                    pnl=row["pnl"],
                    pnl_pct=row["pnl_pct"],
                    strategy=row["strategy"],
                    exit_reason=row["exit_reason"],
                    sector=row["sector"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )
                for row in rows
            ]
    
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime = None
    ) -> List[CandleRecord]:
        """Get candles for a symbol and timeframe."""
        with self._lock:
            cursor = self.conn.cursor()
            
            query = """
                SELECT * FROM candles 
                WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
            """
            params = [symbol, timeframe, start_time.isoformat()]
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                CandleRecord(
                    symbol=row["symbol"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    timeframe=row["timeframe"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                )
                for row in rows
            ]
    
    def get_daily_pnl(self, days: int = 30) -> List[Dict]:
        """Get daily P&L for the last N days."""
        with self._lock:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT date(exit_time) as trade_date, 
                       SUM(pnl) as total_pnl,
                       COUNT(*) as trade_count,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM trades
                WHERE exit_time >= date('now', ?)
                GROUP BY date(exit_time)
                ORDER BY trade_date DESC
            """, (f"-{days} days",))
            
            return [
                {
                    "date": row["trade_date"],
                    "pnl": row["total_pnl"],
                    "trades": row["trade_count"],
                    "wins": row["wins"],
                    "win_rate": row["wins"] / row["trade_count"] * 100 if row["trade_count"] > 0 else 0
                }
                for row in cursor.fetchall()
            ]
    
    def get_strategy_stats(self, start_date: date = None) -> Dict[str, Dict]:
        """Get statistics by strategy."""
        with self._lock:
            cursor = self.conn.cursor()
            
            query = """
                SELECT strategy,
                       COUNT(*) as trades,
                       SUM(pnl) as total_pnl,
                       AVG(pnl) as avg_pnl,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                       AVG(CASE WHEN pnl <= 0 THEN pnl END) as avg_loss
                FROM trades
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND date(exit_time) >= ?"
                params.append(start_date.isoformat())
            
            query += " GROUP BY strategy"
            
            cursor.execute(query, params)
            
            return {
                row["strategy"]: {
                    "trades": row["trades"],
                    "total_pnl": row["total_pnl"],
                    "avg_pnl": row["avg_pnl"],
                    "win_rate": row["wins"] / row["trades"] * 100 if row["trades"] > 0 else 0,
                    "avg_win": row["avg_win"] or 0,
                    "avg_loss": row["avg_loss"] or 0
                }
                for row in cursor.fetchall()
            }
    
    def save_position(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        entry_time: datetime,
        strategy: str = "",
        stop_loss: float = None,
        take_profit: float = None
    ):
        """Save current position to database."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, side, qty, entry_price, entry_time, strategy, stop_loss, take_profit, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, side, qty, entry_price, entry_time.isoformat(),
                strategy, stop_loss, take_profit, datetime.now().isoformat()
            ))
            self.conn.commit()
    
    def remove_position(self, symbol: str):
        """Remove a position from database."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            self.conn.commit()
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM positions")
            return [dict(row) for row in cursor.fetchall()]
    
    def vacuum(self):
        """Optimize database."""
        with self._lock:
            self.conn.execute("VACUUM")


class TimescaleDBBackend(DatabaseBackend):
    """
    TimescaleDB backend for high-performance time-series storage.
    Requires PostgreSQL with TimescaleDB extension.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "trading",
        user: str = "trader",
        password: str = ""
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self._lock = threading.Lock()
    
    def connect(self):
        """Connect to TimescaleDB."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        self.conn.autocommit = True
        self.create_tables()
        logger.info(f"Connected to TimescaleDB: {self.host}:{self.port}/{self.database}")
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_tables(self):
        """Create database tables and hypertables."""
        with self._lock:
            cursor = self.conn.cursor()
            
            # Create TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    entry_price NUMERIC NOT NULL,
                    exit_price NUMERIC NOT NULL,
                    entry_time TIMESTAMPTZ NOT NULL,
                    exit_time TIMESTAMPTZ NOT NULL,
                    pnl NUMERIC NOT NULL,
                    pnl_pct NUMERIC NOT NULL,
                    strategy TEXT,
                    exit_reason TEXT,
                    sector TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Candles hypertable (optimized for time-series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    timeframe TEXT NOT NULL,
                    open NUMERIC NOT NULL,
                    high NUMERIC NOT NULL,
                    low NUMERIC NOT NULL,
                    close NUMERIC NOT NULL,
                    volume BIGINT NOT NULL,
                    PRIMARY KEY (symbol, timestamp, timeframe)
                );
            """)
            
            # Convert to hypertable if not already
            try:
                cursor.execute("""
                    SELECT create_hypertable('candles', 'timestamp', 
                                            chunk_time_interval => INTERVAL '1 day',
                                            if_not_exists => TRUE);
                """)
            except Exception as e:
                logger.debug(f"Hypertable may already exist: {e}")
            
            # Metrics hypertable
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value NUMERIC NOT NULL,
                    period TEXT NOT NULL,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, metric_name, period)
                );
            """)
            
            try:
                cursor.execute("""
                    SELECT create_hypertable('metrics', 'timestamp',
                                            chunk_time_interval => INTERVAL '7 days',
                                            if_not_exists => TRUE);
                """)
            except Exception:
                pass
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles(symbol, timeframe);")
    
    def insert_trade(self, trade: TradeRecord):
        """Insert a trade record."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades 
                (trade_id, symbol, side, qty, entry_price, exit_price, 
                 entry_time, exit_time, pnl, pnl_pct, strategy, exit_reason, sector, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_id) DO UPDATE SET
                    pnl = EXCLUDED.pnl,
                    exit_price = EXCLUDED.exit_price,
                    exit_time = EXCLUDED.exit_time
            """, (
                trade.trade_id, trade.symbol, trade.side, trade.qty,
                trade.entry_price, trade.exit_price,
                trade.entry_time, trade.exit_time,
                trade.pnl, trade.pnl_pct, trade.strategy, trade.exit_reason,
                trade.sector, json.dumps(trade.metadata or {})
            ))
    
    def insert_candle(self, candle: CandleRecord):
        """Insert a candle record."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO candles (symbol, timestamp, timeframe, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume
            """, (
                candle.symbol, candle.timestamp, candle.timeframe,
                candle.open, candle.high, candle.low, candle.close, candle.volume
            ))
    
    def insert_candles_batch(self, candles: List[CandleRecord]):
        """Insert multiple candles efficiently."""
        if not candles:
            return
        
        with self._lock:
            cursor = self.conn.cursor()
            execute_values(
                cursor,
                """INSERT INTO candles (symbol, timestamp, timeframe, open, high, low, close, volume)
                   VALUES %s
                   ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                       open = EXCLUDED.open, high = EXCLUDED.high,
                       low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume""",
                [(c.symbol, c.timestamp, c.timeframe, c.open, c.high, c.low, c.close, c.volume) 
                 for c in candles]
            )
    
    def get_trades(
        self,
        start_date: date = None,
        end_date: date = None,
        symbol: str = None,
        strategy: str = None
    ) -> List[TradeRecord]:
        """Get trades with filters."""
        with self._lock:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND exit_time::date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND exit_time::date <= %s"
                params.append(end_date)
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            if strategy:
                query += " AND strategy = %s"
                params.append(strategy)
            
            query += " ORDER BY exit_time DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                TradeRecord(
                    trade_id=row["trade_id"],
                    symbol=row["symbol"],
                    side=row["side"],
                    qty=row["qty"],
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    entry_time=row["entry_time"],
                    exit_time=row["exit_time"],
                    pnl=float(row["pnl"]),
                    pnl_pct=float(row["pnl_pct"]),
                    strategy=row["strategy"],
                    exit_reason=row["exit_reason"],
                    sector=row["sector"],
                    metadata=row["metadata"]
                )
                for row in rows
            ]
    
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime = None
    ) -> List[CandleRecord]:
        """Get candles for a symbol and timeframe."""
        with self._lock:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT * FROM candles 
                WHERE symbol = %s AND timeframe = %s AND timestamp >= %s
            """
            params = [symbol, timeframe, start_time]
            
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time)
            
            query += " ORDER BY timestamp ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                CandleRecord(
                    symbol=row["symbol"],
                    timestamp=row["timestamp"],
                    timeframe=row["timeframe"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=row["volume"]
                )
                for row in rows
            ]


# Factory function to get appropriate backend
def get_database(backend_type: str = "sqlite", **kwargs) -> DatabaseBackend:
    """
    Get database backend.
    
    Args:
        backend_type: "sqlite" or "timescaledb"
        **kwargs: Backend-specific connection parameters
    
    Returns:
        DatabaseBackend instance
    """
    if backend_type == "timescaledb":
        if not PSYCOPG2_AVAILABLE:
            logger.warning("TimescaleDB requested but psycopg2 not available, falling back to SQLite")
            return SQLiteBackend(kwargs.get("db_path", "data/trading.db"))
        return TimescaleDBBackend(**kwargs)
    else:
        return SQLiteBackend(kwargs.get("db_path", "data/trading.db"))
