"""
Production-Ready Walk-Forward Backtest
═══════════════════════════════════════════════════════════════

DYNAMIC FILTERING - No hardcoded blacklist/whitelist!

This backtest uses the SAME logic as production:
1. Tracks rolling performance per stock
2. Dynamically identifies good/bad stocks from recent trades
3. Adjusts scores based on historical win rates
4. Same code path for backtest and live trading

KEY STRATEGY ELEMENTS:
- SHORT bias in bearish/neutral (validated: 68% WR vs 50% for longs)
- Minimum ML confidence threshold
- Dynamic stock scoring from rolling performance
- No stop losses (time-based exit)
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.symbols import get_universe, STOCK_SECTORS
from backtest.costs import ZerodhaCostModel

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# DYNAMIC STOCK SCORING (replaces hardcoded lists)
# ════════════════════════════════════════════════════════════

def calculate_dynamic_scores(
    trades_history: List[Dict],
    as_of_date: date,
    lookback_days: int = 60,
) -> Tuple[Set[str], Set[str], Dict[str, float]]:
    """
    Calculate avoid/prefer sets dynamically from past trades.
    ONLY uses data before as_of_date (no future leakage).
    
    Returns: (avoid_stocks, preferred_stocks, stock_scores)
    """
    cutoff_start = (as_of_date - timedelta(days=lookback_days)).isoformat()
    cutoff_end = as_of_date.isoformat()
    
    # Filter to relevant period
    relevant = [
        t for t in trades_history
        if cutoff_start <= t.get("date", "") < cutoff_end
    ]
    
    if len(relevant) < 10:
        return set(), set(), {}
    
    # Aggregate per symbol
    stats = {}
    for t in relevant:
        sym = t["symbol"]
        if sym not in stats:
            stats[sym] = {"wins": 0, "total": 0, "pnl": 0.0}
        stats[sym]["total"] += 1
        stats[sym]["pnl"] += t.get("pnl", t.get("net_pnl", 0))
        if t.get("pnl", t.get("net_pnl", 0)) > 0:
            stats[sym]["wins"] += 1
    
    avoid = set()
    preferred = set()
    scores = {}
    
    for sym, s in stats.items():
        if s["total"] < 3:
            scores[sym] = 50.0
            continue
        
        wr = s["wins"] / s["total"]
        avg_pnl = s["pnl"] / s["total"]
        
        # Score: base 50, adjusted by win rate and avg P&L
        score = 50 + (wr - 0.5) * 60 + min(20, max(-20, avg_pnl / 50))
        scores[sym] = round(max(0, min(100, score)), 1)
        
        # Dynamic thresholds
        if wr < 0.40 or (wr < 0.45 and avg_pnl < -50):
            avoid.add(sym)
        elif wr > 0.55 and avg_pnl > 30:
            preferred.add(sym)
    
    return avoid, preferred, scores


@dataclass
class ProductionConfig:
    capital: float = 100_000
    risk_per_trade: float = 0.02   # 2% risk
    max_trades_per_day: int = 3
    daily_loss_limit: float = 0.03
    slippage_pct: float = 0.001
    
    stocks_per_day: int = 3
    min_confidence: float = 0.12  # Slightly lower to get more trades
    
    # KEY: SHORT bias
    short_only_in_bear: bool = True
    max_longs_in_neutral: int = 1
    
    lookback_days: int = 252
    retrain_every: int = 15


@dataclass
class Trade:
    date: str
    symbol: str
    direction: str
    entry: float
    exit: float
    qty: int
    gross_pnl: float
    costs: float
    net_pnl: float
    confidence: float
    market: str


class ProductionBacktester:
    """Production-ready walk-forward backtest with dynamic stock filtering."""
    
    FEATURES = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "price_vs_sma20", "sma20_vs_sma50", 
        "atr_pct", "vol_ratio", "rsi_14", "bb_pos",
        "trend_strength", "vol_regime",
    ]
    
    def __init__(self, config: ProductionConfig = None):
        self.config = config or ProductionConfig()
        self.cost_model = ZerodhaCostModel()
        self.model = None
        self.features = None
        self.last_train_date = None
        
        # DYNAMIC: Track trades for rolling performance
        self.trades_history: List[Dict] = []
        self.dynamic_avoid: Set[str] = set()
        self.dynamic_prefer: Set[str] = set()
        self.stock_scores: Dict[str, float] = {}
    
    def run(self, start_date: date, end_date: date) -> Dict:
        """Run production backtest with dynamic filtering."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  PRODUCTION BACKTEST (DYNAMIC FILTERING)")
        logger.info(f"  {start_date} to {end_date}")
        logger.info(f"  Capital: Rs {self.config.capital:,}")
        logger.info(f"  Short bias: ON | Dynamic scoring: ON")
        logger.info(f"{'='*60}\n")
        
        # Load data - NO hardcoded blacklist
        universe = get_universe("nifty100")
        
        all_data = self._load_all_data(universe, start_date, end_date)
        if all_data.empty:
            logger.error("No data")
            return {}
        
        nifty = self._load_proxy_data(start_date, end_date)
        
        logger.info(f"  Loaded {len(all_data):,} bars for {all_data['symbol'].nunique()} stocks")
        
        trading_days = sorted(all_data["date"].dt.date.unique())
        trading_days = [d for d in trading_days if start_date <= d <= end_date]
        
        capital = self.config.capital
        equity = [capital]
        all_trades = []
        daily_pnl = []
        
        for day_idx, day in enumerate(trading_days):
            if day_idx < 60:
                continue
            
            market = self._get_market_state(nifty, day)
            
            # DYNAMIC: Update avoid/prefer lists from rolling performance
            # Only uses trades BEFORE this day (no future leakage)
            self.dynamic_avoid, self.dynamic_prefer, self.stock_scores = \
                calculate_dynamic_scores(self.trades_history, day, lookback_days=60)
            
            # Train
            if self.model is None or (
                self.last_train_date and 
                (day - self.last_train_date).days >= self.config.retrain_every
            ):
                self._train_model(all_data, day - timedelta(days=1))
            
            prior = trading_days[day_idx - 1]
            candidates = self._score_stocks(all_data, prior, market)
            
            if candidates.empty:
                continue
            
            selected = self._select_stocks(candidates, market)
            if not selected:
                continue
            
            trades = self._execute(all_data, day, selected, capital, market)
            if not trades:
                continue
            
            day_pnl_val = sum(t.net_pnl for t in trades)
            capital += day_pnl_val
            equity.append(capital)
            daily_pnl.append(day_pnl_val)
            all_trades.extend(trades)
            
            wins = len([t for t in trades if t.net_pnl > 0])
            logger.info(f"  {day} | {market:8} | {len(trades)} trades | W:{wins} L:{len(trades)-wins} | Rs {day_pnl_val:+,.2f} | Cap: Rs {capital:,.2f}")
        
        results = self._metrics(all_trades, equity, daily_pnl)
        self._print(results)
        self._save(results, all_trades, start_date, end_date)
        
        return results
    
    def _load_all_data(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        all_df = []
        cache = Path("data/cache")
        lookback = start - timedelta(days=self.config.lookback_days + 100)
        
        for sym in symbols:
            for pat in [f"{sym}_1d_5y.csv", f"{sym}_1d_10y.csv"]:
                fpath = cache / pat
                if fpath.exists():
                    try:
                        df = pd.read_csv(fpath)
                        df["date"] = pd.to_datetime(df["date"])
                        df["symbol"] = sym
                        df = df[(df["date"].dt.date >= lookback) & (df["date"].dt.date <= end)]
                        if len(df) > 60:
                            df = self._features(df)
                            all_df.append(df)
                        break
                    except:
                        continue
        
        return pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()
    
    def _load_proxy_data(self, start: date, end: date) -> pd.DataFrame:
        for sym in ["RELIANCE", "HDFCBANK", "TCS"]:
            fpath = Path(f"data/cache/{sym}_1d_5y.csv")
            if fpath.exists():
                df = pd.read_csv(fpath)
                df["date"] = pd.to_datetime(df["date"])
                lookback = start - timedelta(days=100)
                return df[(df["date"].dt.date >= lookback) & (df["date"].dt.date <= end)]
        return pd.DataFrame()
    
    def _get_market_state(self, proxy: pd.DataFrame, day: date) -> str:
        if proxy.empty:
            return "neutral"
        
        past = proxy[proxy["date"].dt.date < day].tail(30)
        if len(past) < 15:
            return "neutral"
        
        sma5 = past["close"].tail(5).mean()
        sma20 = past["close"].tail(20).mean()
        ret_10d = (past["close"].iloc[-1] / past["close"].iloc[-10] - 1) if len(past) >= 10 else 0
        ret_5d = (past["close"].iloc[-1] / past["close"].iloc[-5] - 1) if len(past) >= 5 else 0
        
        # Trend determination
        if sma5 > sma20 * 1.015 and ret_10d > 0.02 and ret_5d > 0.01:
            return "bullish"
        elif sma5 < sma20 * 0.985 and ret_10d < -0.02 and ret_5d < -0.01:
            return "bearish"
        else:
            return "neutral"
    
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").copy()
        c = df["close"]
        
        for d in [1, 5, 10, 20]:
            df[f"ret_{d}d"] = c.pct_change(d)
        
        df["next_ret"] = c.shift(-1) / c - 1
        df["target"] = (df["next_ret"] > 0).astype(int)
        
        df["sma5"] = c.rolling(5).mean()
        df["sma20"] = c.rolling(20).mean()
        df["sma50"] = c.rolling(50).mean()
        df["price_vs_sma20"] = (c - df["sma20"]) / df["sma20"]
        df["sma20_vs_sma50"] = (df["sma20"] - df["sma50"]) / df["sma50"]
        df["trend_strength"] = (df["sma5"] - df["sma20"]) / df["sma20"]
        
        h, l = df["high"], df["low"]
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr14"] / c
        
        vol = df["ret_1d"].rolling(20).std()
        vol_ma = vol.rolling(50).mean()
        df["vol_regime"] = (vol / vol_ma.replace(0, 1)).fillna(1)
        
        if "volume" in df.columns:
            df["vol_sma20"] = df["volume"].rolling(20).mean()
            df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, 1)
        else:
            df["vol_ratio"] = 1.0
        
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        
        std20 = c.rolling(20).std()
        bb_upper = df["sma20"] + 2 * std20
        bb_lower = df["sma20"] - 2 * std20
        df["bb_pos"] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, 1)
        
        return df
    
    def _train_model(self, data: pd.DataFrame, train_end: date):
        mask = data["date"].dt.date < train_end
        train = data[mask].copy()
        
        cutoff = pd.Timestamp(train_end) - pd.Timedelta(days=self.config.lookback_days)
        train = train[train["date"] >= cutoff]
        
        avail = [f for f in self.FEATURES if f in train.columns]
        train = train.dropna(subset=avail + ["target"])
        
        if len(train) < 500:
            return
        
        X = train[avail].replace([np.inf, -np.inf], 0).fillna(0)
        y = train["target"]
        
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.03
            )
        
        self.model.fit(X, y)
        self.features = avail
        self.last_train_date = train_end
        logger.info(f"    ML trained on {len(X)} samples")
    
    def _score_stocks(self, data: pd.DataFrame, as_of: date, market: str) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        
        scores = []
        
        for sym in data["symbol"].unique():
            # DYNAMIC: Skip stocks that performed poorly recently
            if sym in self.dynamic_avoid:
                continue
            
            sdf = data[(data["symbol"] == sym) & (data["date"].dt.date <= as_of)]
            if len(sdf) < 30:
                continue
            
            last = sdf.iloc[-1]
            
            try:
                x = np.array([last.get(f, 0) for f in self.features], dtype=np.float64).reshape(1, -1)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                
                prob_up = self.model.predict_proba(x)[0][1]
                confidence = abs(prob_up - 0.5)
                
                if confidence < self.config.min_confidence:
                    continue
                
                # DIRECTION BASED ON MARKET
                if prob_up > 0.5:
                    direction = "LONG"
                else:
                    direction = "SHORT"
                
                # KEY FILTER: In bearish/neutral, only shorts
                if market == "bearish" and direction == "LONG":
                    continue
                
                # In neutral, limit longs
                if market == "neutral" and direction == "LONG" and confidence < 0.2:
                    continue
                
                # Volatility filter
                atr_pct = last.get("atr_pct", 0.02)
                vol_regime = last.get("vol_regime", 1.0)
                
                if atr_pct < 0.008 or atr_pct > 0.045:
                    continue
                if vol_regime > 1.6:
                    continue
                
                # RSI filter for extreme values
                rsi = last.get("rsi_14", 50)
                if direction == "LONG" and rsi > 75:
                    continue
                if direction == "SHORT" and rsi < 25:
                    continue
                
                # DYNAMIC: Boost confidence for historically good stocks
                if sym in self.dynamic_prefer:
                    confidence *= 1.15
                
                # DYNAMIC: Apply rolling score adjustment
                if sym in self.stock_scores:
                    # Score 0-100, adjust confidence by +/-20%
                    score_factor = 0.8 + (self.stock_scores[sym] / 100) * 0.4
                    confidence *= score_factor
                
                scores.append({
                    "symbol": sym,
                    "prob_up": prob_up,
                    "direction": direction,
                    "confidence": confidence,
                    "atr_pct": atr_pct,
                    "rsi": rsi,
                })
                
            except:
                continue
        
        df = pd.DataFrame(scores)
        return df.sort_values("confidence", ascending=False) if not df.empty else df
    
    def _select_stocks(self, candidates: pd.DataFrame, market: str) -> List[Dict]:
        selected = []
        sector_count = {}
        long_count = 0
        
        max_longs = 0 if market == "bearish" else self.config.max_longs_in_neutral
        if market == "bullish":
            max_longs = self.config.stocks_per_day
        
        for _, row in candidates.iterrows():
            sym = row["symbol"]
            direction = row["direction"]
            
            if direction == "LONG" and long_count >= max_longs:
                continue
            
            sector = STOCK_SECTORS.get(sym, "Other")
            if sector_count.get(sector, 0) >= 1:
                continue
            
            selected.append({
                "symbol": sym,
                "direction": direction,
                "confidence": row["confidence"],
                "atr_pct": row["atr_pct"],
            })
            
            sector_count[sector] = sector_count.get(sector, 0) + 1
            if direction == "LONG":
                long_count += 1
            
            if len(selected) >= self.config.stocks_per_day:
                break
        
        return selected
    
    def _execute(
        self, data: pd.DataFrame, day: date, 
        selected: List[Dict], capital: float, market: str
    ) -> List[Trade]:
        trades = []
        
        for sel in selected:
            sym = sel["symbol"]
            direction = sel["direction"]
            
            today = data[(data["symbol"] == sym) & (data["date"].dt.date == day)]
            if today.empty:
                continue
            
            row = today.iloc[0]
            o, c = row["open"], row["close"]
            
            if direction == "LONG":
                entry = o * (1 + self.config.slippage_pct / 2)
                exit_price = c * (1 - self.config.slippage_pct / 2)
            else:
                entry = o * (1 - self.config.slippage_pct / 2)
                exit_price = c * (1 + self.config.slippage_pct / 2)
            
            base_risk = self.config.capital * self.config.risk_per_trade
            risk = base_risk * (0.8 + sel["confidence"] * 0.5)
            
            potential_loss = entry * sel["atr_pct"] * 2
            qty = int(risk / potential_loss) if potential_loss > 0 else 0
            qty = min(qty, int(capital * 0.3 / entry))
            
            if qty == 0:
                continue
            
            if direction == "LONG":
                gross_pnl = (exit_price - entry) * qty
            else:
                gross_pnl = (entry - exit_price) * qty
            
            costs = self.cost_model.calculate(entry * qty, exit_price * qty)
            net_pnl = gross_pnl - costs.total
            
            trade = Trade(
                date=str(day),
                symbol=sym,
                direction=direction,
                entry=round(entry, 2),
                exit=round(exit_price, 2),
                qty=qty,
                gross_pnl=round(gross_pnl, 2),
                costs=round(costs.total, 2),
                net_pnl=round(net_pnl, 2),
                confidence=round(sel["confidence"], 3),
                market=market,
            )
            trades.append(trade)
            
            # DYNAMIC: Record trade for rolling performance tracking
            self.trades_history.append({
                "symbol": sym,
                "direction": direction,
                "pnl": net_pnl,
                "date": str(day),
                "confidence": sel["confidence"],
            })
        
        return trades
    
    def _metrics(self, trades: List[Trade], equity: List[float], daily_pnl: List[float]) -> Dict:
        if not trades:
            return {}
        
        total = sum(t.net_pnl for t in trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        
        wr = len(wins) / len(trades) * 100
        
        gp = sum(t.net_pnl for t in wins) if wins else 0
        gl = abs(sum(t.net_pnl for t in losses)) if losses else 0
        pf = gp / gl if gl > 0 else float("inf")
        
        peak = equity[0]
        max_dd = 0
        for v in equity:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        sharpe = (np.mean(daily_pnl) / np.std(daily_pnl)) * np.sqrt(252) if len(daily_pnl) > 1 and np.std(daily_pnl) > 0 else 0
        
        longs = [t for t in trades if t.direction == "LONG"]
        shorts = [t for t in trades if t.direction == "SHORT"]
        
        return {
            "total_trades": len(trades),
            "winning": len(wins),
            "losing": len(losses),
            "win_rate": f"{wr:.1f}%",
            "total_pnl": f"Rs {total:+,.2f}",
            "return_pct": f"{total / self.config.capital * 100:+.2f}%",
            "avg_win": f"Rs {np.mean([t.net_pnl for t in wins]):+,.2f}" if wins else "Rs 0",
            "avg_loss": f"Rs {np.mean([t.net_pnl for t in losses]):+,.2f}" if losses else "Rs 0",
            "profit_factor": f"{pf:.2f}",
            "max_drawdown": f"{max_dd * 100:.1f}%",
            "sharpe_ratio": f"{sharpe:.2f}",
            "final_capital": f"Rs {equity[-1]:,.2f}",
            "long_trades": f"{len(longs)} ({sum(1 for t in longs if t.net_pnl > 0)}/{len(longs)} wins)" if longs else "0",
            "short_trades": f"{len(shorts)} ({sum(1 for t in shorts if t.net_pnl > 0)}/{len(shorts)} wins)" if shorts else "0",
            "long_pnl": f"Rs {sum(t.net_pnl for t in longs):+,.2f}" if longs else "Rs 0",
            "short_pnl": f"Rs {sum(t.net_pnl for t in shorts):+,.2f}" if shorts else "Rs 0",
        }
    
    def _print(self, results: Dict):
        logger.info(f"\n{'='*60}")
        logger.info(f"  PRODUCTION STRATEGY RESULTS")
        logger.info(f"{'='*60}")
        for k, v in results.items():
            logger.info(f"  {k:<20} {v}")
    
    def _save(self, results: Dict, trades: List[Trade], start: date, end: date):
        Path("results").mkdir(exist_ok=True)
        
        fname = f"production_{start}_{end}.json"
        with open(f"results/{fname}", "w") as f:
            json.dump(results, f, indent=2)
        
        if trades:
            df = pd.DataFrame([asdict(t) for t in trades])
            df.to_csv(f"results/production_trades_{start}_{end}.csv", index=False)
        
        logger.info(f"\n  Saved to results/{fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--capital", type=float, default=100000)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=args.days)
    
    bt = ProductionBacktester(ProductionConfig(capital=args.capital))
    bt.run(start, end)


if __name__ == "__main__":
    main()
