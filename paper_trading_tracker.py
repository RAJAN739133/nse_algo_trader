#!/usr/bin/env python3
"""
Paper Trading Performance Tracker
═══════════════════════════════════════════════════════════════════════════════

Tracks and analyzes paper trading performance to validate strategy before 
going live with real money.

Features:
- Daily P&L tracking with detailed breakdown
- Strategy performance comparison
- Win rate and risk metrics
- Drawdown analysis
- Comparison vs backtest expectations
- Recommendations for when to go live

Usage:
    python paper_trading_tracker.py              # View dashboard
    python paper_trading_tracker.py --report     # Generate weekly report
    python paper_trading_tracker.py --compare    # Compare to backtest
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

# Paths
RESULTS_DIR = Path("results")
PAPER_LOG = RESULTS_DIR / "paper_trades.json"
DAILY_LOG = RESULTS_DIR / "paper_daily.json"
BACKTEST_REFERENCE = RESULTS_DIR / "backtest_conservative_1month.csv"


class PaperTradingTracker:
    """Track and analyze paper trading performance."""
    
    def __init__(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        self.trades = self._load_trades()
        self.daily_stats = self._load_daily()
        self.starting_capital = 10000
    
    def _load_trades(self):
        if PAPER_LOG.exists():
            return json.loads(PAPER_LOG.read_text())
        return []
    
    def _load_daily(self):
        if DAILY_LOG.exists():
            return json.loads(DAILY_LOG.read_text())
        return {}
    
    def _save_trades(self):
        PAPER_LOG.write_text(json.dumps(self.trades, indent=2, default=str))
    
    def _save_daily(self):
        DAILY_LOG.write_text(json.dumps(self.daily_stats, indent=2, default=str))
    
    def log_trade(self, trade_data: dict):
        """Log a paper trade."""
        trade = {
            "id": len(self.trades) + 1,
            "date": str(date.today()),
            "timestamp": datetime.now().isoformat(),
            "symbol": trade_data.get("symbol"),
            "side": trade_data.get("side"),
            "strategy": trade_data.get("strategy"),
            "entry_price": trade_data.get("entry_price"),
            "exit_price": trade_data.get("exit_price"),
            "shares": trade_data.get("shares"),
            "gross_pnl": trade_data.get("gross_pnl"),
            "costs": trade_data.get("costs"),
            "net_pnl": trade_data.get("net_pnl"),
            "holding_minutes": trade_data.get("holding_minutes"),
            "exit_reason": trade_data.get("exit_reason"),
            "regime": trade_data.get("regime"),
            "vix": trade_data.get("vix"),
            "notes": trade_data.get("notes", "")
        }
        self.trades.append(trade)
        self._save_trades()
        self._update_daily(trade)
        return trade
    
    def _update_daily(self, trade):
        """Update daily statistics."""
        day = trade["date"]
        if day not in self.daily_stats:
            self.daily_stats[day] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "gross_pnl": 0,
                "net_pnl": 0,
                "costs": 0,
                "max_drawdown": 0,
                "peak_pnl": 0,
                "strategies": defaultdict(lambda: {"trades": 0, "pnl": 0})
            }
        
        stats = self.daily_stats[day]
        stats["trades"] += 1
        stats["gross_pnl"] += trade["gross_pnl"] or 0
        stats["net_pnl"] += trade["net_pnl"] or 0
        stats["costs"] += trade["costs"] or 0
        
        if (trade["net_pnl"] or 0) > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        
        # Track peak and drawdown
        if stats["net_pnl"] > stats["peak_pnl"]:
            stats["peak_pnl"] = stats["net_pnl"]
        drawdown = stats["peak_pnl"] - stats["net_pnl"]
        if drawdown > stats["max_drawdown"]:
            stats["max_drawdown"] = drawdown
        
        # Strategy breakdown
        strategy = trade["strategy"] or "UNKNOWN"
        if isinstance(stats["strategies"], dict):
            if strategy not in stats["strategies"]:
                stats["strategies"][strategy] = {"trades": 0, "pnl": 0}
            stats["strategies"][strategy]["trades"] += 1
            stats["strategies"][strategy]["pnl"] += trade["net_pnl"] or 0
        
        self._save_daily()
    
    def get_performance_summary(self, days=None):
        """Get overall performance summary."""
        if not self.trades:
            return None
        
        df = pd.DataFrame(self.trades)
        df["date"] = pd.to_datetime(df["date"])
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df["date"] >= cutoff]
        
        if df.empty:
            return None
        
        total_trades = len(df)
        wins = len(df[df["net_pnl"] > 0])
        losses = len(df[df["net_pnl"] <= 0])
        
        total_net_pnl = df["net_pnl"].sum()
        total_gross_pnl = df["gross_pnl"].sum()
        total_costs = df["costs"].sum()
        
        avg_win = df[df["net_pnl"] > 0]["net_pnl"].mean() if wins > 0 else 0
        avg_loss = abs(df[df["net_pnl"] <= 0]["net_pnl"].mean()) if losses > 0 else 0
        
        # Calculate daily returns for Sharpe
        daily_pnl = df.groupby("date")["net_pnl"].sum()
        sharpe = 0
        if len(daily_pnl) > 1 and daily_pnl.std() > 0:
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
        
        # Max drawdown
        cumulative = df["net_pnl"].cumsum()
        rolling_max = cumulative.expanding().max()
        drawdowns = rolling_max - cumulative
        max_drawdown = drawdowns.max()
        
        # Strategy breakdown
        strategy_perf = df.groupby("strategy").agg({
            "net_pnl": ["sum", "count", "mean"],
            "gross_pnl": "sum"
        }).round(2)
        
        return {
            "period_days": (df["date"].max() - df["date"].min()).days + 1,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades * 100 if total_trades > 0 else 0,
            "total_net_pnl": total_net_pnl,
            "total_gross_pnl": total_gross_pnl,
            "total_costs": total_costs,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else 0,
            "profit_factor": abs(df[df["net_pnl"] > 0]["net_pnl"].sum() / df[df["net_pnl"] <= 0]["net_pnl"].sum()) if losses > 0 else float('inf'),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "return_pct": total_net_pnl / self.starting_capital * 100,
            "strategy_breakdown": strategy_perf.to_dict() if not strategy_perf.empty else {}
        }
    
    def compare_to_backtest(self):
        """Compare paper trading results to backtest."""
        if not BACKTEST_REFERENCE.exists():
            return {"error": "No backtest reference found"}
        
        backtest = pd.read_csv(BACKTEST_REFERENCE)
        paper = self.get_performance_summary()
        
        if not paper:
            return {"error": "No paper trades to compare"}
        
        # Normalize to same number of days
        bt_days = len(backtest["date"].unique()) if "date" in backtest.columns else 22
        paper_days = paper["period_days"]
        
        bt_daily_return = backtest["net_pnl"].sum() / bt_days if bt_days > 0 else 0
        paper_daily_return = paper["total_net_pnl"] / paper_days if paper_days > 0 else 0
        
        return {
            "backtest": {
                "days": bt_days,
                "total_trades": len(backtest),
                "net_pnl": backtest["net_pnl"].sum(),
                "daily_avg": bt_daily_return,
                "win_rate": len(backtest[backtest["net_pnl"] > 0]) / len(backtest) * 100
            },
            "paper": {
                "days": paper_days,
                "total_trades": paper["total_trades"],
                "net_pnl": paper["total_net_pnl"],
                "daily_avg": paper_daily_return,
                "win_rate": paper["win_rate"]
            },
            "comparison": {
                "pnl_ratio": paper_daily_return / bt_daily_return if bt_daily_return > 0 else 0,
                "trade_freq_ratio": (paper["total_trades"] / paper_days) / (len(backtest) / bt_days) if bt_days > 0 else 0,
                "win_rate_diff": paper["win_rate"] - (len(backtest[backtest["net_pnl"] > 0]) / len(backtest) * 100),
            },
            "verdict": self._get_verdict(paper_daily_return, bt_daily_return)
        }
    
    def _get_verdict(self, paper_daily, bt_daily):
        """Get recommendation based on comparison."""
        ratio = paper_daily / bt_daily if bt_daily > 0 else 0
        
        if ratio >= 0.8:
            return "✅ GOOD: Paper trading within 80% of backtest. Consider scaling up."
        elif ratio >= 0.5:
            return "⚠️ CAUTION: Paper at 50-80% of backtest. Investigate slippage/execution."
        elif ratio >= 0.2:
            return "🔶 CONCERN: Paper significantly underperforming. Review strategy."
        elif ratio > 0:
            return "🔴 POOR: Major gap between backtest and paper. Do not go live yet."
        else:
            return "❌ LOSING: Paper trading is negative. Continue paper trading, review strategy."
    
    def go_live_readiness(self):
        """Check if ready to go live."""
        summary = self.get_performance_summary()
        
        if not summary:
            return {
                "ready": False,
                "reason": "No paper trades yet. Trade for at least 2 weeks.",
                "checklist": self._get_checklist(None)
            }
        
        checklist = self._get_checklist(summary)
        passed = sum(1 for item in checklist.values() if item["passed"])
        total = len(checklist)
        
        return {
            "ready": passed >= total * 0.8,  # 80% of checks must pass
            "score": f"{passed}/{total}",
            "checklist": checklist,
            "recommendation": self._get_recommendation(summary, passed, total)
        }
    
    def _get_checklist(self, summary):
        """Generate go-live checklist."""
        if not summary:
            return {
                "min_trades": {"required": "≥30 trades", "actual": "0", "passed": False},
                "min_days": {"required": "≥10 trading days", "actual": "0", "passed": False},
                "positive_pnl": {"required": "Net profit > 0", "actual": "N/A", "passed": False},
                "win_rate": {"required": "≥30%", "actual": "N/A", "passed": False},
                "max_drawdown": {"required": "<10% of capital", "actual": "N/A", "passed": False},
                "profit_factor": {"required": ">1.2", "actual": "N/A", "passed": False},
                "consistency": {"required": "≥60% profitable days", "actual": "N/A", "passed": False},
            }
        
        # Calculate profitable days
        daily_pnl = {}
        for trade in self.trades:
            day = trade["date"]
            daily_pnl[day] = daily_pnl.get(day, 0) + (trade["net_pnl"] or 0)
        profitable_days = sum(1 for pnl in daily_pnl.values() if pnl > 0)
        total_days = len(daily_pnl)
        consistency = profitable_days / total_days * 100 if total_days > 0 else 0
        
        return {
            "min_trades": {
                "required": "≥30 trades",
                "actual": str(summary["total_trades"]),
                "passed": summary["total_trades"] >= 30
            },
            "min_days": {
                "required": "≥10 trading days",
                "actual": str(summary["period_days"]),
                "passed": summary["period_days"] >= 10
            },
            "positive_pnl": {
                "required": "Net profit > 0",
                "actual": f"₹{summary['total_net_pnl']:.0f}",
                "passed": summary["total_net_pnl"] > 0
            },
            "win_rate": {
                "required": "≥30%",
                "actual": f"{summary['win_rate']:.1f}%",
                "passed": summary["win_rate"] >= 30
            },
            "max_drawdown": {
                "required": "<10% of capital",
                "actual": f"₹{summary['max_drawdown']:.0f} ({summary['max_drawdown']/self.starting_capital*100:.1f}%)",
                "passed": summary["max_drawdown"] < self.starting_capital * 0.10
            },
            "profit_factor": {
                "required": ">1.2",
                "actual": f"{summary['profit_factor']:.2f}",
                "passed": summary["profit_factor"] > 1.2
            },
            "consistency": {
                "required": "≥60% profitable days",
                "actual": f"{consistency:.0f}% ({profitable_days}/{total_days} days)",
                "passed": consistency >= 60
            },
        }
    
    def _get_recommendation(self, summary, passed, total):
        """Get go-live recommendation."""
        pct = passed / total * 100
        
        if pct >= 85:
            return "🟢 READY: You've met most criteria. Consider starting with 50% capital."
        elif pct >= 70:
            return "🟡 ALMOST: Close to ready. Address failing items for 1 more week."
        elif pct >= 50:
            return "🟠 NOT YET: Need more paper trading. Focus on consistency."
        else:
            return "🔴 CONTINUE PAPER: Significant improvements needed. Review strategy."
    
    def print_dashboard(self):
        """Print performance dashboard."""
        summary = self.get_performance_summary()
        
        print("\n" + "═" * 70)
        print("  📊 PAPER TRADING PERFORMANCE DASHBOARD")
        print("═" * 70)
        
        if not summary:
            print("\n  No trades recorded yet. Run paper trading first.\n")
            return
        
        print(f"""
  📅 Period: {summary['period_days']} trading days
  💰 Starting Capital: ₹{self.starting_capital:,}
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ OVERALL PERFORMANCE                                             │
  ├─────────────────────────────────────────────────────────────────┤
  │ Total Trades:     {summary['total_trades']:>6}     │  Net P&L:    ₹{summary['total_net_pnl']:>+10,.0f}  │
  │ Wins/Losses:   {summary['wins']:>3}/{summary['losses']:<3}     │  Return:      {summary['return_pct']:>+9.2f}%  │
  │ Win Rate:       {summary['win_rate']:>6.1f}%     │  Sharpe:       {summary['sharpe_ratio']:>+8.2f}  │
  └─────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ RISK METRICS                                                    │
  ├─────────────────────────────────────────────────────────────────┤
  │ Avg Win:      ₹{summary['avg_win']:>8,.0f}     │  Max Drawdown: ₹{summary['max_drawdown']:>8,.0f}  │
  │ Avg Loss:     ₹{summary['avg_loss']:>8,.0f}     │  Profit Factor:  {summary['profit_factor']:>8.2f}  │
  │ Win/Loss:        {summary['win_loss_ratio']:>5.2f}x     │  Costs:        ₹{summary['total_costs']:>8,.0f}  │
  └─────────────────────────────────────────────────────────────────┘
""")
        
        # Go-live readiness
        readiness = self.go_live_readiness()
        print("  ┌─────────────────────────────────────────────────────────────────┐")
        print("  │ GO-LIVE READINESS CHECKLIST                                     │")
        print("  ├─────────────────────────────────────────────────────────────────┤")
        for name, check in readiness["checklist"].items():
            status = "✅" if check["passed"] else "❌"
            print(f"  │ {status} {name:<15} {check['required']:<15} → {check['actual']:<15} │")
        print("  ├─────────────────────────────────────────────────────────────────┤")
        print(f"  │ Score: {readiness['score']:<57} │")
        print(f"  │ {readiness['recommendation']:<63} │")
        print("  └─────────────────────────────────────────────────────────────────┘\n")


def import_from_csv(csv_path: str):
    """Import trades from CSV (e.g., from backtest results)."""
    tracker = PaperTradingTracker()
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        tracker.log_trade({
            "symbol": row.get("symbol"),
            "side": row.get("side"),
            "strategy": row.get("strategy") or row.get("trade_type"),
            "entry_price": row.get("entry_price") or row.get("entry"),
            "exit_price": row.get("exit_price") or row.get("exit"),
            "shares": row.get("shares") or row.get("qty"),
            "gross_pnl": row.get("gross_pnl"),
            "costs": row.get("costs") or row.get("transaction_costs"),
            "net_pnl": row.get("net_pnl"),
            "holding_minutes": row.get("holding_minutes"),
            "exit_reason": row.get("exit_reason"),
            "regime": row.get("regime"),
            "vix": row.get("vix"),
        })
    
    print(f"Imported {len(df)} trades from {csv_path}")
    tracker.print_dashboard()


if __name__ == "__main__":
    import sys
    
    tracker = PaperTradingTracker()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--report":
            summary = tracker.get_performance_summary(days=7)
            if summary:
                print(json.dumps(summary, indent=2, default=str))
            else:
                print("No trades in last 7 days")
        
        elif sys.argv[1] == "--compare":
            comparison = tracker.compare_to_backtest()
            print(json.dumps(comparison, indent=2, default=str))
        
        elif sys.argv[1] == "--import" and len(sys.argv) > 2:
            import_from_csv(sys.argv[2])
        
        else:
            print("Usage:")
            print("  python paper_trading_tracker.py           # View dashboard")
            print("  python paper_trading_tracker.py --report  # Weekly report")
            print("  python paper_trading_tracker.py --compare # Compare to backtest")
            print("  python paper_trading_tracker.py --import <csv>  # Import trades")
    else:
        tracker.print_dashboard()
