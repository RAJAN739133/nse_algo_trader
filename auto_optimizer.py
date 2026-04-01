"""
Auto Strategy Optimizer
═══════════════════════
Analyzes daily trade results and auto-tunes ProStrategyV2 parameters
so the bot improves tomorrow based on today's performance.

What it does:
  1. Reads trade CSVs from results/
  2. Analyzes win rate, SL hit rate, target hit rate by trade type
  3. Computes optimal parameter adjustments
  4. Writes tuned params to config/auto_tuned.yaml
  5. ProStrategyV2 loads auto_tuned.yaml at startup if present

Runs automatically after market close (via scheduler).

Usage:
    python auto_optimizer.py analyze       # Analyze last 7 days
    python auto_optimizer.py optimize      # Analyze + write new params
    python auto_optimizer.py reset         # Reset to default params
    python auto_optimizer.py report        # Print optimization report
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
TUNED_CONFIG = Path("config/auto_tuned.yaml")
OPTIMIZATION_LOG = Path("logs/optimization_history.json")

# Default V2 params (baseline)
DEFAULT_PARAMS = {
    "orb_candles": 3,
    "min_breakout_buffer_pct": 0.003,
    "volume_spike_mult": 1.3,
    "base_rr_ratio": 1.5,
    "high_vol_rr_ratio": 2.0,
    "max_gap_pct": 0.015,
    "max_orb_to_adr_ratio": 0.60,
    "vwap_rsi_overbought": 73,
    "vwap_rsi_oversold": 27,
    "vwap_entry_sigma": 2.0,
    "max_risk_pct": 0.02,
    "cooldown_candles": 6,
    "time_decay_candles": 18,
    "partial_exit_at_rr": 1.0,
    "partial_exit_pct": 0.50,
    "require_retest": True,
    "momentum_threshold": 0.005,
}

# Param bounds (min, max, step)
PARAM_BOUNDS = {
    "min_breakout_buffer_pct": (0.001, 0.006, 0.001),
    "volume_spike_mult":       (1.0, 2.0, 0.1),
    "base_rr_ratio":           (1.0, 2.5, 0.25),
    "high_vol_rr_ratio":       (1.5, 3.0, 0.25),
    "max_orb_to_adr_ratio":    (0.40, 0.80, 0.05),
    "vwap_rsi_overbought":     (65, 80, 2),
    "vwap_rsi_oversold":       (20, 35, 2),
    "vwap_entry_sigma":        (1.5, 2.5, 0.25),
    "max_risk_pct":            (0.01, 0.03, 0.005),
    "cooldown_candles":        (3, 10, 1),
    "time_decay_candles":      (12, 24, 2),
    "partial_exit_at_rr":      (0.5, 1.5, 0.25),
    "partial_exit_pct":        (0.30, 0.70, 0.10),
    "momentum_threshold":      (0.002, 0.008, 0.001),
}


def load_recent_trades(days=7):
    """Load trade results from last N days."""
    all_trades = []
    for i in range(days):
        d = date.today() - timedelta(days=i)
        for prefix in ["live_v3_", "live_v2_", "paper_"]:
            f = RESULTS_DIR / f"{prefix}{d}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["trade_date"] = str(d)
                all_trades.append(df)
                break  # Only load one per day

    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True)


def analyze_trades(trades_df):
    """Compute performance metrics by trade type."""
    if trades_df.empty:
        return {}

    analysis = {
        "total_trades": len(trades_df),
        "total_pnl": float(trades_df["net_pnl"].sum()),
        "win_rate": float((trades_df["net_pnl"] > 0).mean()),
        "avg_pnl": float(trades_df["net_pnl"].mean()),
        "avg_win": float(trades_df[trades_df["net_pnl"] > 0]["net_pnl"].mean()) if (trades_df["net_pnl"] > 0).any() else 0,
        "avg_loss": float(trades_df[trades_df["net_pnl"] <= 0]["net_pnl"].mean()) if (trades_df["net_pnl"] <= 0).any() else 0,
    }

    # By trade type
    if "type" in trades_df.columns:
        for ttype in trades_df["type"].unique():
            sub = trades_df[trades_df["type"] == ttype]
            analysis[f"{ttype}_count"] = len(sub)
            analysis[f"{ttype}_win_rate"] = float((sub["net_pnl"] > 0).mean())
            analysis[f"{ttype}_avg_pnl"] = float(sub["net_pnl"].mean())

    # By exit reason
    if "reason" in trades_df.columns:
        for reason in trades_df["reason"].unique():
            sub = trades_df[trades_df["reason"] == reason]
            analysis[f"exit_{reason}_count"] = len(sub)
            analysis[f"exit_{reason}_pct"] = float(len(sub) / len(trades_df))

    # By direction
    if "direction" in trades_df.columns:
        for d in ["LONG", "SHORT"]:
            sub = trades_df[trades_df["direction"] == d]
            if len(sub) > 0:
                analysis[f"{d}_count"] = len(sub)
                analysis[f"{d}_win_rate"] = float((sub["net_pnl"] > 0).mean())

    return analysis


def compute_adjustments(analysis, current_params=None):
    """
    Based on trade analysis, compute parameter adjustments.

    Logic:
      - High SL hit rate → widen stops (increase buffer, reduce risk)
      - Low win rate on ORB → tighten filters (increase volume mult)
      - Time decay exits high → reduce time_decay_candles
      - Mostly losing on VWAP → widen VWAP bands
      - LONG losing, SHORT winning → lower RSI overbought threshold
      - etc.
    """
    params = dict(current_params or DEFAULT_PARAMS)
    reasons = []

    if not analysis or analysis.get("total_trades", 0) < 3:
        return params, ["Not enough trades (< 3) to optimize"]

    win_rate = analysis.get("win_rate", 0.5)
    sl_rate = analysis.get("exit_STOP_LOSS_count", 0) + analysis.get("exit_STOP_LOSS_RT_count", 0)
    total = analysis.get("total_trades", 1)
    sl_pct = sl_rate / total if total > 0 else 0

    # ── Stop Loss too tight? ──
    if sl_pct > 0.5:
        # More than 50% trades hit SL → widen buffer
        old = params["min_breakout_buffer_pct"]
        params["min_breakout_buffer_pct"] = _nudge_up(old, "min_breakout_buffer_pct")
        params["max_risk_pct"] = _nudge_up(params["max_risk_pct"], "max_risk_pct")
        reasons.append(f"SL hit rate {sl_pct:.0%} > 50% → widened buffer {old} → {params['min_breakout_buffer_pct']}")

    # ── Win rate too low? ──
    if win_rate < 0.35:
        # Tighten entry filters
        params["volume_spike_mult"] = _nudge_up(params["volume_spike_mult"], "volume_spike_mult")
        reasons.append(f"Win rate {win_rate:.0%} < 35% → raised volume filter")

    # ── ORB breakouts failing? ──
    orb_wr = analysis.get("ORB_BREAKOUT_win_rate", analysis.get("ORB_BREAKDOWN_win_rate", 0.5))
    if orb_wr < 0.30:
        params["max_orb_to_adr_ratio"] = _nudge_down(params["max_orb_to_adr_ratio"], "max_orb_to_adr_ratio")
        reasons.append(f"ORB win rate {orb_wr:.0%} < 30% → tightened ADR filter")

    # ── VWAP trades failing? ──
    vwap_wr = analysis.get("VWAP_OVERSOLD_win_rate", analysis.get("VWAP_OVERBOUGHT_win_rate", 0.5))
    if vwap_wr < 0.30:
        params["vwap_entry_sigma"] = _nudge_up(params["vwap_entry_sigma"], "vwap_entry_sigma")
        reasons.append(f"VWAP win rate {vwap_wr:.0%} < 30% → widened VWAP bands")

    # ── Time decay exits too high? ──
    td_pct = analysis.get("exit_TIME_DECAY_pct", 0)
    if td_pct > 0.3:
        params["time_decay_candles"] = _nudge_down(params["time_decay_candles"], "time_decay_candles")
        reasons.append(f"Time decay exits {td_pct:.0%} > 30% → earlier exit")

    # ── R:R too aggressive or conservative? ──
    if win_rate > 0.55 and analysis.get("avg_win", 0) < abs(analysis.get("avg_loss", 0)) * 0.8:
        # High win rate but winners too small → increase R:R
        params["base_rr_ratio"] = _nudge_up(params["base_rr_ratio"], "base_rr_ratio")
        reasons.append(f"Win rate ok ({win_rate:.0%}) but avg_win < avg_loss → raised R:R")

    # ── LONG vs SHORT bias ──
    long_wr = analysis.get("LONG_win_rate", 0.5)
    short_wr = analysis.get("SHORT_win_rate", 0.5)
    if long_wr < 0.25 and short_wr > 0.5:
        params["vwap_rsi_oversold"] = _nudge_down(params["vwap_rsi_oversold"], "vwap_rsi_oversold")
        reasons.append(f"LONG struggling ({long_wr:.0%}), SHORT doing well ({short_wr:.0%}) → stricter long RSI")

    # ── Win rate high, keep as is ──
    if win_rate > 0.55 and not reasons:
        reasons.append(f"Win rate {win_rate:.0%} — strategy performing well, minimal adjustments")

    return params, reasons


def _nudge_up(val, param_name):
    """Increase a parameter within bounds."""
    if param_name not in PARAM_BOUNDS:
        return val
    mn, mx, step = PARAM_BOUNDS[param_name]
    new_val = min(mx, val + step)
    return round(new_val, 4) if isinstance(new_val, float) else new_val


def _nudge_down(val, param_name):
    """Decrease a parameter within bounds."""
    if param_name not in PARAM_BOUNDS:
        return val
    mn, mx, step = PARAM_BOUNDS[param_name]
    new_val = max(mn, val - step)
    return round(new_val, 4) if isinstance(new_val, float) else new_val


def save_tuned_params(params, reasons):
    """Write tuned params to YAML for next trading day."""
    TUNED_CONFIG.parent.mkdir(exist_ok=True)

    output = {
        "# Auto-tuned by auto_optimizer.py": None,
        "# Date": str(date.today()),
        "# Reasons": reasons,
        "pro_strategy_v2": params,
    }

    with open(TUNED_CONFIG, "w") as f:
        f.write(f"# Auto-tuned by auto_optimizer.py on {date.today()}\n")
        f.write(f"# Reasons:\n")
        for r in reasons:
            f.write(f"#   - {r}\n")
        f.write(f"\npro_strategy_v2:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")

    logger.info(f"  Tuned params saved: {TUNED_CONFIG}")

    # Also log history
    history = []
    if OPTIMIZATION_LOG.exists():
        try:
            history = json.loads(OPTIMIZATION_LOG.read_text())
        except:
            pass
    history.append({
        "date": str(date.today()),
        "params": {k: v for k, v in params.items() if v != DEFAULT_PARAMS.get(k)},
        "reasons": reasons,
    })
    # Keep last 30 days
    history = history[-30:]
    OPTIMIZATION_LOG.write_text(json.dumps(history, indent=2))


def load_tuned_params():
    """Load auto-tuned params if available. Returns dict or None."""
    if not TUNED_CONFIG.exists():
        return None
    try:
        with open(TUNED_CONFIG) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("pro_strategy_v2", None)
    except:
        return None


def run_optimize():
    """Main optimization flow — analyze recent trades, compute adjustments, save."""
    logger.info(f"\n{'═' * 50}")
    logger.info(f"  AUTO OPTIMIZER — {date.today()}")
    logger.info(f"{'═' * 50}")

    trades = load_recent_trades(days=7)
    if trades.empty:
        logger.info("  No recent trades found. Skipping optimization.")
        return

    logger.info(f"  Loaded {len(trades)} trades from last 7 days")

    analysis = analyze_trades(trades)
    logger.info(f"\n  ─── ANALYSIS ───")
    logger.info(f"  Total trades: {analysis['total_trades']}")
    logger.info(f"  Win rate: {analysis['win_rate']:.0%}")
    logger.info(f"  Total P&L: ₹{analysis['total_pnl']:+,.0f}")
    logger.info(f"  Avg win: ₹{analysis['avg_win']:+,.0f} | Avg loss: ₹{analysis['avg_loss']:+,.0f}")

    # Load current params
    current = load_tuned_params() or DEFAULT_PARAMS

    # Compute adjustments
    new_params, reasons = compute_adjustments(analysis, current)

    logger.info(f"\n  ─── ADJUSTMENTS ───")
    for r in reasons:
        logger.info(f"  • {r}")

    # Show changed params
    changed = {k: v for k, v in new_params.items() if v != DEFAULT_PARAMS.get(k)}
    if changed:
        logger.info(f"\n  ─── CHANGED PARAMS ───")
        for k, v in changed.items():
            logger.info(f"  {k}: {DEFAULT_PARAMS.get(k)} → {v}")

    save_tuned_params(new_params, reasons)
    logger.info(f"\n  ✅ Optimization complete. Params saved for tomorrow.")


def run_report():
    """Print optimization history."""
    if not OPTIMIZATION_LOG.exists():
        print("  No optimization history yet.")
        return

    history = json.loads(OPTIMIZATION_LOG.read_text())
    print(f"\n  OPTIMIZATION HISTORY (last {len(history)} entries)")
    print(f"  {'═' * 50}")
    for entry in history[-10:]:
        print(f"\n  📅 {entry['date']}")
        for r in entry.get("reasons", []):
            print(f"    • {r}")
        changed = entry.get("params", {})
        if changed:
            print(f"    Changed: {changed}")


def run_reset():
    """Reset to default params."""
    if TUNED_CONFIG.exists():
        TUNED_CONFIG.unlink()
        print("  ✅ Reset to default params. auto_tuned.yaml deleted.")
    else:
        print("  Already using defaults.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "optimize":
        run_optimize()
    elif cmd == "analyze":
        trades = load_recent_trades(7)
        if not trades.empty:
            a = analyze_trades(trades)
            for k, v in sorted(a.items()):
                print(f"  {k}: {v}")
        else:
            print("  No trades found.")
    elif cmd == "report":
        run_report()
    elif cmd == "reset":
        run_reset()
    else:
        print("Usage: python auto_optimizer.py [analyze|optimize|report|reset]")
