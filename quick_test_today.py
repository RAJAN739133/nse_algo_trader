"""
Quick Test — Intraday candle-by-candle simulation.
Downloads 5-min intraday data, walks through candles like a real trader.

Features:
  - ORB range detection (first 15 min)
  - Entry on breakout/breakdown with volume
  - SL/TGT hit during the day (not just square off at 3:10)
  - Re-entry allowed after a trade closes
  - Realistic timestamps from actual candle data
  - Sends detailed Telegram alert

Usage:
  python quick_test_today.py                     # Nifty 50
  python quick_test_today.py --stocks HDFCBANK   # Single stock
  python quick_test_today.py --universe nifty250  # Nifty 250
"""
import os, sys, json, argparse, pickle, logging
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.symbols import NIFTY_50, get_universe
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS
from backtest.costs import ZerodhaCostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 50
MAX_TRADES_PER_STOCK = 2


def send_telegram(msg):
    for cfg in ["config/config_test.yaml", "config/config_prod.yaml"]:
        if Path(cfg).exists():
            with open(cfg) as f:
                c = yaml.safe_load(f) or {}
            a = c.get("alerts", {})
            token = a.get("telegram_bot_token", "")
            chat_id = a.get("telegram_chat_id", "")
            if token and chat_id:
                try:
                    import urllib.request
                    data = json.dumps({"chat_id": chat_id, "text": msg}).encode()
                    req = urllib.request.Request(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        data=data, headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=10)
                    logger.info("  Telegram sent!")
                except Exception as e:
                    logger.error(f"  Telegram error: {e}")
                return


def fetch_intraday(symbol, period="5d", interval="5m"):
    """Download intraday 5-min candles from yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["symbol"] = symbol
        return df
    except Exception as e:
        logger.warning(f"  {symbol}: intraday fetch failed — {e}")
        return None


def simulate_intraday_trade(df, direction, config):
    """
    Walk through 5-min candles like a real trader.

    Steps:
    1. Find ORB range from first 3 candles (9:15-9:30)
    2. Wait for breakout/breakdown after ORB
    3. Enter with SL and TGT
    4. Check SL/TGT on each subsequent candle
    5. Square off at 15:10 if neither hit
    6. Allow re-entry if trade exits before 14:00

    Returns list of trades (can be 0, 1, or 2 per stock).
    """
    cost_model = ZerodhaCostModel()
    capital = config["capital"]["total"]
    risk_pct = config["capital"]["risk_per_trade"]
    trades = []

    if df is None or len(df) < 10:
        return trades

    # Parse timestamps
    times = pd.to_datetime(df["datetime"])
    df = df.copy()
    df["_time"] = times

    # Get today's candles only (last trading day in data)
    last_date = times.dt.date.iloc[-1]
    today_mask = times.dt.date == last_date
    day_df = df[today_mask].reset_index(drop=True)

    if len(day_df) < 6:
        return trades

    # ── Step 1: Find ORB range (first 15 min = first 3 five-min candles) ──
    orb_candles = day_df.iloc[:3]
    orb_high = orb_candles["high"].max()
    orb_low = orb_candles["low"].min()
    orb_range = orb_high - orb_low
    symbol = day_df["symbol"].iloc[0] if "symbol" in day_df.columns else "UNKNOWN"

    if orb_range < 0.5:
        return trades  # too narrow, skip

    logger.info(f"    {symbol} ORB: High={orb_high:,.2f} Low={orb_low:,.2f} Range={orb_range:,.2f}")

    # ── Step 2-5: Walk through remaining candles ──
    in_trade = False
    trade_count = 0
    entry_price = 0
    entry_time = None
    trade_side = None
    sl = 0
    tgt = 0
    shares = 0
    atr = orb_range  # use ORB range as ATR proxy

    for i in range(3, len(day_df)):
        row = day_df.iloc[i]
        candle_time = row["_time"]
        high, low, close = row["high"], row["low"], row["close"]
        volume = row.get("volume", 0)
        hour, minute = candle_time.hour, candle_time.minute

        # ── Square off time: 15:10 ──
        if hour >= 15 and minute >= 10:
            if in_trade:
                exit_price = close
                exit_time = candle_time
                if trade_side == "SHORT":
                    gross = (entry_price - exit_price) * shares
                else:
                    gross = (exit_price - entry_price) * shares
                costs = cost_model.calculate(entry_price * shares, exit_price * shares).total
                net = gross - costs
                trades.append({
                    "symbol": symbol, "direction": trade_side,
                    "entry": round(entry_price, 2), "exit": round(exit_price, 2),
                    "entry_time": str(entry_time), "exit_time": str(exit_time),
                    "sl": round(sl, 2), "tgt": round(tgt, 2),
                    "qty": shares, "gross": round(gross, 2),
                    "costs": round(costs, 2), "net_pnl": round(net, 2),
                    "reason": "SQUARE_OFF",
                })
                in_trade = False
            break  # done for the day

        # ── If in a trade, check SL and TGT ──
        if in_trade:
            hit_sl = False
            hit_tgt = False

            if trade_side == "LONG":
                if low <= sl:
                    hit_sl = True
                    exit_price = sl
                elif high >= tgt:
                    hit_tgt = True
                    exit_price = tgt
            else:  # SHORT
                if high >= sl:
                    hit_sl = True
                    exit_price = sl
                elif low <= tgt:
                    hit_tgt = True
                    exit_price = tgt

            if hit_sl or hit_tgt:
                exit_time = candle_time
                reason = "STOP_LOSS" if hit_sl else "TARGET"
                if trade_side == "SHORT":
                    gross = (entry_price - exit_price) * shares
                else:
                    gross = (exit_price - entry_price) * shares
                costs = cost_model.calculate(entry_price * shares, exit_price * shares).total
                net = gross - costs
                trades.append({
                    "symbol": symbol, "direction": trade_side,
                    "entry": round(entry_price, 2), "exit": round(exit_price, 2),
                    "entry_time": str(entry_time), "exit_time": str(exit_time),
                    "sl": round(sl, 2), "tgt": round(tgt, 2),
                    "qty": shares, "gross": round(gross, 2),
                    "costs": round(costs, 2), "net_pnl": round(net, 2),
                    "reason": reason,
                })
                in_trade = False
                trade_count += 1

                # Can re-enter if before 14:00 and under max trades
                if trade_count >= MAX_TRADES_PER_STOCK or hour >= 14:
                    break
                continue

            # ── Trailing stop: move SL to breakeven after 1:1 RR ──
            if trade_side == "LONG":
                unrealised = close - entry_price
                if unrealised >= atr * 1.0 and sl < entry_price:
                    sl = entry_price + 0.10  # breakeven + buffer
            else:
                unrealised = entry_price - close
                if unrealised >= atr * 1.0 and sl > entry_price:
                    sl = entry_price - 0.10

            continue

        # ── Not in trade — look for entry signal ──
        if trade_count >= MAX_TRADES_PER_STOCK:
            continue
        # Don't enter after 14:30
        if hour >= 14 and minute >= 30:
            continue

        # Volume check (need above average)
        vol_avg = day_df["volume"].iloc[max(0, i-5):i].mean()
        vol_ok = volume > vol_avg * 1.2 if vol_avg > 0 else True

        if direction == "LONG" and close > orb_high and vol_ok:
            # ── LONG breakout above ORB high ──
            entry_price = close
            entry_time = candle_time
            trade_side = "LONG"
            sl = orb_low - atr * 0.2  # below ORB low
            tgt = entry_price + (entry_price - sl) * 1.5  # 1:1.5 RR
            risk = entry_price - sl
            shares = max(1, int(capital * risk_pct / risk))
            in_trade = True
            logger.info(f"    → {symbol} LONG @ Rs {entry_price:,.2f} at {candle_time.strftime('%H:%M')} | SL: Rs {sl:,.2f} | TGT: Rs {tgt:,.2f} | Qty: {shares}")

        elif direction == "SHORT" and close < orb_low and vol_ok:
            # ── SHORT breakdown below ORB low ──
            entry_price = close
            entry_time = candle_time
            trade_side = "SHORT"
            sl = orb_high + atr * 0.2  # above ORB high
            tgt = entry_price - (sl - entry_price) * 1.5  # 1:1.5 RR
            risk = sl - entry_price
            shares = max(1, int(capital * risk_pct / risk))
            in_trade = True
            logger.info(f"    → {symbol} SHORT @ Rs {entry_price:,.2f} at {candle_time.strftime('%H:%M')} | SL: Rs {sl:,.2f} | TGT: Rs {tgt:,.2f} | Qty: {shares}")

        # ── VWAP mean reversion (after 10:30, if no ORB trade) ──
        elif trade_count == 0 and hour >= 10 and minute >= 30:
            mid = (day_df["high"].iloc[:i].max() + day_df["low"].iloc[:i].min()) / 2
            band = (day_df["high"].iloc[:i].max() - day_df["low"].iloc[:i].min()) * 0.3

            if direction == "LONG" and close < mid - band and vol_ok:
                entry_price = close
                entry_time = candle_time
                trade_side = "LONG"
                sl = mid - band * 2
                tgt = mid
                risk = entry_price - sl
                if risk > 0:
                    shares = max(1, int(capital * risk_pct / risk))
                    in_trade = True
                    logger.info(f"    → {symbol} VWAP LONG @ Rs {entry_price:,.2f} at {candle_time.strftime('%H:%M')}")

            elif direction == "SHORT" and close > mid + band and vol_ok:
                entry_price = close
                entry_time = candle_time
                trade_side = "SHORT"
                sl = mid + band * 2
                tgt = mid
                risk = sl - entry_price
                if risk > 0:
                    shares = max(1, int(capital * risk_pct / risk))
                    in_trade = True
                    logger.info(f"    → {symbol} VWAP SHORT @ Rs {entry_price:,.2f} at {candle_time.strftime('%H:%M')}")

    return trades


def run(symbols, label="Nifty 50"):
    logger.info(f"\n{'='*60}")
    logger.info(f"  INTRADAY SIMULATION — {date.today()} — {label}")
    logger.info(f"  Stocks: {len(symbols)} | Threshold: {SCORE_THRESHOLD}")
    logger.info(f"  Max trades/stock: {MAX_TRADES_PER_STOCK}")
    logger.info(f"{'='*60}")

    # ── Step 1: Load daily data + score stocks ──
    loader = DataLoader()
    df = loader.load_backtest_data(symbols, target_date=date.today().isoformat())
    if df.empty:
        logger.error("  No daily data!")
        return

    featured = []
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            featured.append(add_features(sdf))
    if not featured:
        logger.error("  Not enough data")
        return
    all_feat = pd.concat(featured, ignore_index=True)

    # Load ML model
    model, feats = None, None
    mp = Path("models/stock_predictor.pkl")
    if mp.exists():
        with open(mp, "rb") as f:
            d = pickle.load(f)
        model, feats = d["model"], d["features"]

    avail = [c for c in (feats or FEATURE_COLS) if c in all_feat.columns]
    scores = score_stocks(all_feat, model, avail if model else None)

    # Show scores
    logger.info(f"\n  {'Symbol':<12}{'Score':>6}{'Long':>6}{'Short':>6} {'Dir':<6}{'RSI':>6} {'Strategy'}")
    logger.info(f"  {'-'*60}")
    for _, r in scores.head(15).iterrows():
        star = " ***" if r["score"] >= SCORE_THRESHOLD else ""
        d = r.get("direction", "LONG")
        arrow = "▲" if d == "LONG" else "▼"
        logger.info(f"  {r['symbol']:<12}{r['score']:>5.0f}{r.get('long_score',0):>6.0f}{r.get('short_score',0):>6.0f} {arrow}{d:<5}{r['rsi']:>6.1f} {r['strategy']}{star}")

    picks = scores[scores["score"] >= SCORE_THRESHOLD].head(5)
    if picks.empty:
        picks = scores.head(3)
        logger.info(f"\n  No stocks above {SCORE_THRESHOLD} — using top 3")

    # ── Step 2: Fetch intraday data + simulate ──
    logger.info(f"\n  Fetching intraday 5-min candles...")
    with open("config/config_test.yaml") as f:
        config = yaml.safe_load(f)

    all_trades = []
    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        direction = pick.get("direction", "LONG")
        logger.info(f"\n  ─── {sym} ({direction}) Score: {pick['score']:.0f} RSI: {pick['rsi']:.1f} ───")

        intraday_df = fetch_intraday(sym, period="5d", interval="5m")
        if intraday_df is None or len(intraday_df) < 10:
            logger.warning(f"  {sym}: no intraday data, skipping")
            continue

        stock_trades = simulate_intraday_trade(intraday_df, direction, config)
        all_trades.extend(stock_trades)

    # ── Step 3: Summary ──
    total_pnl = sum(t["net_pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["net_pnl"] > 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"  RESULT: {len(all_trades)} trades | {wins} wins | Net P&L: Rs {total_pnl:+,.2f}")
    logger.info(f"{'='*60}")
    for t in all_trades:
        emoji = "✅" if t["net_pnl"] > 0 else "❌"
        et = t["entry_time"].split(" ")[-1][:5] if " " in str(t["entry_time"]) else t["entry_time"]
        xt = t["exit_time"].split(" ")[-1][:5] if " " in str(t["exit_time"]) else t["exit_time"]
        logger.info(f"  {emoji} {t['symbol']} {t['direction']} | {et} Rs {t['entry']:,.2f} → {xt} Rs {t['exit']:,.2f} | P&L: Rs {t['net_pnl']:+,.2f} | {t['reason']}")

    # ── Step 4: Telegram ──
    tg = [f"🤖 Rajan Stock Bot — {date.today()}"]
    tg.append(f"📋 {label} | Capital: Rs {config['capital']['total']:,.2f}")

    # Claude Brain status
    try:
        from strategies.claude_brain import ClaudeBrain
        brain = ClaudeBrain(config=config)
        if brain.enabled:
            advice = brain.get_morning_analysis(vix=15, fii_net=0, recent_trades=[], stock_scores=[])
            if advice.get("notes","").startswith("Default"):
                tg.append("🧠 Claude Brain: ⚠️ Fallback (check API credits)")
            else:
                tg.append(f"🧠 Claude Brain: {advice.get('risk_level','normal')}")
        else:
            tg.append("🧠 Claude Brain: ❌ Disabled")
    except:
        pass

    tg.append("")
    tg.append("📊 Top Picks:")
    for _, r in picks.head(5).iterrows():
        d = r.get("direction", "LONG")
        arrow = "📈" if d == "LONG" else "📉"
        tg.append(f"  {arrow} {r['symbol']}: {r['score']:.0f}/100 (RSI {r['rsi']:.1f}) {d}")

    tg.append("")
    if all_trades:
        tg.append(f"💰 Trades ({len(all_trades)}):")
        tg.append(f"{'─'*35}")
        for t in all_trades:
            emoji = "✅" if t["net_pnl"] > 0 else "❌"
            et = str(t["entry_time"]).split(" ")[-1][:5]
            xt = str(t["exit_time"]).split(" ")[-1][:5]
            tg.append(f"{emoji} {t['symbol']} ({t['direction']})")
            tg.append(f"  Entry: {et} @ Rs {t['entry']:,.2f}")
            tg.append(f"  Exit:  {xt} @ Rs {t['exit']:,.2f}")
            tg.append(f"  Qty: {t['qty']} | SL: Rs {t['sl']:,.2f} | TGT: Rs {t['tgt']:,.2f}")
            tg.append(f"  P&L: Rs {t['net_pnl']:+,.2f} (Costs: Rs {t['costs']:,.2f}) | {t['reason']}")
            tg.append("")

        tg.append(f"{'─'*35}")
        tg.append(f"📋 Summary:")
        tg.append(f"  Net P&L: Rs {total_pnl:+,.2f}")
        tg.append(f"  Won: {wins}/{len(all_trades)} | Capital: Rs {config['capital']['total'] + total_pnl:,.2f}")
    else:
        tg.append("📭 No trades — no intraday signals triggered")

    msg = "\n".join(tg)
    logger.info(f"\n  Telegram message:\n{msg}")
    send_telegram(msg)

    # Save results
    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"results/intraday_{label.replace(' ','_')}_{date.today()}.csv", index=False)
        logger.info(f"  Results saved to results/")

    return all_trades


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stocks", nargs="+", default=None)
    p.add_argument("--universe", default=None, choices=["nifty50","nifty100","nifty250"])
    p.add_argument("--threshold", type=int, default=50)
    args = p.parse_args()

    if args.threshold:
        SCORE_THRESHOLD = args.threshold

    if args.stocks:
        run(args.stocks, label=", ".join(args.stocks))
    elif args.universe:
        run(get_universe(args.universe), label=f"Nifty {args.universe.replace('nifty','')}")
    else:
        run(NIFTY_50, label="Nifty 50")
