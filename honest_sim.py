"""
HONEST Pro Simulator — Research-backed, no cheating.

Uses ProStrategy (based on academic papers + backtest findings):
  - 0.3% breakout buffer (not 0.01% — reduces fake breakdowns)
  - Full-range stop (73% WR vs 53% half-range — AAPL TOS study)
  - Volume spike 1.5x required (WealthHub, Zarattini 2024)
  - Momentum filter (no shorting into green surges)
  - 30-min cooldown after SL (standard risk mgmt)
  - VWAP RSI 75/25 (stricter than 65/35 — fewer false signals)
  - Retest confirmation (2-candle hold)
  - No entries after 14:00

Usage:
  python honest_sim.py                          # Nifty 50
  python honest_sim.py --stocks HDFCBANK BPCL
  python honest_sim.py --universe nifty250
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
from strategies.pro_strategy import ProStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 50


def send_telegram(msg):
    for cfg in ["config/config_test.yaml", "config/config_prod.yaml"]:
        if Path(cfg).exists():
            with open(cfg) as f:
                c = yaml.safe_load(f) or {}
            a = c.get("alerts", {})
            token = a.get("telegram_bot_token", "")
            # Support both single chat_id and list of chat_ids
            chat_ids = a.get("telegram_chat_ids", [])
            single_id = a.get("telegram_chat_id", "")
            if single_id and single_id not in chat_ids:
                chat_ids.append(single_id)
            if token and chat_ids:
                try:
                    import urllib.request
                    for chat_id in chat_ids:
                        if not chat_id or chat_id == "SECOND_CHAT_ID":
                            continue
                        data = json.dumps({"chat_id": chat_id, "text": msg}).encode()
                        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                            data=data, headers={"Content-Type": "application/json"})
                        urllib.request.urlopen(req, timeout=10)
                    logger.info(f"  Telegram sent to {len(chat_ids)} recipients!")
                except Exception as e:
                    logger.error(f"  Telegram error: {e}")
                return


def fetch_intraday(symbol):
    try:
        import yfinance as yf
        df = yf.Ticker(f"{symbol}.NS").history(period="5d", interval="5m")
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["symbol"] = symbol
        times = pd.to_datetime(df["datetime"])
        last_date = times.dt.date.iloc[-1]
        df = df[times.dt.date == last_date].reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"  {symbol}: fetch failed — {e}")
        return None


class HonestProTrader:
    """Walks candle-by-candle using ProStrategy. Zero future data."""

    def __init__(self, symbol, direction, config):
        self.symbol = symbol
        self.direction = direction
        self.capital = config["capital"]["total"]
        self.cost_model = ZerodhaCostModel()
        self.strategy = ProStrategy(config.get("strategies", {}).get("pro", {}))

        # State
        self.orb = None
        self.in_trade = False
        self.entry_price = 0
        self.entry_time = None
        self.side = None
        self.sl = 0
        self.tgt = 0
        self.shares = 0
        self.sl_moved_to_be = False
        self.trade_type = ""

        self.trades = []
        self.trade_count = 0
        self.max_trades = 2
        self.cooldown_until = 0  # candle index when cooldown ends
        self.log = []

    def _log(self, t_str, msg):
        entry = f"  [{t_str}] {self.symbol}: {msg}"
        self.log.append(entry)
        logger.info(entry)

    def _close_trade(self, exit_price, exit_time, reason):
        if self.side == "SHORT":
            gross = (self.entry_price - exit_price) * self.shares
        else:
            gross = (exit_price - self.entry_price) * self.shares
        costs = self.cost_model.calculate(self.entry_price * self.shares, exit_price * self.shares).total
        net = gross - costs
        et = str(self.entry_time).split("+")[0]
        xt = str(exit_time).split("+")[0]

        self.trades.append({
            "symbol": self.symbol, "direction": self.side,
            "entry": round(self.entry_price, 2), "exit": round(exit_price, 2),
            "entry_time": et, "exit_time": xt,
            "sl": round(self.sl, 2), "tgt": round(self.tgt, 2),
            "qty": self.shares, "gross": round(gross, 2),
            "costs": round(costs, 2), "net_pnl": round(net, 2),
            "reason": reason, "type": self.trade_type,
        })

        emoji = "✅" if net > 0 else "❌"
        et_s = str(self.entry_time).split(" ")[-1].split("+")[0][:5]
        xt_s = str(exit_time).split(" ")[-1].split("+")[0][:5]
        self._log(xt_s, f"{emoji} EXIT {self.side} @ Rs {exit_price:,.2f} | P&L: Rs {net:+,.2f} | {reason} ({self.trade_type}, entered {et_s})")

        self.in_trade = False
        self.trade_count += 1
        self.sl_moved_to_be = False

    def _enter_trade(self, signal, idx):
        self.entry_price = signal["entry"]
        self.entry_time = signal["time"]
        self.side = signal["side"]
        self.sl = signal["sl"]
        self.tgt = signal["tgt"]
        self.trade_type = signal["type"]
        risk = signal["risk"]
        self.shares = max(1, int(self.capital * self.strategy.max_risk_pct / max(risk, 0.01)))
        self.in_trade = True
        self.sl_moved_to_be = False

        t_str = signal["time"].strftime("%H:%M")
        side_emoji = "🟢" if self.side == "LONG" else "🔴"
        self._log(t_str, f"{side_emoji} {self.side} ({self.trade_type}) @ Rs {self.entry_price:,.2f} | "
                  f"SL: Rs {self.sl:,.2f} | TGT: Rs {self.tgt:,.2f} | Qty: {self.shares} | {signal['reason']}")

    def process_candle(self, i, candles):
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        t_str = t.strftime("%H:%M")
        h, l, c = row["high"], row["low"], row["close"]
        hour, minute = t.hour, t.minute

        # ── Phase 1: ORB Formation ──
        if self.orb is None:
            if i < self.strategy.orb_candles:
                if i == 0:
                    self._log(t_str, "📊 Market open. Watching ORB...")
                return
            self.orb = self.strategy.compute_orb(candles, self.strategy.orb_candles)
            if self.orb is None or self.orb["range"] < 0.5:
                self._log(t_str, "⚠️ ORB too narrow")
                return
            self._log(t_str, f"📐 ORB: High={self.orb['high']:,.2f} Low={self.orb['low']:,.2f} "
                      f"Range={self.orb['range']:,.2f}")

        # ── Phase 2: Square off ──
        if hour >= self.strategy.square_off_hour and minute >= self.strategy.square_off_minute:
            if self.in_trade:
                self._close_trade(c, t, "SQUARE_OFF")
            return

        # ── Phase 3: Manage open trade ──
        if self.in_trade:
            if self.side == "LONG":
                if l <= self.sl:
                    self._close_trade(self.sl, t, "STOP_LOSS")
                    self.cooldown_until = i + self.strategy.cooldown_candles
                    return
                if h >= self.tgt:
                    self._close_trade(self.tgt, t, "TARGET")
                    return
                # Trail to breakeven after 1x risk
                unrealised = c - self.entry_price
                initial_risk = self.entry_price - self.sl
                if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                    old_sl = self.sl
                    self.sl = self.entry_price + 0.10
                    self.sl_moved_to_be = True
                    self._log(t_str, f"🔒 SL → breakeven Rs {self.sl:,.2f} (was Rs {old_sl:,.2f})")
            else:
                if h >= self.sl:
                    self._close_trade(self.sl, t, "STOP_LOSS")
                    self.cooldown_until = i + self.strategy.cooldown_candles
                    return
                if l <= self.tgt:
                    self._close_trade(self.tgt, t, "TARGET")
                    return
                unrealised = self.entry_price - c
                initial_risk = self.sl - self.entry_price
                if initial_risk > 0 and unrealised >= initial_risk and not self.sl_moved_to_be:
                    old_sl = self.sl
                    self.sl = self.entry_price - 0.10
                    self.sl_moved_to_be = True
                    self._log(t_str, f"🔒 SL → breakeven Rs {self.sl:,.2f} (was Rs {old_sl:,.2f})")
            return

        # ── Phase 4: Look for entry (with cooldown check) ──
        if self.trade_count >= self.max_trades:
            return
        if i < self.cooldown_until:
            return  # cooling down after SL

        # Try ORB breakout (9:30 - 10:30)
        if hour < 10 or (hour == 10 and minute <= 30):
            signal = self.strategy.generate_orb_signal(candles, i, self.orb, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return

        # Try Pullback (10:00 - 11:30)
        if 10 <= hour <= 11:
            signal = self.strategy.generate_pullback_signal(candles, i, self.orb, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return

        # Try VWAP reversion (10:30 - 14:00)
        if hour >= 10 and (hour < 14 or (hour == 14 and minute == 0)):
            signal = self.strategy.generate_vwap_signal(candles, i, self.direction)
            if signal:
                self._enter_trade(signal, i)
                return


def run(symbols, label="Nifty 50"):
    logger.info(f"\n{'='*60}")
    logger.info(f"  HONEST PRO SIM — {date.today()} — {label}")
    logger.info(f"  Research-backed: buffer, full-range SL, momentum filter")
    logger.info(f"{'='*60}")

    # Score stocks
    loader = DataLoader()
    df = loader.load_backtest_data(symbols, target_date=date.today().isoformat())
    featured = []
    for sym in (df["symbol"].unique() if not df.empty else []):
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            featured.append(add_features(sdf))
    if not featured:
        logger.error("  No data!")
        return

    all_feat = pd.concat(featured, ignore_index=True)
    model, feats = None, None
    mp = Path("models/stock_predictor.pkl")
    if mp.exists():
        with open(mp, "rb") as f:
            d = pickle.load(f)
        model, feats = d["model"], d["features"]
    avail = [c for c in (feats or FEATURE_COLS) if c in all_feat.columns]
    scores = score_stocks(all_feat, model, avail if model else None)

    logger.info(f"\n  ─── PRE-MARKET SCORING ───")
    for _, r in scores.head(10).iterrows():
        d = r.get("direction", "LONG")
        arrow = "▲" if d == "LONG" else "▼"
        star = "***" if r["score"] >= SCORE_THRESHOLD else ""
        logger.info(f"  {r['symbol']:<12}{r['score']:>5.0f} {arrow}{d:<5} RSI={r['rsi']:>5.1f} {star}")

    picks = scores[scores["score"] >= SCORE_THRESHOLD].head(5)
    if picks.empty:
        picks = scores.head(3)

    with open("config/config_test.yaml") as f:
        config = yaml.safe_load(f)

    all_trades = []
    all_logs = []

    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        direction = pick.get("direction", "LONG")
        logger.info(f"\n{'─'*50}")
        logger.info(f"  {sym} | {direction} | Score: {pick['score']:.0f} | RSI: {pick['rsi']:.1f}")
        logger.info(f"{'─'*50}")

        candles = fetch_intraday(sym)
        if candles is None or len(candles) < 6:
            logger.warning(f"  {sym}: no intraday data")
            continue

        trader = HonestProTrader(sym, direction, config)
        for i in range(len(candles)):
            trader.process_candle(i, candles)

        all_trades.extend(trader.trades)
        all_logs.extend(trader.log)
        if not trader.trades:
            logger.info(f"  {sym}: No signal triggered — filters prevented bad entries")

    total_pnl = sum(t["net_pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["net_pnl"] > 0)
    losses = len(all_trades) - wins

    logger.info(f"\n{'='*60}")
    logger.info(f"  HONEST PRO RESULTS — {date.today()}")
    logger.info(f"{'='*60}")
    for t in all_trades:
        emoji = "✅" if t["net_pnl"] > 0 else "❌"
        et = str(t["entry_time"]).split(" ")[-1].split("+")[0][:5]
        xt = str(t["exit_time"]).split(" ")[-1].split("+")[0][:5]
        logger.info(f"  {emoji} {t['symbol']} {t['direction']} ({t['type']}) | "
                     f"{et} Rs {t['entry']:,.2f} → {xt} Rs {t['exit']:,.2f} | "
                     f"Rs {t['net_pnl']:+,.2f} | {t['reason']}")
    logger.info(f"\n  Trades: {len(all_trades)} | Won: {wins} | Lost: {losses}")
    logger.info(f"  Net P&L: Rs {total_pnl:+,.2f}")

    # Telegram
    tg = [f"🤖 Rajan Stock Bot — {date.today()}"]
    tg.append(f"📋 {label} | PRO Strategy (research-backed)")
    tg.append(f"💰 Capital: Rs {config['capital']['total']:,.2f}")
    tg.append("")

    tg.append("📊 Picks:")
    for _, r in picks.iterrows():
        d = r.get("direction", "LONG")
        arrow = "📈" if d == "LONG" else "📉"
        tg.append(f"  {arrow} {r['symbol']}: {r['score']:.0f}/100 (RSI {r['rsi']:.1f}) {d}")
    tg.append("")

    if all_trades:
        tg.append(f"💰 Trades ({len(all_trades)}):")
        tg.append("───────────────────────────")
        for t in all_trades:
            emoji = "✅" if t["net_pnl"] > 0 else "❌"
            et = str(t["entry_time"]).split(" ")[-1].split("+")[0][:5]
            xt = str(t["exit_time"]).split(" ")[-1].split("+")[0][:5]
            tg.append(f"{emoji} {t['symbol']} ({t['direction']} {t['type']})")
            tg.append(f"  {et} Entry @ Rs {t['entry']:,.2f}")
            tg.append(f"  {xt} Exit  @ Rs {t['exit']:,.2f}")
            tg.append(f"  Qty: {t['qty']} | SL: Rs {t['sl']:,.2f} | TGT: Rs {t['tgt']:,.2f}")
            tg.append(f"  P&L: Rs {t['net_pnl']:+,.2f} (Costs: Rs {t['costs']:,.2f})")
            tg.append(f"  {t['reason']}")
            tg.append("")
        tg.append("───────────────────────────")
        tg.append(f"📋 Net P&L: Rs {total_pnl:+,.2f}")
        tg.append(f"🏆 Won: {wins}/{len(all_trades)}")
        tg.append(f"💰 Capital: Rs {config['capital']['total']+total_pnl:,.2f}")
    else:
        tg.append("📭 No trades — all filtered by pro strategy")
        tg.append("(Momentum, volume, buffer, or RSI filters blocked entries)")

    tg.append("")
    tg.append("🔬 Pro filters active:")
    tg.append("  0.3% buffer | Full-range SL | Vol 1.5x")
    tg.append("  Momentum check | 30min cooldown | RSI 75/25")

    send_telegram("\n".join(tg))

    if all_trades:
        pd.DataFrame(all_trades).to_csv(f"results/honest_pro_{label.replace(' ','_')}_{date.today()}.csv", index=False)
    if all_logs:
        with open(f"logs/honest_pro_{date.today()}.log", "w") as f:
            f.write("\n".join(all_logs))

    return all_trades


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stocks", nargs="+", default=None)
    p.add_argument("--universe", default=None, choices=["nifty50","nifty100","nifty250"])
    p.add_argument("--threshold", type=int, default=50)
    args = p.parse_args()
    SCORE_THRESHOLD = args.threshold

    if args.stocks:
        run(args.stocks, label=", ".join(args.stocks))
    elif args.universe:
        run(get_universe(args.universe), label=f"Nifty {args.universe.replace('nifty','')}")
    else:
        run(NIFTY_50, label="Nifty 50")
