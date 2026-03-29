"""
Paper Trading Runner — Test the algo with NO real money.

Usage from IntelliJ or terminal:
  python paper_trader.py simulate                        # Normal day
  python paper_trader.py simulate --scenario crash       # Crash day
  python paper_trader.py simulate --scenario rally       # Rally day
  python paper_trader.py simulate --date 2024-03-15      # Specific date
  python paper_trader.py multi-test --days 10            # 10-day backtest
  python paper_trader.py scenarios                       # All 5 scenarios
  python paper_trader.py analyse                         # Review results
"""
import os, sys, argparse, logging
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.symbols import DEFAULT_UNIVERSE
from backtest.costs import ZerodhaCostModel

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/paper_{date.today()}.log")],
)
logger = logging.getLogger(__name__)


class SimulatedMarket:
    """Generates realistic intraday 1-min candles for testing."""
    def __init__(self, symbols, test_date=None, scenario="normal"):
        self.symbols = symbols
        self.test_date = test_date or date.today().isoformat()
        self.scenario = scenario
        self.data = {}
        self.current_minute = 0
        np.random.seed(hash(self.test_date) % 2**31)
        presets = {
            "normal":   {"vix":14.5,"gap":0.008,"trend": 0.0002,"vmul":1.0},
            "volatile": {"vix":22.0,"gap":0.020,"trend": 0.0000,"vmul":2.0},
            "crash":    {"vix":32.0,"gap":0.040,"trend":-0.0008,"vmul":3.0},
            "rally":    {"vix":12.0,"gap":0.010,"trend": 0.0006,"vmul":1.5},
            "flat":     {"vix":11.0,"gap":0.003,"trend": 0.0000,"vmul":0.7},
        }
        s = presets.get(scenario, presets["normal"])
        self.vix = s["vix"]
        mins = pd.date_range(f"{self.test_date} 09:15", f"{self.test_date} 15:30", freq="1min")
        for sym in symbols:
            base = np.random.uniform(200, 3000)
            gap = np.random.uniform(-s["gap"], s["gap"])
            prev_close = base
            prices = [base * (1 + gap)]
            vols = [int(np.random.randint(20000, 100000) * s["vmul"])]
            for i in range(1, len(mins)):
                v = 0.003 if i < 15 else (0.002 if i > 360 else 0.001)
                prices.append(prices[-1] * (1 + np.random.randn() * v * s["vmul"] + s["trend"]))
                bv = np.random.randint(5000, 50000) * (3 if i < 15 else (2 if i > 360 else 1))
                vols.append(int(bv * s["vmul"]))
            c = np.array(prices)
            n = np.abs(np.random.randn(len(c))) * 0.001 * c
            self.data[sym] = {
                "df": pd.DataFrame({"datetime":mins[:len(c)],"open":np.round(np.roll(c,1),2),
                    "high":np.round(c+n,2),"low":np.round(c-n*0.8,2),
                    "close":np.round(c,2),"volume":vols[:len(c)]}),
                "prev_close": round(prev_close, 2),
                "gap_pct": round(gap * 100, 2),
            }
            self.data[sym]["df"].iloc[0, self.data[sym]["df"].columns.get_loc("open")] = round(base*(1+gap), 2)

    def get_quote(self, symbol):
        if symbol not in self.data: return None
        d = self.data[symbol]
        idx = min(self.current_minute, len(d["df"]) - 1)
        row = d["df"].iloc[idx]
        return {"symbol":symbol,"last_price":row["close"],"open":d["df"].iloc[0]["open"],
                "high":d["df"].iloc[:idx+1]["high"].max(),"low":d["df"].iloc[:idx+1]["low"].min(),
                "volume":d["df"].iloc[:idx+1]["volume"].sum(),"prev_close":d["prev_close"],"gap_pct":d["gap_pct"]}

    def get_15min_candles(self, symbol):
        if symbol not in self.data: return pd.DataFrame()
        df = self.data[symbol]["df"].iloc[:self.current_minute+1].copy().set_index("datetime")
        return df.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()

    def get_candles(self, symbol, n=None):
        if symbol not in self.data: return pd.DataFrame()
        return self.data[symbol]["df"].iloc[:self.current_minute+1].copy()

    def advance(self, minutes=1):
        self.current_minute = min(self.current_minute + minutes, 375)

    def is_open(self): return self.current_minute < 375

    def time_str(self):
        h, m = divmod(15 + self.current_minute, 60)
        return f"{9+h//60+h:02d}:{m:02d}" if h < 60 else f"{9+h//60:02d}:{m:02d}"


class PaperTrader:
    """Simulated order execution with full cost model."""
    def __init__(self, config, market):
        self.config = config; self.market = market
        self.capital = config["capital"]["total"]
        self.positions = {}; self.closed = []; self.pnl = 0; self.count = 0
        self.cost_model = ZerodhaCostModel()
        self.max_trades = config["capital"]["max_trades_per_day"]

    def can_trade(self):
        if self.count >= self.max_trades: return False, "Max trades"
        if self.pnl < -self.capital * self.config["capital"]["daily_loss_limit"]: return False, "Loss limit"
        return True, "OK"

    def buy(self, sym, price, qty, sl, target, strat):
        if sym in self.positions: return False
        ok, reason = self.can_trade()
        if not ok:
            logger.info(f"    Skip {sym}: {reason}"); return False
        cost = self.cost_model.total_cost(price, qty, "intraday")
        self.positions[sym] = {"entry":price,"qty":qty,"sl":sl,"target":target,
            "strat":strat,"time":self.market.current_minute,"cost":cost,"high":price}
        self.count += 1
        logger.info(f"  BUY  {sym} @ {price:.2f} x{qty} | SL:{sl:.2f} TGT:{target:.2f} | {strat}")
        return True

    def sell(self, sym, price, reason):
        if sym not in self.positions: return
        p = self.positions.pop(sym)
        gross = (price - p["entry"]) * p["qty"]
        costs = p["cost"] + self.cost_model.total_cost(price, p["qty"], "intraday")
        net = gross - costs
        self.pnl += net
        self.closed.append({"symbol":sym,"strategy":p["strat"],"entry":p["entry"],
            "exit":price,"qty":p["qty"],"gross":round(gross,2),"costs":round(costs,2),
            "net_pnl":round(net,2),"reason":reason})
        e = "+" if net > 0 else "-"
        logger.info(f"  SELL {sym} @ {price:.2f} | P&L: {e}{abs(net):.2f} | {reason}")

    def check_stops(self):
        for sym in list(self.positions):
            p = self.positions[sym]
            q = self.market.get_quote(sym)
            if not q: continue
            px = q["last_price"]
            if px > p["high"]: p["high"] = px
            if px <= p["sl"]: self.sell(sym, p["sl"], "STOP_LOSS")
            elif px >= p["target"]: self.sell(sym, p["target"], "TARGET")
            elif p["high"] > p["entry"] * 1.01:
                trail = p["entry"] + (p["high"] - p["entry"]) * 0.5
                if px <= trail and trail > p["sl"]: self.sell(sym, px, "TRAILING")

    def square_off(self):
        for sym in list(self.positions):
            q = self.market.get_quote(sym)
            if q: self.sell(sym, q["last_price"], "SQUARE_OFF")


def check_orb(market, sym, cfg):
    candles = market.get_15min_candles(sym)
    if len(candles) < 2: return None
    orb_h, orb_l = candles.iloc[0]["high"], candles.iloc[0]["low"]
    rng = orb_h - orb_l
    if rng < 1: return None
    q = market.get_quote(sym)
    px = q["last_price"]
    if px > orb_h * 1.001:
        atr = rng * 1.5
        sl = px - atr * cfg.get("atr_stop_multiplier", 1.5)
        tgt = px + atr * cfg.get("trail_after_rr", 1.5) * cfg.get("atr_stop_multiplier", 1.5)
        return {"entry":px,"sl":sl,"target":tgt,"strat":"ORB"}
    return None


def check_vwap(market, sym, cfg):
    df = market.get_candles(sym)
    if len(df) < 30: return None
    df["vwap"] = (df["close"]*df["volume"]).cumsum() / df["volume"].cumsum()
    std = df["close"].rolling(20).std().iloc[-1]
    if pd.isna(std) or std == 0: return None
    vwap, px = df["vwap"].iloc[-1], df["close"].iloc[-1]
    lower = vwap - std * cfg.get("entry_band", 1.0)
    if px < lower:
        return {"entry":px,"sl":vwap - std*cfg.get("stop_band",2.0),"target":vwap,"strat":"VWAP"}
    return None


def run_simulate(test_date=None, scenario="normal", stocks=None):
    cfg_path = Path("config/config_test.yaml")
    if not cfg_path.exists(): cfg_path = Path("config/config_example.yaml")
    with open(cfg_path) as f: config = yaml.safe_load(f)

    symbols = stocks or DEFAULT_UNIVERSE[:10]
    logger.info(f"\n{'='*60}\n  PAPER TRADE | {test_date or 'today'} | {scenario} | {len(symbols)} stocks\n{'='*60}")

    market = SimulatedMarket(symbols, test_date, scenario)
    trader = PaperTrader(config, market)
    logger.info(f"  VIX: {market.vix:.1f}")

    if market.vix > config["filters"]["vix_skip_threshold"]:
        logger.info(f"  VIX too high — NO TRADING. Capital protected."); return []

    # Pre-market scan
    candidates = []
    for sym in symbols:
        q = market.get_quote(sym)
        if abs(q["gap_pct"]) <= config["filters"]["max_gap_percent"]:
            candidates.append(sym)
            logger.info(f"    {sym:>12} | {q['last_price']:>8.2f} | Gap:{q['gap_pct']:>+5.1f}% | OK")
        else:
            logger.info(f"    {sym:>12} | {q['last_price']:>8.2f} | Gap:{q['gap_pct']:>+5.1f}% | SKIP")

    orb_done = set()
    while market.is_open():
        market.advance(1)
        trader.check_stops()
        m = market.current_minute
        # ORB window 9:30-10:30 (minute 15-75)
        if 15 <= m <= 75:
            for sym in candidates:
                if sym in orb_done or sym in trader.positions: continue
                sig = check_orb(market, sym, config["strategies"]["orb"])
                if sig:
                    risk = sig["entry"] - sig["sl"]
                    if risk > 0:
                        qty = max(1, int(config["capital"]["total"]*config["capital"]["risk_per_trade"]/risk))
                        trader.buy(sym, sig["entry"], qty, sig["sl"], sig["target"], sig["strat"])
                        orb_done.add(sym)
        # VWAP window 14:00-15:00 (minute 285-345)
        if 285 <= m <= 345 and market.vix < 18:
            for sym in candidates:
                if sym in trader.positions: continue
                sig = check_vwap(market, sym, config["strategies"]["vwap"])
                if sig:
                    risk = sig["entry"] - sig["sl"]
                    if risk > 0:
                        qty = max(1, int(config["capital"]["total"]*config["capital"]["risk_per_trade"]/risk))
                        trader.buy(sym, sig["entry"], qty, sig["sl"], sig["target"], sig["strat"])
        # Square off at 15:10 (minute 355)
        if m >= 355:
            logger.info(f"\n  3:10 PM — Square off"); trader.square_off(); break

    # Summary
    logger.info(f"\n{'='*60}\n  RESULTS — {test_date or date.today()}\n{'='*60}")
    if not trader.closed:
        logger.info("  No trades. Capital preserved.")
    else:
        total = sum(t["net_pnl"] for t in trader.closed)
        wins = [t for t in trader.closed if t["net_pnl"] > 0]
        logger.info(f"  Trades: {len(trader.closed)} | Won: {len(wins)} | Lost: {len(trader.closed)-len(wins)}")
        logger.info(f"  Net P&L: Rs {total:+,.2f}")
        logger.info(f"  Costs: Rs {sum(t['costs'] for t in trader.closed):,.2f}")
        for t in trader.closed:
            e = "WIN " if t["net_pnl"] > 0 else "LOSS"
            logger.info(f"    {e} {t['symbol']:>12} | {t['entry']:.2f} -> {t['exit']:.2f} | Rs {t['net_pnl']:+,.2f} | {t['reason']}")
        pd.DataFrame(trader.closed).to_csv(f"results/paper_{test_date or date.today()}.csv", index=False)
        logger.info(f"  Saved to results/paper_{test_date or date.today()}.csv")
    return trader.closed


def run_multi(days=10, scenario="normal"):
    logger.info(f"\n  MULTI-DAY TEST: {days} days | {scenario}")
    all_t, daily = [], []
    for i in range(days):
        d = (date.today() - timedelta(days=days-i)).isoformat()
        trades = run_simulate(d, scenario)
        all_t.extend(trades)
        daily.append({"date":d,"pnl":sum(t["net_pnl"] for t in trades),"n":len(trades)})
    total = sum(d["pnl"] for d in daily)
    logger.info(f"\n{'='*60}\n  {days}-DAY SUMMARY\n{'='*60}")
    logger.info(f"  Total P&L: Rs {total:+,.2f} | Trades: {len(all_t)} | Win days: {sum(1 for d in daily if d['pnl']>0)}/{days}")
    if all_t:
        w = [t for t in all_t if t["net_pnl"] > 0]
        logger.info(f"  Win rate: {len(w)/len(all_t)*100:.0f}%")


def analyse():
    files = sorted(Path("results").glob("paper_*.csv"))
    if not files: print("No results. Run: python paper_trader.py simulate"); return
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"\n  {len(files)} sessions | {len(all_df)} trades | P&L: Rs {all_df['net_pnl'].sum():+,.2f}")
    print(f"  Win rate: {(all_df['net_pnl']>0).mean()*100:.0f}% | Best: Rs {all_df['net_pnl'].max():+,.2f} | Worst: Rs {all_df['net_pnl'].min():+,.2f}")
    for s, g in all_df.groupby("strategy"):
        print(f"  {s}: {len(g)} trades, Rs {g['net_pnl'].sum():+,.2f}, win {(g['net_pnl']>0).mean()*100:.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paper Trading Test Runner")
    p.add_argument("command", choices=["simulate","multi-test","scenarios","analyse"])
    p.add_argument("--date", default=None)
    p.add_argument("--scenario", default="normal", choices=["normal","volatile","crash","rally","flat"])
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--stocks", nargs="+", default=None)
    a = p.parse_args()
    if a.command == "simulate": run_simulate(a.date, a.scenario, a.stocks)
    elif a.command == "multi-test": run_multi(a.days, a.scenario)
    elif a.command == "scenarios":
        for sc in ["normal","volatile","crash","rally","flat"]: run_simulate(a.date, sc, a.stocks)
    elif a.command == "analyse": analyse()
