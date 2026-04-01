#!/usr/bin/env python3
"""
FAST pipeline: Retrain with enhanced candle features → 7-day V3 backtest.
Uses already-cached yfinance data (150 stocks). NSE enrichment at trade-time only.
"""
import os, sys, pickle, logging, json, warnings
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from config.symbols import NIFTY_50
from data.data_loader import DataLoader
from data.train_pipeline import add_features, FEATURE_COLS

EXTRA_FEATURES = [
    "pat_doji","pat_dragonfly","pat_gravestone",
    "pat_bullish_harami","pat_bearish_harami",
    "pat_tweezer_top","pat_tweezer_bottom",
    "gap_pct","body_to_range","upper_shadow_pct","lower_shadow_pct",
    "consecutive_green","consecutive_red",
]
ALL_FEATURES = FEATURE_COLS + EXTRA_FEATURES

def add_candle_features(df):
    df = add_features(df)
    c,h,l,o = df["close"],df["high"],df["low"],df["open"]
    body = abs(c-o); rng = h-l
    uw = h - pd.concat([c,o],axis=1).max(axis=1)
    lw = pd.concat([c,o],axis=1).min(axis=1) - l
    df["pat_doji"] = (body < rng*0.1).astype(int)
    df["pat_dragonfly"] = ((lw>rng*0.6)&(uw<rng*0.1)&(body<rng*0.1)).astype(int)
    df["pat_gravestone"] = -((uw>rng*0.6)&(lw<rng*0.1)&(body<rng*0.1)).astype(int)
    po,pc = o.shift(1),c.shift(1)
    prev_body = abs(pc-po)
    df["pat_bullish_harami"] = ((pc<po)&(c>o)&(o>pc)&(c<po)&(body<prev_body*0.5)).astype(int)
    df["pat_bearish_harami"] = -((pc>po)&(c<o)&(o<pc)&(c>po)&(body<prev_body*0.5)).astype(int)
    df["pat_tweezer_top"] = -((abs(h-h.shift(1))/h<0.001)&(c.shift(1)>o.shift(1))&(c<o)).astype(int)
    df["pat_tweezer_bottom"] = ((abs(l-l.shift(1))/l<0.001)&(c.shift(1)<o.shift(1))&(c>o)).astype(int)
    df["gap_pct"] = (o-c.shift(1))/c.shift(1)
    df["body_to_range"] = np.where(rng>0, body/rng, 0)
    df["upper_shadow_pct"] = np.where(rng>0, uw/rng, 0)
    df["lower_shadow_pct"] = np.where(rng>0, lw/rng, 0)
    green = (c>o).astype(int); red = (c<o).astype(int)
    cg,cr = [0],[0]
    for i in range(1,len(df)):
        cg.append(cg[-1]+1 if green.iloc[i] else 0)
        cr.append(cr[-1]+1 if red.iloc[i] else 0)
    df["consecutive_green"]=cg; df["consecutive_red"]=cr
    for col in ALL_FEATURES:
        if col not in df.columns: df[col]=0
    return df

def retrain():
    print("\n═══ STEP 1: RETRAIN ML MODEL (enhanced candle features) ═══\n")
    loader = DataLoader()
    df = loader.load_backtest_data(NIFTY_50, target_date=date.today().isoformat())
    featured=[]
    for sym in (df["symbol"].unique() if not df.empty else []):
        sdf=df[df["symbol"]==sym].copy()
        if len(sdf)>50: featured.append(add_candle_features(sdf))
    if not featured: return None,None
    all_feat = pd.concat(featured, ignore_index=True)
    all_feat["future_ret"] = all_feat.groupby("symbol")["close"].pct_change(1).shift(-1)
    all_feat["target"] = (all_feat["future_ret"]>0).astype(int)
    all_feat = all_feat.dropna(subset=["target"])
    avail = [c for c in ALL_FEATURES if c in all_feat.columns]
    X = all_feat[avail].fillna(0); y = all_feat["target"]
    split = int(len(X)*0.8)
    X_tr,X_te = X.iloc[:split],X.iloc[split:]
    y_tr,y_te = y.iloc[:split],y.iloc[split:]
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    model = GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
        subsample=0.8,min_samples_leaf=20,random_state=42)
    model.fit(X_tr,y_tr)
    tr_acc = accuracy_score(y_tr,model.predict(X_tr))
    te_acc = accuracy_score(y_te,model.predict(X_te))
    print(f"  Features: {len(avail)} (was 24, now +{len(EXTRA_FEATURES)} candle patterns)")
    print(f"  Samples: {len(X):,} ({len(X_tr):,} train, {len(X_te):,} test)")
    print(f"  Train acc: {tr_acc:.1%} | Test acc: {te_acc:.1%}")
    imp = pd.Series(model.feature_importances_,index=avail).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for f,s in imp.head(10).items(): print(f"    {f:<25} {s:.4f}")
    with open("models/stock_predictor.pkl","wb") as f:
        pickle.dump({"model":model,"features":avail,"enhanced":True,
            "train_acc":tr_acc,"test_acc":te_acc,"trained_on":str(date.today())},f)
    print(f"\n  Saved: models/stock_predictor.pkl")
    return model,avail

def find_trading_days(n=7):
    print(f"\n═══ STEP 2: FIND LAST {n} TRADING DAYS ═══\n")
    import yfinance as yf
    df = yf.Ticker("^NSEI").history(period="1mo",interval="1d")
    days = sorted([d.date() for d in df.index if d.date() < date.today()])
    last = days[-n:] if len(days)>=n else days
    for d in last: print(f"  {d} ({d.strftime('%A')})")
    return last

def run_v3_day(target_date):
    from live_paper_v3 import run
    run(backtest_date=target_date.isoformat())
    csv = Path(f"results/live_v3_{target_date}.csv")
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()

def main():
    print(f"\n{'='*60}")
    print(f"  ENHANCED RETRAIN + 7-DAY V3 BACKTEST")
    print(f"  Each day = fresh trading day, Telegram on every trade")
    print(f"{'='*60}")

    model,feats = retrain()
    if model is None: return

    days = find_trading_days(7)

    print(f"\n═══ STEP 3: V3 BACKTEST — {len(days)} DAYS ═══")
    print(f"  No peeking. Each day treated as live.\n")

    results = []
    for day in days:
        print(f"\n  {'─'*50}")
        print(f"  {day} ({day.strftime('%A')})")
        print(f"  {'─'*50}")
        trades = run_v3_day(day)
        if len(trades)>0:
            pnl=trades["net_pnl"].sum(); w=(trades["net_pnl"]>0).sum()
            wr=w/len(trades)*100
            results.append({"date":str(day),"day":day.strftime("%a"),"trades":len(trades),
                "wins":int(w),"losses":len(trades)-int(w),"pnl":round(float(pnl),2),"wr":round(wr,1)})
        else:
            results.append({"date":str(day),"day":day.strftime("%a"),"trades":0,"wins":0,"losses":0,"pnl":0,"wr":0})

    total_pnl=sum(r["pnl"] for r in results)
    total_t=sum(r["trades"] for r in results)
    total_w=sum(r["wins"] for r in results)
    avg_wr=total_w/total_t*100 if total_t>0 else 0

    print(f"\n{'='*60}")
    print(f"  7-DAY V3 RESULTS ({len(feats)} features, candle patterns)")
    print(f"{'='*60}")
    print(f"\n  {'Date':<14}{'Day':<6}{'Trades':>6}{'Wins':>5}{'WR':>5}{'P&L':>12}")
    print(f"  {'─'*48}")
    for r in results:
        e="✅" if r["pnl"]>0 else ("❌" if r["pnl"]<0 else "➖")
        print(f"  {e}{r['date']:<13}{r['day']:<6}{r['trades']:>6}{r['wins']:>5}{r['wr']:>4.0f}%{r['pnl']:>+11,.2f}")
    print(f"  {'─'*48}")
    print(f"  {'TOTAL':<20}{total_t:>6}{total_w:>5}{avg_wr:>4.0f}%{total_pnl:>+11,.2f}")
    print(f"\n  Capital: Rs 1,00,000 | Return: {total_pnl/100000*100:+.2f}%")

    with open("results/7day_backtest_summary.json","w") as f:
        json.dump({"results":results,"total_pnl":total_pnl,"total_trades":total_t,
            "win_rate":avg_wr,"features":len(feats),"model":"enhanced_candle"},f,indent=2)
    print(f"  Saved: results/7day_backtest_summary.json")

    from live_paper_v3 import send_telegram, load_config
    cfg=load_config()
    lines=[f"📊 7-DAY V3 BACKTEST",f"Model: {len(feats)} features (candle patterns)",""]
    for r in results:
        e="✅" if r["pnl"]>0 else "❌" if r["pnl"]<0 else "➖"
        lines.append(f"{e} {r['date']} {r['day']} | {r['trades']}T {r['wins']}W | Rs {r['pnl']:+,.0f}")
    lines.append(f"\n💰 TOTAL: Rs {total_pnl:+,.0f} | {total_t}T | WR: {avg_wr:.0f}%")
    send_telegram("\n".join(lines),cfg)

if __name__=="__main__": main()
