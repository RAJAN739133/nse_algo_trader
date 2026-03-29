"""
ML Training Pipeline — Train model to predict stock direction.

Usage:
  python -m data.train_pipeline download   # Guide for data download
  python -m data.train_pipeline train      # Train on downloaded data
  python -m data.train_pipeline demo       # Demo with synthetic data
  python -m data.train_pipeline score      # Score stocks for today
"""
import os, sys, pickle, logging
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.symbols import DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent
MODEL_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "ret_1d","ret_5d","ret_20d","price_vs_sma20","price_vs_sma50",
    "price_vs_sma200","sma20_vs_sma50","atr_pct","vol_20","bb_width",
    "bb_pos","rsi_14","rsi_7","macd_hist","vol_ratio","candle_score",
    "pat_hammer","pat_shooting_star","pat_engulf_bull","pat_engulf_bear",
    "pat_morning_star","pat_evening_star","pat_3_white","pat_3_black",
]


def add_features(df):
    """30+ indicators + 12 candlestick patterns."""
    df = df.sort_values("date").copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["ret_1d"] = c.pct_change()
    df["ret_5d"] = c.pct_change(5)
    df["ret_20d"] = c.pct_change(20)
    for p in [5,10,20,50,200]:
        df[f"sma_{p}"] = c.rolling(p).mean()
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    df["price_vs_sma20"] = (c - df["sma_20"]) / df["sma_20"]
    df["price_vs_sma50"] = (c - df["sma_50"]) / df["sma_50"]
    df["price_vs_sma200"] = (c - df["sma_200"]) / df["sma_200"]
    df["sma20_vs_sma50"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / c
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    std20 = c.rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2*std20
    df["bb_lower"] = df["sma_20"] - 2*std20
    df["bb_width"] = (df["bb_upper"]-df["bb_lower"]) / df["sma_20"]
    df["bb_pos"] = (c-df["bb_lower"]) / (df["bb_upper"]-df["bb_lower"])
    for period in [7,14]:
        delta = c.diff()
        gain = delta.where(delta>0,0).ewm(alpha=1/period,min_periods=period).mean()
        loss = (-delta.where(delta<0,0)).ewm(alpha=1/period,min_periods=period).mean()
        df[f"rsi_{period}"] = 100 - 100/(1 + gain/loss.replace(0,np.nan))
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    if "volume" in df.columns and not df["volume"].isna().all():
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0,np.nan)
    else:
        df["vol_ratio"] = 1.0
    # Candlestick patterns
    body = abs(c-o); rng = h-l
    uw = h - pd.concat([c,o],axis=1).max(axis=1)
    lw = pd.concat([c,o],axis=1).min(axis=1) - l
    po, pc = o.shift(1), c.shift(1)
    df["pat_doji"] = (body < rng*0.1).astype(int)
    df["pat_hammer"] = ((lw>body*2)&(uw<body*0.5)&(c>o)).astype(int)
    df["pat_shooting_star"] = -((uw>body*2)&(lw<body*0.5)&(c<o)).astype(int)
    df["pat_engulf_bull"] = ((c>o)&(pc<po)&(o<=pc)&(c>=po)).astype(int)
    df["pat_engulf_bear"] = -((c<o)&(pc>po)&(o>=pc)&(c<=po)).astype(int)
    df["pat_morning_star"] = ((c.shift(2)<o.shift(2))&(body.shift(1)<rng.shift(1)*0.2)&(c>o)&(c>(o.shift(2)+c.shift(2))/2)).astype(int)
    df["pat_evening_star"] = -((c.shift(2)>o.shift(2))&(body.shift(1)<rng.shift(1)*0.2)&(c<o)&(c<(o.shift(2)+c.shift(2))/2)).astype(int)
    df["pat_3_white"] = ((c>o)&(c.shift(1)>o.shift(1))&(c.shift(2)>o.shift(2))&(c>c.shift(1))&(c.shift(1)>c.shift(2))).astype(int)
    df["pat_3_black"] = -((c<o)&(c.shift(1)<o.shift(1))&(c.shift(2)<o.shift(2))&(c<c.shift(1))&(c.shift(1)<c.shift(2))).astype(int)
    pcols = [x for x in df.columns if x.startswith("pat_")]
    df["candle_score"] = df[pcols].sum(axis=1)
    df["target_dir"] = (c.shift(-1)/c - 1 > 0).astype(int)
    return df


def generate_synthetic(symbols, years=10):
    """Synthetic data for demo/testing."""
    np.random.seed(42); all_data = []
    n = years * 252
    dates = pd.bdate_range("2015-01-01", periods=n)
    mkt = np.random.randn(n)*0.012 + 0.0003
    for i, crash in [(500,-0.08),(800,-0.10),(1200,-0.12),(1250,0.08),(1800,-0.06),(2200,-0.04)]:
        if i < n: mkt[i] = crash
    for sym in symbols:
        vol = np.random.uniform(0.012, 0.022)
        ret = mkt*(0.6+np.random.rand()*0.8) + np.random.randn(n)*vol + 0.0003
        close = 100*np.exp(np.cumsum(ret))
        iv = np.abs(np.random.randn(n))*vol*close
        all_data.append(pd.DataFrame({"date":dates[:n],"symbol":sym,
            "open":np.round(close+np.random.randn(n)*vol*close*0.3,2),
            "high":np.round(close+iv*0.6,2),"low":np.round(close-iv*0.4,2),
            "close":np.round(close,2),"volume":np.random.lognormal(14,1,n).astype(int)}))
    return pd.concat(all_data, ignore_index=True)


def train_model(df):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
    avail = [c for c in FEATURE_COLS if c in df.columns]
    mdf = df.dropna(subset=avail+["target_dir"]).copy()
    X, y = mdf[avail].fillna(0), mdf["target_dir"]
    print(f"  Features: {len(avail)} | Samples: {len(X):,} | Up: {y.mean():.1%}")
    tscv = TimeSeriesSplit(n_splits=3)
    for fold,(tri,tei) in enumerate(tscv.split(X),1):
        m = GradientBoostingClassifier(n_estimators=50,max_depth=3,learning_rate=0.1,random_state=42)
        m.fit(X.iloc[tri],y.iloc[tri])
        print(f"  Fold {fold}: {accuracy_score(y.iloc[tei],m.predict(X.iloc[tei])):.1%}")
    final = GradientBoostingClassifier(n_estimators=50,max_depth=3,learning_rate=0.1,random_state=42)
    final.fit(X,y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR/"stock_predictor.pkl","wb") as f:
        pickle.dump({"model":final,"features":avail},f)
    print(f"  Model saved: {MODEL_DIR/'stock_predictor.pkl'}")
    imp = pd.Series(final.feature_importances_,index=avail).sort_values(ascending=False)
    print("\n  Top features:")
    for feat,val in imp.head(10).items():
        print(f"    {feat:<22} {val:.3f} {'█'*int(val*80)}")
    return final, avail


def score_stocks(df, model=None, features=None, vix=15):
    scores = []
    for sym in df["symbol"].unique():
        s = df[df["symbol"]==sym]
        if len(s) < 50: continue
        lat = s.iloc[-1]; sc = 0
        if model and features:
            try:
                x = np.nan_to_num(lat[features].values.reshape(1,-1))
                sc += int(model.predict_proba(x)[0][1]*30)
            except: sc += 15
        tech = 0
        if lat.get("price_vs_sma20",0)>0: tech+=5
        if lat.get("price_vs_sma50",0)>0: tech+=5
        if lat.get("sma20_vs_sma50",0)>0: tech+=5
        rsi = lat.get("rsi_14",50)
        tech += 10 if rsi<30 else (8 if rsi<40 else (5 if rsi<60 else 0))
        sc += min(25,tech)
        sc += min(15, max(0, int(lat.get("candle_score",0)*5+7)))
        vr = lat.get("vol_ratio",1.0); vr = 1.0 if pd.isna(vr) else vr
        sc += min(15, int(min(vr,3)*5))
        sc += max(0, min(15, 12 if vix<15 else (7 if vix<20 else 4)))
        scores.append({"symbol":sym,"score":min(100,sc),"price":round(lat.get("close",0),2),
            "rsi":round(rsi,1),"strategy":"ORB" if rsi>35 else "VWAP"})
    return pd.DataFrame(scores).sort_values("score",ascending=False)


def main():
    action = sys.argv[1] if len(sys.argv)>1 else "demo"
    if action == "download":
        print("""
  On your machine run:
    pip install yfinance
    python -c "
    import yfinance as yf
    for s in ['RELIANCE','HDFCBANK','TCS','SBIN','INFY','ICICIBANK','ITC','BAJFINANCE']:
        df = yf.download(f'{s}.NS', start='2001-01-01')
        df.to_csv(f'data/{s}.csv')
        print(f'{s}: {len(df)} rows')
    "
        """); return

    print(f"\n{'='*60}\n  ML PIPELINE — {action.upper()}\n{'='*60}")
    stocks = DEFAULT_UNIVERSE[:10]
    raw = generate_synthetic(stocks, years=10)
    print(f"  Data: {len(raw):,} rows | {raw['symbol'].nunique()} stocks")
    featured = pd.concat([add_features(raw[raw["symbol"]==s].copy()) for s in raw["symbol"].unique()])
    print(f"  Features: {len(featured.columns)} columns")

    if action in ("demo","train"):
        model, feats = train_model(featured)
    if action in ("demo","score"):
        model_path = MODEL_DIR/"stock_predictor.pkl"
        if action=="score" and model_path.exists():
            with open(model_path,"rb") as f: d=pickle.load(f)
            model,feats = d["model"],d["features"]
        elif action=="score": model,feats = None,None
        sc = score_stocks(featured, model if 'model' in dir() else None, feats if 'feats' in dir() else None)
        print(f"\n  {'Rank':<5}{'Symbol':<12}{'Score':>6}{'RSI':>6}{'Strategy':<8}")
        print("  "+"-"*40)
        for i,(_,r) in enumerate(sc.head(15).iterrows(),1):
            star = " *" if r["score"]>=60 else ""
            print(f"  {i:<5}{r['symbol']:<12}{r['score']:>5.0f}{r['rsi']:>6.1f} {r['strategy']:<8}{star}")
        picks = sc[sc["score"]>=60].head(3)
        print(f"\n  TODAY'S PICKS: {len(picks)} stocks" if len(picks) else "\n  No stocks qualify — sit out")
        for _,p in picks.iterrows(): print(f"    {p['symbol']} score={p['score']} via {p['strategy']}")

if __name__ == "__main__": main()
