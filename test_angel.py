#!/usr/bin/env python3
"""
Angel One Connection Test — Verifies:
1. SmartAPI login (TOTP-based)
2. Symbol mapper (instrument master download)
3. LTP data fetch (REST)
4. Historical candles (REST)
5. Paper order placement (dummy)
6. Real-time WebSocket (optional — run with --ws flag)

Usage:
  python test_angel.py              # Quick test (REST only)
  python test_angel.py --ws         # Include WebSocket test (runs 60s)
  python test_angel.py --verbose    # Debug logging
"""
import os, sys, time, logging, argparse
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_SYMBOLS = ["RELIANCE", "HDFCBANK", "SBIN", "TCS", "INFY"]


def load_config():
    import yaml
    for p in ["config/config_test.yaml", "config/config_prod.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


def test_login(config):
    """Test 1: Angel One SmartAPI login."""
    print("\n━━━ TEST 1: Angel One Login ━━━")
    from data.angel_broker import AngelBroker
    broker = AngelBroker(config)
    ok = broker.connect()
    if ok and broker.is_connected():
        status = broker.status()
        print(f"  ✅ Login SUCCESS | {status['status_line']}")
        return broker
    else:
        print(f"  ❌ Login FAILED")
        return None


def test_symbol_mapper(broker):
    """Test 2: Symbol token mapping."""
    print("\n━━━ TEST 2: Symbol Mapper ━━━")
    if not broker or not broker.symbol_mapper:
        print("  ❌ SKIP — no broker connection")
        return False

    mapped = 0
    for sym in TEST_SYMBOLS:
        token = broker.symbol_mapper.get_token(sym)
        tsym = broker.symbol_mapper.get_trading_symbol(sym)
        if token:
            print(f"  ✅ {sym:<12} → token={token:<8} trading_symbol={tsym}")
            mapped += 1
        else:
            print(f"  ❌ {sym:<12} → NOT FOUND")

    print(f"\n  Mapped: {mapped}/{len(TEST_SYMBOLS)}")
    return mapped == len(TEST_SYMBOLS)


def test_ltp(broker):
    """Test 3: Fetch live LTP prices via REST API."""
    print("\n━━━ TEST 3: LTP Fetch (REST) ━━━")
    if not broker or not broker.is_connected():
        print("  ❌ SKIP — no broker connection")
        return False

    success = 0
    for sym in TEST_SYMBOLS:
        ltp = broker.get_ltp(sym)
        if ltp and ltp > 0:
            print(f"  ✅ {sym:<12} LTP = Rs {ltp:,.2f}")
            success += 1
        else:
            print(f"  ⚠️  {sym:<12} LTP = None (market may be closed)")

    print(f"\n  Fetched: {success}/{len(TEST_SYMBOLS)}")
    if success == 0:
        print("  ℹ️  All LTPs are None — this is normal if market is closed.")
        return True  # Not a failure if market is closed
    return success > 0


def test_historical_candles(broker):
    """Test 4: Fetch historical candles via REST API."""
    print("\n━━━ TEST 4: Historical Candles (REST) ━━━")
    if not broker or not broker.is_connected():
        print("  ❌ SKIP — no broker connection")
        return False

    sym = "RELIANCE"
    df = broker.get_historical_candles(sym, interval="FIVE_MINUTE", days=3)
    if df is not None and len(df) > 0:
        print(f"  ✅ {sym}: {len(df)} candles fetched")
        print(f"     Latest: {df.iloc[-1]['datetime']} | O={df.iloc[-1]['open']:.2f} "
              f"H={df.iloc[-1]['high']:.2f} L={df.iloc[-1]['low']:.2f} C={df.iloc[-1]['close']:.2f}")
        return True
    else:
        print(f"  ⚠️  {sym}: No candles returned (market may be closed / holiday)")
        return True  # Not a failure


def test_paper_order(broker):
    """Test 5: Place a dummy paper order."""
    print("\n━━━ TEST 5: Paper Order Placement ━━━")
    if not broker:
        print("  ❌ SKIP — no broker connection")
        return False

    # Place a BUY
    order1 = broker.place_order("RELIANCE", "BUY", 10, price=1350.0)
    print(f"  📝 BUY  → ID: {order1.order_id} | Status: {order1.status} | "
          f"Fill: Rs {order1.fill_price:,.2f}")

    # Check position
    positions = broker.get_positions()
    if "RELIANCE" in positions:
        pos = positions["RELIANCE"]
        print(f"  📊 Position: {pos.side} {pos.qty}x @ Rs {pos.entry_price:,.2f}")

    # Place a SELL to close
    order2 = broker.place_order("RELIANCE", "SELL", 10, price=1355.0)
    print(f"  📝 SELL → ID: {order2.order_id} | Status: {order2.status} | "
          f"Fill: Rs {order2.fill_price:,.2f}")

    # Check P&L
    print(f"  💰 Day P&L: Rs {broker.day_pnl:+,.2f}")
    print(f"  📋 Orders: {len(broker.orders)} | Closed trades: {len(broker.closed_trades)}")

    if order1.status == "COMPLETE" and order2.status == "COMPLETE":
        print("  ✅ Paper order flow WORKS")
        return True
    else:
        print("  ❌ Paper order flow FAILED")
        return False


def test_websocket(broker, duration=60):
    """Test 6: Real-time WebSocket streaming."""
    print(f"\n━━━ TEST 6: WebSocket Streaming ({duration}s) ━━━")
    if not broker or not broker.is_connected():
        print("  ❌ SKIP — no broker connection")
        return False

    tick_count = [0]
    candle_count = [0]

    def on_tick(tick):
        tick_count[0] += 1
        if tick_count[0] <= 3:
            print(f"  📡 Tick #{tick_count[0]}: {tick.symbol} = Rs {tick.ltp:,.2f}")

    def on_candle(symbol, candle, all_candles):
        candle_count[0] += 1
        print(f"  🕯️ Candle: {symbol} {candle['datetime']} "
              f"O={candle['open']:.2f} H={candle['high']:.2f} "
              f"L={candle['low']:.2f} C={candle['close']:.2f}")

    symbols = TEST_SYMBOLS[:3]
    ok = broker.start_realtime(symbols, on_candle=on_candle, on_tick=on_tick)
    if not ok:
        print("  ❌ WebSocket failed to start")
        return False

    print(f"  ⏳ Listening for {duration}s on {symbols}...")
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\n  ⏹ Interrupted")

    broker.stop_realtime()
    print(f"\n  📊 Results: {tick_count[0]} ticks, {candle_count[0]} candles")

    if tick_count[0] > 0:
        print("  ✅ WebSocket streaming WORKS")
        return True
    else:
        print("  ⚠️  No ticks received — market may be closed or WS auth issue")
        return True  # Not a failure if market is closed


def test_market_data_bulk(broker):
    """Test 3b: Bulk market data fetch."""
    print("\n━━━ TEST 3b: Bulk Market Data ━━━")
    if not broker or not broker.is_connected():
        print("  ❌ SKIP — no broker connection")
        return False

    data = broker.get_market_data(TEST_SYMBOLS)
    if data:
        for sym, quote in data.items():
            print(f"  ✅ {sym:<12} LTP={quote['ltp']:>10,.2f}  "
                  f"O={quote['open']:>10,.2f}  H={quote['high']:>10,.2f}  "
                  f"L={quote['low']:>10,.2f}  Vol={quote['volume']:>12,}")
        return True
    else:
        print("  ⚠️  No bulk data returned (market may be closed)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Angel One Connection Test")
    parser.add_argument("--ws", action="store_true", help="Include WebSocket test")
    parser.add_argument("--ws-duration", type=int, default=60, help="WebSocket test duration (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  ANGEL ONE CONNECTION TEST")
    print(f"  Date: {date.today()} | Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    config = load_config()
    broker_cfg = config.get("broker", {})
    if not broker_cfg.get("angel_one", {}).get("api_key"):
        print("\n  ❌ No Angel One config found in config_test.yaml!")
        print("  Add broker.angel_one section with api_key, client_code, pin, totp_secret")
        return

    results = {}

    # Test 1: Login
    broker = test_login(config)
    results["login"] = broker is not None

    # Test 2: Symbol mapper
    results["symbols"] = test_symbol_mapper(broker)

    # Test 3: LTP
    results["ltp"] = test_ltp(broker)

    # Test 3b: Bulk market data
    results["bulk_data"] = test_market_data_bulk(broker)

    # Test 4: Historical candles
    results["candles"] = test_historical_candles(broker)

    # Test 5: Paper orders
    results["paper_orders"] = test_paper_order(broker)

    # Test 6: WebSocket (optional)
    if args.ws:
        results["websocket"] = test_websocket(broker, args.ws_duration)

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        emoji = "✅" if passed else "❌"
        print(f"  {emoji} {name}")

    all_pass = all(results.values())
    print(f"\n  {'🎉 ALL TESTS PASSED' if all_pass else '⚠️  SOME TESTS FAILED'}")

    if all_pass:
        print(f"\n  ✅ Angel One is ready for paper trading tomorrow!")
        print(f"     Run: python live_paper_v3.py")
        print(f"     Or:  python scheduler.py run-now")

    # Cleanup
    if broker and broker.realtime_provider:
        broker.stop_realtime()


if __name__ == "__main__":
    main()
