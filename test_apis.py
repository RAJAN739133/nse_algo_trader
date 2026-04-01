#!/usr/bin/env python3
"""
API Health Check — Tests both Claude Brain and Angel One APIs.

Usage:
  python test_apis.py
"""
import os, sys, json, logging, time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config():
    import yaml
    for p in ["config/config_test.yaml", "config/config_prod.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


# ════════════════════════════════════════════════════════════
# TEST 1: CLAUDE API
# ════════════════════════════════════════════════════════════

def test_claude_api(config):
    print("\n" + "=" * 60)
    print("  TEST 1: CLAUDE AI BRAIN API")
    print("=" * 60)

    api_key = config.get("claude", {}).get("api_key", "")
    if not api_key or api_key.startswith("YOUR_") or api_key == "TEST_KEY":
        print("  ❌ No Claude API key found in config")
        print("     Add to config_test.yaml → claude.api_key")
        print("     Get from: https://console.anthropic.com/settings/keys")
        return False

    print(f"  API Key: {api_key[:12]}...{api_key[-4:]}")

    # Step 1: Test raw API call
    print("\n  ── Step 1: Raw API call ──")
    try:
        import requests
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say 'API working' in exactly 2 words."}],
            },
            timeout=30,
        )
        data = resp.json()

        if "error" in data:
            err = data["error"]
            err_type = err.get("type", "unknown")
            err_msg = err.get("message", "No details")
            print(f"  ❌ API Error: {err_type}")
            print(f"     {err_msg}")

            if "credit" in err_msg.lower() or "billing" in err_msg.lower():
                print("\n  💳 BILLING ISSUE — Your credits may have expired")
                print("     Fix: https://console.anthropic.com/settings/plans")
                print("     Add credits or upgrade plan")
            elif "authentication" in err_type.lower() or "invalid" in err_msg.lower():
                print("\n  🔑 AUTH ISSUE — API key is invalid or revoked")
                print("     Fix: https://console.anthropic.com/settings/keys")
                print("     Create a new key")
            elif "rate" in err_type.lower():
                print("\n  ⏱️  RATE LIMITED — Too many requests")
                print("     Wait a minute and try again")
            elif "overloaded" in err_msg.lower():
                print("\n  🔥 API OVERLOADED — Temporary, try again later")
            return False

        # Success
        text = data.get("content", [{}])[0].get("text", "")
        model = data.get("model", "unknown")
        input_tokens = data.get("usage", {}).get("input_tokens", 0)
        output_tokens = data.get("usage", {}).get("output_tokens", 0)
        print(f"  ✅ API Response: \"{text.strip()}\"")
        print(f"     Model: {model}")
        print(f"     Tokens: {input_tokens} in + {output_tokens} out")

    except requests.exceptions.ConnectionError:
        print("  ❌ Connection failed — check internet")
        return False
    except requests.exceptions.Timeout:
        print("  ❌ Request timed out (>30s)")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

    # Step 2: Test Claude Brain class
    print("\n  ── Step 2: ClaudeBrain integration ──")
    try:
        from strategies.claude_brain import ClaudeBrain
        brain = ClaudeBrain(config=config)
        if not brain.enabled:
            print("  ⚠️  ClaudeBrain.enabled = False")
            print("     Check: config → claude.enabled = true")
            print(f"     API key detected: {'Yes' if brain.api_key else 'No'}")
            return False

        print("  ✅ ClaudeBrain initialized (enabled=True)")

        # Test morning analysis
        result = brain.get_morning_analysis(
            vix=15.0, fii_net=-500, recent_trades=[], stock_scores=[]
        )
        print(f"  ✅ Morning analysis returned:")
        print(f"     Risk level: {result.get('risk_level', 'N/A')}")
        print(f"     Max trades: {result.get('max_trades', 'N/A')}")
        print(f"     Notes: {result.get('notes', 'N/A')}")
        print(f"     Validated: {result.get('validated', False)}")

        if result.get("validated"):
            print("\n  🎉 Claude Brain is FULLY WORKING")
            return True
        else:
            print("\n  ⚠️  Response not validated — check API errors above")
            return False

    except Exception as e:
        print(f"  ❌ ClaudeBrain error: {e}")
        return False


# ════════════════════════════════════════════════════════════
# TEST 2: ANGEL ONE API
# ════════════════════════════════════════════════════════════

def test_angel_api(config):
    print("\n" + "=" * 60)
    print("  TEST 2: ANGEL ONE SmartAPI")
    print("=" * 60)

    angel_cfg = config.get("broker", {}).get("angel_one", {})
    if not angel_cfg.get("api_key"):
        # Try angel_config.yaml as fallback
        try:
            import yaml
            with open("config/angel_config.yaml") as f:
                ac = yaml.safe_load(f) or {}
            angel_cfg = ac.get("angel_one", {})
        except:
            pass

    if not angel_cfg.get("api_key"):
        print("  ❌ No Angel One config found")
        print("     Add broker.angel_one section to config_test.yaml")
        return False

    api_key = angel_cfg.get("api_key", "")
    client_code = angel_cfg.get("client_code", "")
    print(f"  API Key: {api_key}")
    print(f"  Client: {client_code}")

    # Step 1: SmartAPI Login
    print("\n  ── Step 1: SmartAPI Login (TOTP) ──")
    try:
        from data.angel_auth import AngelAuth
        auth = AngelAuth(
            api_key=angel_cfg.get("api_key", ""),
            client_code=angel_cfg.get("client_code", ""),
            pin=angel_cfg.get("pin", ""),
            totp_secret=angel_cfg.get("totp_secret", ""),
        )
        smart_api = auth.login()

        if auth.auth_token == "MOCK_TOKEN":
            print("  ⚠️  Running in MOCK mode (credentials may be wrong)")
            print("     Check: api_key, client_code, pin, totp_secret")
            return False

        print(f"  ✅ Login SUCCESS")
        print(f"     Auth token: {auth.auth_token[:20]}...")
        print(f"     Feed token: {auth.feed_token[:20] if auth.feed_token else 'N/A'}...")

    except ImportError as e:
        print(f"  ❌ Missing package: {e}")
        print("     Run: pip install smartapi-python pyotp")
        return False
    except Exception as e:
        print(f"  ❌ Login FAILED: {e}")
        if "Invalid" in str(e):
            print("     → Check TOTP secret / PIN")
        elif "Connection" in str(e):
            print("     → Check internet connection")
        return False

    # Step 2: Profile
    print("\n  ── Step 2: Profile Fetch ──")
    try:
        profile = smart_api.getProfile(auth.refresh_token)
        if profile and profile.get("data"):
            name = profile["data"].get("name", "Unknown")
            email = profile["data"].get("email", "N/A")
            exchanges = profile["data"].get("exchanges", [])
            print(f"  ✅ Profile: {name}")
            print(f"     Email: {email}")
            print(f"     Exchanges: {exchanges}")
        else:
            print(f"  ⚠️  Profile response: {profile}")
    except Exception as e:
        print(f"  ⚠️  Profile error: {e} (session may still work)")

    # Step 3: Symbol mapper
    print("\n  ── Step 3: Instrument Master ──")
    try:
        from data.angel_symbols import AngelSymbolMapper
        mapper = AngelSymbolMapper()
        loaded = mapper.load()
        test_syms = ["RELIANCE", "HDFCBANK", "SBIN", "TCS", "INFY"]
        mapped = 0
        for sym in test_syms:
            token = mapper.get_token(sym)
            if token:
                mapped += 1
        print(f"  ✅ Mapped {mapped}/{len(test_syms)} test symbols")
        if not loaded:
            print("     (Using fallback token map — instrument master download failed)")
    except Exception as e:
        print(f"  ❌ Symbol mapper error: {e}")
        mapper = None

    # Step 4: LTP fetch
    print("\n  ── Step 4: LTP Data ──")
    ltp_success = 0
    for sym in ["RELIANCE", "SBIN", "TCS"]:
        try:
            token = mapper.get_token(sym) if mapper else None
            if not token:
                continue
            trading_sym = mapper.get_trading_symbol(sym)
            data = smart_api.ltpData("NSE", trading_sym, token)
            if data and data.get("data"):
                ltp = data["data"].get("ltp", 0)
                if ltp and float(ltp) > 0:
                    print(f"  ✅ {sym:<12} LTP = Rs {float(ltp):,.2f}")
                    ltp_success += 1
                else:
                    print(f"  ⚠️  {sym:<12} LTP = {ltp} (market may be closed)")
            else:
                print(f"  ⚠️  {sym:<12} No data: {data}")
        except Exception as e:
            print(f"  ❌ {sym:<12} Error: {e}")

    if ltp_success == 0:
        print("  ℹ️  All LTPs are zero/empty — normal if market is closed today")

    # Step 5: Historical candles
    print("\n  ── Step 5: Historical Candles ──")
    try:
        from datetime import timedelta
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=5)
        params = {
            "exchange": "NSE",
            "symboltoken": mapper.get_token("RELIANCE") if mapper else "2885",
            "interval": "FIVE_MINUTE",
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": to_dt.strftime("%Y-%m-%d 15:30"),
        }
        data = smart_api.getCandleData(params)
        if data and data.get("data"):
            candles = data["data"]
            print(f"  ✅ RELIANCE: {len(candles)} candles fetched")
            if candles:
                last = candles[-1]
                print(f"     Latest: {last[0]} O={last[1]} H={last[2]} L={last[3]} C={last[4]} V={last[5]}")
        else:
            print(f"  ⚠️  No candle data: {data}")
    except Exception as e:
        print(f"  ❌ Candle fetch error: {e}")

    # Step 6: Paper order test
    print("\n  ── Step 6: Paper Order (Dummy) ──")
    try:
        from data.angel_broker import AngelBroker
        # Build config with angel_one under broker
        test_config = dict(config)
        if "broker" not in test_config:
            test_config["broker"] = {}
        test_config["broker"]["angel_one"] = angel_cfg
        test_config["broker"]["mode"] = "paper"

        broker = AngelBroker(test_config)
        broker.smart_api = smart_api
        broker.auth = auth
        broker.symbol_mapper = mapper

        # Place dummy BUY
        order = broker.place_order("RELIANCE", "BUY", 5, price=1350.0)
        print(f"  📝 BUY  5x RELIANCE → {order.status} @ Rs {order.fill_price:,.2f} | ID: {order.order_id}")

        # Place dummy SELL to close
        order2 = broker.place_order("RELIANCE", "SELL", 5, price=1355.0)
        print(f"  📝 SELL 5x RELIANCE → {order2.status} @ Rs {order2.fill_price:,.2f} | ID: {order2.order_id}")
        print(f"  💰 P&L: Rs {broker.day_pnl:+,.2f}")

        if order.status == "COMPLETE" and order2.status == "COMPLETE":
            print("  ✅ Paper order flow WORKS")
        else:
            print("  ❌ Paper order flow has issues")
    except Exception as e:
        print(f"  ❌ Broker error: {e}")

    return True


# ════════════════════════════════════════════════════════════
# TEST 3: QUICK CONNECTIVITY CHECK
# ════════════════════════════════════════════════════════════

def test_connectivity():
    print("\n" + "=" * 60)
    print("  TEST 0: CONNECTIVITY")
    print("=" * 60)

    # Check internet
    try:
        import requests
        r = requests.get("https://api.anthropic.com", timeout=5)
        print(f"  ✅ Anthropic API reachable (HTTP {r.status_code})")
    except Exception as e:
        print(f"  ❌ Anthropic API unreachable: {e}")

    try:
        import requests
        r = requests.get("https://apiconnect.angelone.in", timeout=5)
        print(f"  ✅ Angel One API reachable (HTTP {r.status_code})")
    except Exception as e:
        print(f"  ⚠️  Angel One API: {e}")

    # Check packages
    packages = {
        "requests": "API calls",
        "SmartApi": "Angel One (pip install smartapi-python)",
        "pyotp": "TOTP auth",
        "yfinance": "Market data fallback",
        "yaml": "Config files (pyyaml)",
    }
    print("\n  ── Packages ──")
    for pkg, desc in packages.items():
        try:
            mod = __import__(pkg if pkg != "yaml" else "yaml")
            ver = getattr(mod, "__version__", "?")
            print(f"  ✅ {pkg:<12} v{ver:<10} ({desc})")
        except ImportError:
            print(f"  ❌ {pkg:<12} MISSING    ({desc})")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NSE ALGO TRADER — API HEALTH CHECK")
    print(f"  Date: {date.today()} | Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    config = load_config()
    results = {}

    # Test 0: Connectivity
    test_connectivity()

    # Test 1: Claude API
    results["claude_brain"] = test_claude_api(config)

    # Test 2: Angel One
    results["angel_one"] = test_angel_api(config)

    # Summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for name, ok in results.items():
        emoji = "✅" if ok else "❌"
        print(f"  {emoji} {name}")

    all_ok = all(results.values())
    if all_ok:
        print(f"\n  🎉 ALL APIs WORKING — Ready for trading tomorrow!")
    else:
        print(f"\n  ⚠️  Some APIs have issues — see details above")

    # Tomorrow's readiness
    from scheduler import is_trading_day, get_holiday_name
    from datetime import timedelta
    tomorrow = date.today() + timedelta(days=1)
    if is_trading_day(tomorrow):
        print(f"\n  📅 Tomorrow ({tomorrow}): TRADING DAY ✅")
    else:
        reason = "Weekend" if tomorrow.weekday() >= 5 else get_holiday_name(tomorrow)
        print(f"\n  📅 Tomorrow ({tomorrow}): CLOSED ({reason})")


if __name__ == "__main__":
    main()
