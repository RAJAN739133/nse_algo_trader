#!/usr/bin/env python3
"""
SYSTEM TEST — Comprehensive Integration Test
════════════════════════════════════════════════════════════

Tests all components of the algo trading system:
1. Configuration loading
2. Database connectivity
3. Broker authentication
4. ML model loading
5. Strategy initialization
6. Telegram connectivity
7. Claude API (optional)

Usage:
    python test_system.py
    python test_system.py --verbose
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("SystemTest")


class SystemTest:
    """Comprehensive system integration test."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.config = None
        
    def run_all(self) -> bool:
        """Run all tests."""
        print("\n" + "═" * 60)
        print("  NSE ALGO TRADER — System Integration Test")
        print("═" * 60 + "\n")
        
        tests = [
            ("Configuration", self.test_config),
            ("Database", self.test_database),
            ("ML Model", self.test_ml_model),
            ("Strategies", self.test_strategies),
            ("Broker Auth", self.test_broker),
            ("Telegram", self.test_telegram),
            ("Claude API", self.test_claude),
            ("Risk Controls", self.test_risk),
            ("System Monitor", self.test_monitor),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_fn in tests:
            try:
                result, message = test_fn()
                self.results[name] = (result, message)
                
                if result:
                    print(f"  ✅ {name}: {message}")
                    passed += 1
                else:
                    print(f"  ❌ {name}: {message}")
                    failed += 1
                    
            except Exception as e:
                print(f"  ❌ {name}: Exception - {e}")
                self.results[name] = (False, str(e))
                failed += 1
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        
        print("\n" + "─" * 60)
        print(f"  Results: {passed} passed, {failed} failed")
        print("─" * 60)
        
        if failed == 0:
            print("\n  🎉 All tests passed! System is ready.")
        else:
            print(f"\n  ⚠️  {failed} test(s) failed. Please fix before running.")
            
        return failed == 0
        
    def test_config(self) -> tuple:
        """Test configuration loading."""
        import yaml
        
        config_path = Path(__file__).parent / "config" / "config_test.yaml"
        
        if not config_path.exists():
            return False, "config_test.yaml not found"
            
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Required for trading
        required = ["broker", "capital"]
        missing = [k for k in required if k not in self.config]
        
        if missing:
            return False, f"Missing keys: {missing}"
            
        capital = self.config.get("capital", {}).get("total", 0)
        if capital < 1000:
            return False, f"Capital too low: {capital}"
        
        # Warnings for optional but recommended
        warnings = []
        if "telegram" not in self.config:
            warnings.append("telegram")
        if "claude" not in self.config:
            warnings.append("claude")
            
        msg = f"Loaded, capital: Rs {capital:,}"
        if warnings:
            msg += f" (warn: {warnings} not configured)"
            
        return True, msg
        
    def test_database(self) -> tuple:
        """Test database connectivity."""
        from core.trade_database import TradeDB
        
        db = TradeDB()
        
        # Test write
        trade_id = db.log_trade({
            "symbol": "_TEST_",
            "direction": "LONG",
            "strategy": "TEST",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "quantity": 1,
            "entry_time": datetime.now().isoformat(),
            "exit_time": datetime.now().isoformat(),
            "net_pnl": 1.0,
        })
        
        if not trade_id:
            return False, "Failed to write"
            
        # Test read
        trade = db.get_trade(trade_id)
        if not trade:
            return False, "Failed to read"
            
        return True, f"Connected, trade #{trade_id}"
        
    def test_ml_model(self) -> tuple:
        """Test ML model loading."""
        import pickle
        
        model_paths = [
            Path(__file__).parent / "models" / "stock_predictor_v2.pkl",
            Path(__file__).parent / "models" / "stock_predictor.pkl",
        ]
        
        for path in model_paths:
            if path.exists():
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                return True, f"Loaded {path.name}"
                
        return False, "No model found in models/"
        
    def test_strategies(self) -> tuple:
        """Test strategy initialization."""
        from strategies.pro_strategy_v2 import ProStrategyV2
        from strategies.candle_patterns import CandlePatternDetector
        
        strategy = ProStrategyV2({})
        patterns = CandlePatternDetector()
        
        return True, "ProStrategyV2, CandlePatterns OK"
        
    def test_broker(self) -> tuple:
        """Test broker authentication."""
        if not self.config:
            return False, "Config not loaded"
            
        broker_config = self.config.get("broker", {}).get("angel_one", {})
        
        if not broker_config.get("api_key"):
            return False, "API key not configured"
            
        # Don't actually connect in test mode
        mode = self.config.get("broker", {}).get("mode", "paper")
        
        if mode == "paper":
            return True, f"Mode: {mode} (ready to test)"
        else:
            return True, f"Mode: {mode} (LIVE - be careful!)"
            
    def test_telegram(self) -> tuple:
        """Test Telegram configuration."""
        if not self.config:
            return False, "Config not loaded"
            
        tg_config = self.config.get("telegram", {})
        
        if not tg_config.get("bot_token"):
            return True, "Not configured (optional)"
            
        if not tg_config.get("chat_ids"):
            return True, "Token set but no chat_ids"
            
        chat_count = len(tg_config.get("chat_ids", []))
        return True, f"Configured, {chat_count} chat(s)"
        
    def test_claude(self) -> tuple:
        """Test Claude API configuration."""
        if not self.config:
            return False, "Config not loaded"
            
        claude_config = self.config.get("claude", {})
        
        if not claude_config.get("api_key"):
            # Check environment
            if os.environ.get("ANTHROPIC_API_KEY"):
                return True, "API key from environment"
            return False, "API key not configured"
            
        return True, "API key configured"
        
    def test_risk(self) -> tuple:
        """Test risk controls."""
        from core.system_monitor import KillSwitch, TradingHours
        
        # Test kill switch
        if KillSwitch.is_active():
            return False, "Kill switch is currently active!"
            
        # Test trading hours
        is_market = TradingHours.is_market_hours()
        
        return True, f"Kill switch OK, Market: {'OPEN' if is_market else 'CLOSED'}"
        
    def test_monitor(self) -> tuple:
        """Test system monitor."""
        from core.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        metrics = monitor.get_system_metrics()
        
        return True, f"CPU: {metrics.cpu_percent:.1f}%, RAM: {metrics.memory_percent:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="System Integration Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    tester = SystemTest(verbose=args.verbose)
    success = tester.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
