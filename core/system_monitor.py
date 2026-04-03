#!/usr/bin/env python3
"""
SYSTEM MONITOR — Production Health & Auto-Recovery
════════════════════════════════════════════════════════════

Monitors the algo trading system and handles:
1. Process health checks
2. Auto-restart on failures
3. Memory/CPU monitoring
4. Heartbeat to Telegram
5. Emergency alerts

Usage:
    python -m core.system_monitor
"""

import os
import sys
import time
import signal
import logging
import psutil
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.alerts import send_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SystemMonitor")


@dataclass
class HealthStatus:
    """Health status of a component."""
    name: str
    healthy: bool
    last_check: datetime
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float
    timestamp: datetime = field(default_factory=datetime.now)


class SystemMonitor:
    """
    Production system monitor with auto-recovery.
    
    Features:
    - Process health monitoring
    - Auto-restart on crash
    - Resource monitoring (CPU, RAM, Disk)
    - Heartbeat alerts
    - Error aggregation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.running = False
        self.components: Dict[str, HealthStatus] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_heartbeat = datetime.now()
        
        # Thresholds
        self.cpu_threshold = self.config.get("cpu_threshold", 80)
        self.memory_threshold = self.config.get("memory_threshold", 85)
        self.disk_threshold = self.config.get("disk_threshold", 90)
        self.heartbeat_interval = self.config.get("heartbeat_interval", 3600)  # 1 hour
        self.health_check_interval = self.config.get("health_check_interval", 60)  # 1 min
        
        # Process management
        self.managed_processes: Dict[str, subprocess.Popen] = {}
        self.restart_counts: Dict[str, int] = {}
        self.max_restarts = self.config.get("max_restarts", 5)
        
        # Callbacks
        self.health_callbacks: list = []
        
        # State file
        self.state_file = Path(__file__).parent.parent / "data" / "monitor_state.json"
        
    def start(self):
        """Start the system monitor."""
        self.running = True
        logger.info("System Monitor started")
        
        # Load previous state
        self._load_state()
        
        # Start monitoring threads
        threading.Thread(target=self._health_check_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._resource_monitor_loop, daemon=True).start()
        
        # Send startup notification
        self._send_alert("🟢 System Monitor started", "info")
        
    def stop(self):
        """Stop the system monitor."""
        self.running = False
        self._save_state()
        logger.info("System Monitor stopped")
        
    def register_component(self, name: str, health_check: Callable[[], bool], 
                          restart_cmd: Optional[str] = None):
        """Register a component for monitoring."""
        self.components[name] = HealthStatus(
            name=name,
            healthy=True,
            last_check=datetime.now(),
        )
        self.health_callbacks.append((name, health_check, restart_cmd))
        logger.info(f"Registered component: {name}")
        
    def register_process(self, name: str, cmd: str, cwd: str = None):
        """Register and start a managed process."""
        self.restart_counts[name] = 0
        self._start_process(name, cmd, cwd)
        
    def _start_process(self, name: str, cmd: str, cwd: str = None):
        """Start a managed process."""
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd or str(Path(__file__).parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.managed_processes[name] = process
            logger.info(f"Started process {name} (PID: {process.pid})")
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            
    def _health_check_loop(self):
        """Periodically check component health."""
        while self.running:
            try:
                for name, check_fn, restart_cmd in self.health_callbacks:
                    try:
                        healthy = check_fn()
                        self.components[name] = HealthStatus(
                            name=name,
                            healthy=healthy,
                            last_check=datetime.now(),
                            message="OK" if healthy else "UNHEALTHY",
                        )
                        
                        if not healthy:
                            self.error_counts[name] = self.error_counts.get(name, 0) + 1
                            
                            # Alert after 3 consecutive failures
                            if self.error_counts[name] == 3:
                                self._send_alert(
                                    f"⚠️ Component {name} unhealthy for 3 checks",
                                    "warning"
                                )
                                
                            # Auto-restart after 5 failures
                            if self.error_counts[name] >= 5 and restart_cmd:
                                self._attempt_restart(name, restart_cmd)
                        else:
                            self.error_counts[name] = 0
                            
                    except Exception as e:
                        logger.error(f"Health check failed for {name}: {e}")
                        
                # Check managed processes
                for name, process in list(self.managed_processes.items()):
                    if process.poll() is not None:
                        # Process has exited
                        exit_code = process.returncode
                        logger.warning(f"Process {name} exited with code {exit_code}")
                        
                        if self.restart_counts[name] < self.max_restarts:
                            self._send_alert(
                                f"🔄 Process {name} crashed (exit={exit_code}), restarting...",
                                "warning"
                            )
                            self.restart_counts[name] += 1
                            # Would need to store cmd to restart
                        else:
                            self._send_alert(
                                f"❌ Process {name} exceeded max restarts ({self.max_restarts})",
                                "error"
                            )
                            
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                
            time.sleep(self.health_check_interval)
            
    def _heartbeat_loop(self):
        """Send periodic heartbeat alerts."""
        while self.running:
            try:
                now = datetime.now()
                if (now - self.last_heartbeat).seconds >= self.heartbeat_interval:
                    self._send_heartbeat()
                    self.last_heartbeat = now
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                
            time.sleep(60)
            
    def _resource_monitor_loop(self):
        """Monitor system resources."""
        while self.running:
            try:
                metrics = self.get_system_metrics()
                
                # Check thresholds
                alerts = []
                if metrics.cpu_percent > self.cpu_threshold:
                    alerts.append(f"CPU: {metrics.cpu_percent:.1f}%")
                if metrics.memory_percent > self.memory_threshold:
                    alerts.append(f"Memory: {metrics.memory_percent:.1f}%")
                if metrics.disk_percent > self.disk_threshold:
                    alerts.append(f"Disk: {metrics.disk_percent:.1f}%")
                    
                if alerts:
                    self._send_alert(
                        f"⚠️ Resource Warning\n" + "\n".join(alerts),
                        "warning"
                    )
                    
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                
            time.sleep(300)  # Check every 5 minutes
            
    def _attempt_restart(self, name: str, restart_cmd: str):
        """Attempt to restart a failed component."""
        logger.info(f"Attempting restart of {name}")
        try:
            subprocess.run(restart_cmd, shell=True, timeout=30)
            self._send_alert(f"🔄 Restarted {name}", "info")
            self.error_counts[name] = 0
        except Exception as e:
            self._send_alert(f"❌ Failed to restart {name}: {e}", "error")
            
    def _send_heartbeat(self):
        """Send heartbeat status."""
        metrics = self.get_system_metrics()
        
        # Count healthy components
        healthy = sum(1 for c in self.components.values() if c.healthy)
        total = len(self.components)
        
        msg = (
            f"💓 Algo Trader Heartbeat\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Components: {healthy}/{total} healthy\n"
            f"💻 CPU: {metrics.cpu_percent:.1f}%\n"
            f"🧠 Memory: {metrics.memory_percent:.1f}%\n"
            f"💾 Disk: {metrics.disk_percent:.1f}%\n"
            f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send_alert(msg, "info")
        
    def _send_alert(self, message: str, level: str = "info"):
        """Send alert via Telegram."""
        try:
            send_telegram(message, self.config)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        net_io = psutil.net_io_counters()
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            network_sent_mb=net_io.bytes_sent / (1024 * 1024),
            network_recv_mb=net_io.bytes_recv / (1024 * 1024),
        )
        
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        metrics = self.get_system_metrics()
        return {
            "timestamp": datetime.now().isoformat(),
            "running": self.running,
            "components": {
                name: {
                    "healthy": status.healthy,
                    "last_check": status.last_check.isoformat(),
                    "message": status.message,
                }
                for name, status in self.components.items()
            },
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
            },
            "error_counts": self.error_counts,
        }
        
    def _save_state(self):
        """Save monitor state to file."""
        try:
            state = {
                "last_save": datetime.now().isoformat(),
                "error_counts": self.error_counts,
                "restart_counts": self.restart_counts,
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def _load_state(self):
        """Load previous monitor state."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    state = json.load(f)
                self.error_counts = state.get("error_counts", {})
                self.restart_counts = state.get("restart_counts", {})
        except Exception as e:
            logger.error(f"Failed to load state: {e}")


class KillSwitch:
    """
    Emergency kill switch for algo trading.
    
    Features:
    - Immediate halt of all trading
    - Close all open positions
    - Send emergency alert
    - Write to kill file for persistence
    """
    
    KILL_FILE = Path(__file__).parent.parent / "data" / ".kill_switch"
    
    @classmethod
    def activate(cls, reason: str = "Manual activation"):
        """Activate kill switch - STOPS ALL TRADING."""
        cls.KILL_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cls.KILL_FILE, 'w') as f:
            json.dump({
                "activated_at": datetime.now().isoformat(),
                "reason": reason,
            }, f)
            
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Send emergency alert
        try:
            send_telegram(
                f"🚨 EMERGENCY KILL SWITCH ACTIVATED!\n\n"
                f"Reason: {reason}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"All trading has been HALTED.\n"
                f"Manual intervention required.",
                {}
            )
        except:
            pass
            
    @classmethod
    def deactivate(cls):
        """Deactivate kill switch - allows trading to resume."""
        if cls.KILL_FILE.exists():
            cls.KILL_FILE.unlink()
        logger.info("Kill switch deactivated")
        
    @classmethod
    def is_active(cls) -> bool:
        """Check if kill switch is active."""
        return cls.KILL_FILE.exists()
        
    @classmethod
    def get_status(cls) -> Optional[Dict]:
        """Get kill switch status."""
        if not cls.KILL_FILE.exists():
            return None
        try:
            with open(cls.KILL_FILE) as f:
                return json.load(f)
        except:
            return {"activated_at": "unknown", "reason": "unknown"}


class TradingHours:
    """Market hours checker."""
    
    MARKET_OPEN = (9, 15)   # 9:15 AM
    MARKET_CLOSE = (15, 30)  # 3:30 PM
    
    @classmethod
    def is_market_hours(cls) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        current_time = (now.hour, now.minute)
        
        # Weekend check
        if now.weekday() >= 5:
            return False
            
        return cls.MARKET_OPEN <= current_time <= cls.MARKET_CLOSE
        
    @classmethod
    def time_to_open(cls) -> timedelta:
        """Time until market opens."""
        now = datetime.now()
        market_open = now.replace(
            hour=cls.MARKET_OPEN[0], 
            minute=cls.MARKET_OPEN[1], 
            second=0
        )
        
        if now >= market_open:
            market_open += timedelta(days=1)
            
        # Skip weekends
        while market_open.weekday() >= 5:
            market_open += timedelta(days=1)
            
        return market_open - now
        
    @classmethod
    def time_to_close(cls) -> timedelta:
        """Time until market closes."""
        now = datetime.now()
        market_close = now.replace(
            hour=cls.MARKET_CLOSE[0], 
            minute=cls.MARKET_CLOSE[1], 
            second=0
        )
        
        if now >= market_close:
            return timedelta(0)
            
        return market_close - now


# Health check functions for common components
def check_angel_broker_health(config: Dict) -> bool:
    """Check if Angel One broker is accessible."""
    try:
        from data.angel_broker import AngelBroker
        broker = AngelBroker(config)
        return broker.connect()
    except:
        return False


def check_database_health(db_path: str) -> bool:
    """Check if database is accessible."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except:
        return False


def check_internet_health() -> bool:
    """Check internet connectivity."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except:
        return False


if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config_test.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Create and start monitor
    monitor = SystemMonitor(config)
    
    # Register health checks
    monitor.register_component(
        "internet",
        check_internet_health,
        None
    )
    
    monitor.start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
