"""
Telegram alert utility for trade notifications.
Sends you a message whenever the algo places an order, hits a stop, or breaches limits.
"""

import requests
import logging
from datetime import datetime
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)


def send_telegram(message: str, config: Dict[str, Any]) -> bool:
    """
    Send a Telegram message using config.
    
    Args:
        message: The message to send
        config: Config dict with telegram.bot_token and telegram.chat_ids
        
    Returns:
        True if sent successfully, False otherwise
    """
    tg_config = config.get("telegram", {})
    bot_token = tg_config.get("bot_token")
    chat_ids = tg_config.get("chat_ids", [])
    
    if not bot_token or not chat_ids:
        logger.debug("Telegram not configured, skipping alert")
        return False
        
    if isinstance(chat_ids, (int, str)):
        chat_ids = [chat_ids]
        
    success = True
    base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    for chat_id in chat_ids:
        try:
            resp = requests.post(
                base_url,
                json={
                    "chat_id": str(chat_id),
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram send failed: {resp.text}")
                success = False
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            success = False
            
    return success


class TelegramAlert:
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, message: str) -> bool:
        """Send a message to your Telegram chat."""
        if not self.enabled:
            logger.info(f"[ALERT DISABLED] {message}")
            return False

        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            full_msg = f"🤖 *Algo Trader* | {timestamp}\n\n{message}"

            resp = requests.post(
                self.base_url,
                json={
                    "chat_id": self.chat_id,
                    "text": full_msg,
                    "parse_mode": "Markdown",
                },
                timeout=5,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
            return False

    def order_placed(self, symbol: str, side: str, qty: int, price: float, stop: float):
        self.send(
            f"📊 *ORDER PLACED*\n"
            f"Symbol: `{symbol}`\n"
            f"Side: {side}\n"
            f"Qty: {qty}\n"
            f"Price: ₹{price:,.2f}\n"
            f"Stop Loss: ₹{stop:,.2f}"
        )

    def stop_hit(self, symbol: str, loss: float):
        self.send(
            f"🛑 *STOP HIT*\n"
            f"Symbol: `{symbol}`\n"
            f"Loss: ₹{loss:,.2f}"
        )

    def target_hit(self, symbol: str, profit: float):
        self.send(
            f"✅ *TARGET HIT*\n"
            f"Symbol: `{symbol}`\n"
            f"Profit: ₹{profit:,.2f}"
        )

    def daily_limit_hit(self, total_loss: float):
        self.send(
            f"🚨 *DAILY LOSS LIMIT HIT*\n"
            f"Total loss today: ₹{total_loss:,.2f}\n"
            f"All trading stopped for today."
        )

    def daily_summary(self, trades: int, pnl: float, win_rate: float):
        emoji = "📈" if pnl >= 0 else "📉"
        self.send(
            f"{emoji} *DAILY SUMMARY*\n"
            f"Trades: {trades}\n"
            f"P&L: ₹{pnl:,.2f}\n"
            f"Win Rate: {win_rate:.0f}%"
        )
