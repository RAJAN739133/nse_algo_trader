"""
Telegram Bot Setup + Test — Run this to connect your phone to the algo.

BEFORE running this script, do these 3 things on your phone:
1. Open Telegram app
2. Search for @BotFather → tap Start → type /newbot
3. Follow the prompts (name: Rajan Stock Bot, username: rajan_stock_XXXX_bot)
4. Copy the token BotFather gives you
5. Search for @userinfobot → tap Start → copy your chat ID number

Then run: python setup_telegram.py
"""
import sys
import requests
import yaml
from pathlib import Path


def test_telegram(token, chat_id):
    """Send a test message to verify connection."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    message = (
        "🤖 *Rajan Stock Bot Connected!*\n\n"
        "Your algo trading bot is now linked to this chat.\n"
        "You will receive:\n"
        "• Trade alerts (BUY/SELL)\n"
        "• Daily P&L summary\n"
        "• Risk warnings\n"
        "• Kill switch commands\n\n"
        "Commands you can send:\n"
        "/status — Current positions\n"
        "/pnl — Today's P&L\n"
        "/stop — Emergency stop\n"
        "/start — Resume trading\n"
    )
    resp = requests.post(url, json={
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    })
    return resp.json()


def setup():
    print("""
    ╔══════════════════════════════════════════════╗
    ║     TELEGRAM BOT SETUP — Rajan Stock Bot     ║
    ╚══════════════════════════════════════════════╝

    Before running this, you need to create the bot on your phone:

    Step 1: Open Telegram on your phone
    Step 2: Search for @BotFather
    Step 3: Tap Start, then type: /newbot
    Step 4: Name: Rajan Stock Bot
    Step 5: Username: rajan_stock_0739_bot (must end with _bot)
    Step 6: BotFather gives you a TOKEN (long string starting with numbers)

    Step 7: Search for @userinfobot in Telegram
    Step 8: Tap Start — it shows your CHAT ID (a number)

    Step 9: IMPORTANT! Search for YOUR bot (@rajan_stock_0739_bot)
            and tap START. This connects your phone to the bot.
    """)

    token = input("  Paste your bot TOKEN here: ").strip()
    if not token or ":" not in token:
        print("  Invalid token! It should look like: 7123456789:AAH3kL...")
        return

    chat_id = input("  Paste your CHAT ID here: ").strip()
    if not chat_id or not chat_id.lstrip("-").isdigit():
        print("  Invalid chat ID! It should be a number like: 123456789")
        return

    print(f"\n  Testing connection...")
    result = test_telegram(token, chat_id)

    if result.get("ok"):
        print(f"  ✅ SUCCESS! Check your Telegram — you should see the welcome message!")

        # Save to config files
        for cfg_name in ["config/config_test.yaml", "config/config_prod.yaml"]:
            cfg_path = Path(cfg_name)
            if cfg_path.exists():
                with open(cfg_path) as f:
                    config = yaml.safe_load(f) or {}
                config.setdefault("alerts", {})
                config["alerts"]["telegram_enabled"] = True
                config["alerts"]["telegram_bot_token"] = token
                config["alerts"]["telegram_chat_id"] = chat_id
                with open(cfg_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"  ✅ Saved to {cfg_name}")

        print(f"\n  Done! Your bot is ready. Run the algo and you'll get alerts on Telegram.")
        print(f"  Test: python paper_trader.py simulate")
    else:
        error = result.get("description", "Unknown error")
        print(f"  ❌ FAILED: {error}")
        if "chat not found" in error.lower():
            print("  → You need to open YOUR bot in Telegram and tap START first!")
            print("  → Search for @rajan_stock_0739_bot and tap Start, then run this again.")
        elif "unauthorized" in error.lower():
            print("  → Token is wrong. Go to @BotFather and copy the token again.")


def send_alert(message):
    """Quick function to send an alert from anywhere in the project."""
    cfg_path = Path("config/config_test.yaml")
    if not cfg_path.exists():
        return False
    with open(cfg_path) as f:
        config = yaml.safe_load(f) or {}
    alerts = config.get("alerts", {})
    if not alerts.get("telegram_enabled"):
        return False
    token = alerts.get("telegram_bot_token", "")
    chat_id = alerts.get("telegram_chat_id", "")
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        )
        return resp.json().get("ok", False)
    except:
        return False


if __name__ == "__main__":
    setup()
