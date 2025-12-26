#!/usr/bin/env python3
"""
Minimal Telegram test - bypasses infra package imports
"""
import asyncio
import os
import sys
from pathlib import Path

# Don't import infra package, just the specific module
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    try:
        # Direct import without going through infra/__init__.py
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "telegram_bot",
            Path(__file__).parent / "infra" / "telegram_bot.py"
        )
        telegram_bot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(telegram_bot)

        TelegramDashboardBot = telegram_bot.TelegramDashboardBot

        from dotenv import load_dotenv
        load_dotenv()

        token = os.getenv("TG_BOT_TOKEN", "")
        chat_id = os.getenv("TG_CHAT_ID", "")

        if not token or not chat_id:
            print("❌ Set TG_BOT_TOKEN and TG_CHAT_ID in .env file")
            return

        print("🤖 Testing Telegram bot...")
        print(f"Token: {token[:10]}...{token[-5:]}")
        print(f"Chat ID: {chat_id}\n")

        bot = TelegramDashboardBot(token, chat_id)

        if await bot.test_connection():
            print("\n✅ Bot connected!")

            success = await bot.send_message(
                "🤖 <b>Test Message</b>\n\nTelegram bot is working!",
                parse_mode="HTML"
            )

            if success:
                print("✅ Message sent successfully!")
                print("📱 Check your Telegram group")
            else:
                print("❌ Failed to send message")
        else:
            print("❌ Connection failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
