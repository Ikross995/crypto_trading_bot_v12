#!/usr/bin/env python3
"""
Quick Telegram Bot Test Script
Тестирует подключение к Telegram без запуска всего торгового бота
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_telegram():
    """Тест Telegram бота."""
    try:
        # Import after path is set
        from infra.telegram_bot import TelegramDashboardBot
        from dotenv import load_dotenv

        print("=" * 60)
        print("🤖 TELEGRAM BOT TEST")
        print("=" * 60)
        print()

        # Load .env
        load_dotenv()

        # Get settings
        bot_token = os.getenv("TG_BOT_TOKEN", "")
        chat_id = os.getenv("TG_CHAT_ID", "")

        if not bot_token or not chat_id:
            print("❌ Error: Missing Telegram configuration")
            print()
            print("Please set in .env file:")
            print("  TG_BOT_TOKEN=your_token_from_botfather")
            print("  TG_CHAT_ID=your_chat_id")
            print()
            print("See TELEGRAM_SETUP.md for step-by-step instructions")
            return False

        print(f"📋 Configuration:")
        print(f"  Token: {bot_token[:10]}...{bot_token[-5:]}")
        print(f"  Chat ID: {chat_id}")
        print()

        # Create bot
        print("🔌 Connecting to Telegram...")
        bot = TelegramDashboardBot(bot_token, chat_id)

        # Test connection
        if not await bot.test_connection():
            print("❌ Failed to connect")
            print()
            print("Check your TG_BOT_TOKEN in .env file")
            return False

        print()
        print("✅ Bot connected successfully!")
        print()

        # Send test message
        print("📤 Sending test message...")
        success = await bot.send_message(
            "🤖 <b>Trading Bot Test</b>\n\n"
            "✅ Connection successful!\n"
            "Dashboard updates will be sent here.\n\n"
            "<i>Test completed at " + asyncio.get_event_loop().time().__str__() + "</i>",
            parse_mode="HTML"
        )

        if success:
            print("✅ Test message sent!")
            print()
            print("=" * 60)
            print("🎉 SUCCESS! Telegram bot is ready!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("  1. Check your Telegram group for the test message")
            print("  2. Set TG_DASHBOARD_ENABLED=true in .env")
            print("  3. Run your trading bot normally")
            print()
            return True
        else:
            print("❌ Failed to send message")
            print()
            print("Possible issues:")
            print("  - Bot not added to the group")
            print("  - Bot doesn't have send message permissions")
            print("  - Wrong Chat ID")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print()
        print("Required packages:")
        print("  pip install aiohttp python-dotenv loguru")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_telegram())
    sys.exit(0 if result else 1)
