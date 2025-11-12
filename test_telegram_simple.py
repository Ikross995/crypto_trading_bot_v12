#!/usr/bin/env python3
"""
Super Simple Telegram Test - No local imports
Just pure requests to Telegram API
"""
import asyncio
import os

try:
    import aiohttp
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("\nInstall:")
    print("  pip install aiohttp python-dotenv")
    exit(1)

async def test_bot():
    # Load .env
    load_dotenv()

    token = os.getenv("TG_BOT_TOKEN", "")
    chat_id = os.getenv("TG_CHAT_ID", "")

    if not token:
        print("❌ TG_BOT_TOKEN not set in .env")
        return False

    if not chat_id:
        print("❌ TG_CHAT_ID not set in .env")
        return False

    print("=" * 60)
    print("🤖 TELEGRAM API TEST (Direct)")
    print("=" * 60)
    print()
    print(f"Token: {token[:10]}...{token[-5:]}")
    print(f"Chat ID: {chat_id}")
    print()

    # Test getMe
    print("🔌 Testing connection...")
    async with aiohttp.ClientSession() as session:
        url = f"https://api.telegram.org/bot{token}/getMe"
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"❌ Connection failed: {resp.status}")
                text = await resp.text()
                print(text)
                return False

            data = await resp.json()
            if not data.get('ok'):
                print("❌ Bot authentication failed")
                return False

            bot_info = data.get('result', {})
            username = bot_info.get('username', 'Unknown')
            print(f"✅ Connected: @{username}")

    # Send test message
    print()
    print("📤 Sending test message...")

    async with aiohttp.ClientSession() as session:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": (
                "🤖 <b>Telegram Test</b>\n\n"
                "✅ Your bot is working!\n"
                "Connection successful."
            ),
            "parse_mode": "HTML"
        }

        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                print(f"❌ Send failed: {resp.status}")
                text = await resp.text()
                print(text)
                print()
                print("Common issues:")
                print("  - Bot not added to group")
                print("  - Wrong Chat ID")
                print("  - Bot doesn't have send permissions")
                return False

            data = await resp.json()
            if not data.get('ok'):
                print("❌ Message not sent")
                print(data)
                return False

    print("✅ Message sent!")
    print()
    print("=" * 60)
    print("🎉 SUCCESS!")
    print("=" * 60)
    print()
    print("Check your Telegram group for the test message")
    print()
    print("Next steps:")
    print("  1. Add these to your .env:")
    print("     TG_DASHBOARD_ENABLED=true")
    print("     TG_DASHBOARD_INTERVAL=300")
    print()
    print("  2. Run your trading bot normally")
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_bot())
        exit(0 if result else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
