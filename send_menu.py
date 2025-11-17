#!/usr/bin/env python3
"""
Быстрая отправка меню в Telegram бота
Используйте этот скрипт, чтобы сразу увидеть кнопки в боте
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from infra.telegram_bot import TelegramDashboardBot
from loguru import logger


async def main():
    """Отправить главное меню в бота."""
    # Загружаем .env
    env_file = Path('.env')
    if env_file.exists():
        load_dotenv()
    else:
        print("❌ Файл .env не найден!")
        print("\nСоздайте файл .env с настройками:")
        print("TG_BOT_TOKEN=ваш_токен_бота")
        print("TG_CHAT_ID=ваш_chat_id")
        print("TG_WEBAPP_URL=ваш_ngrok_url (опционально)")
        return

    # Читаем настройки
    bot_token = os.getenv("TG_BOT_TOKEN", "")
    chat_id = os.getenv("TG_CHAT_ID", "")
    webapp_url = os.getenv("TG_WEBAPP_URL", "")

    if not bot_token or not chat_id:
        print("❌ TG_BOT_TOKEN и TG_CHAT_ID должны быть установлены в .env файле!")
        return

    print("""
╔═══════════════════════════════════════════════════════════╗
║       📱 Отправка меню в Telegram бота                   ║
╚═══════════════════════════════════════════════════════════╝
    """)

    print(f"🤖 Bot Token: {bot_token[:10]}...{bot_token[-5:]}")
    print(f"💬 Chat ID: {chat_id}")
    if webapp_url:
        print(f"🌐 Web App URL: {webapp_url}")
    else:
        print(f"⚠️  Web App URL не установлен (добавьте TG_WEBAPP_URL в .env)")

    print()

    # Создаем бота
    bot = TelegramDashboardBot(bot_token, chat_id, webapp_url=webapp_url)

    # Тестируем соединение
    print("🔄 Проверка подключения к боту...")
    if not await bot.test_connection():
        print("❌ Не удалось подключиться к боту!")
        print("Проверьте TG_BOT_TOKEN в .env файле")
        return

    print("✅ Подключение успешно!")
    print()

    # Отправляем приветственное сообщение
    print("📤 Отправка приветственного сообщения...")
    welcome_msg = """
🤖 <b>Trading Bot Menu Initialized!</b>

Бот готов к использованию!

<b>Доступные команды:</b>
/menu - Показать главное меню с кнопками
/status - Текущий статус бота
/positions - Открытые позиции
/stats - Статистика торговли
/help - Помощь

<b>💡 Совет:</b> Используйте кнопки ниже для навигации!
    """

    success = await bot.send_message(welcome_msg)

    if not success:
        print("❌ Не удалось отправить сообщение!")
        print("Проверьте, что бот добавлен в группу/канал")
        print("и имеет права на отправку сообщений")
        return

    print("✅ Приветственное сообщение отправлено!")
    print()

    # Отправляем главное меню с кнопками
    print("📤 Отправка главного меню с кнопками...")
    success = await bot.send_main_menu(webapp_url=webapp_url)

    if success:
        print("✅ Меню с кнопками отправлено!")
        print()
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║  🎉 ГОТОВО! Проверьте Telegram - кнопки должны быть!     ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()
        print("📱 Откройте ваш Telegram чат/группу")
        print("👆 Нажмите на кнопки для навигации")
        print()
        if webapp_url:
            print("🌐 Кнопка 'Интерактивный Dashboard' откроет Web App")
        else:
            print("💡 Добавьте TG_WEBAPP_URL в .env для интерактивного дашборда")
            print("   Например: TG_WEBAPP_URL=https://your-ngrok-url.ngrok-free.app")
    else:
        print("❌ Не удалось отправить меню!")
        print("Проверьте логи выше для деталей")


if __name__ == "__main__":
    asyncio.run(main())
