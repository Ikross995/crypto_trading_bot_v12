#!/usr/bin/env python3
"""
🔄 Standalone Copy Trader Launcher
Запуск копи-трейдера отдельно от основного бота

Два режима работы:
1. Leader Mode - публикация сигналов в Telegram канал
2. Follower Mode - копирование сигналов из Telegram канала

Использование:
    python start_copy_trader.py --mode follower
    python start_copy_trader.py --mode leader --bot-port 8080
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
try:
    from loguru import logger
    import logging

    # Отключаем стандартный handler
    logger.remove()

    # Добавляем цветной вывод в консоль
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Добавляем файл для ошибок
    logger.add(
        "logs/copy_trader_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="00:00",
        retention="7 days"
    )

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def print_banner():
    """Красивый баннер запуска."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║         🔄 STANDALONE COPY TRADER v1.0                  ║
║                                                          ║
║    Автоматическое копирование сделок через Telegram     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Standalone Copy Trader - копирование сделок через Telegram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Follower mode - копирование сигналов из канала
  python start_copy_trader.py --mode follower --env .env.follower

  # Leader mode - публикация сигналов (требует запущенного основного бота)
  python start_copy_trader.py --mode leader --bot-port 8080

  # Follower mode с кастомными настройками
  python start_copy_trader.py --mode follower --testnet --verbose

Конфигурация через .env файл:
  TG_BOT_TOKEN=your_bot_token
  TG_SOURCE_CHANNEL=@trading_signals
  BINANCE_API_KEY=your_api_key
  BINANCE_API_SECRET=your_api_secret
  COPY_MODE=follower
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['follower', 'leader'],
        required=True,
        help='Режим работы: follower (копирование) или leader (публикация)'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='.env',
        help='Путь к .env файлу с конфигурацией (по умолчанию: .env)'
    )

    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Использовать Binance Testnet (для follower mode)'
    )

    parser.add_argument(
        '--bot-port',
        type=int,
        default=8080,
        help='Порт для подключения к основному боту (для leader mode)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод логов'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Тестовый режим без реального выполнения ордеров'
    )

    return parser.parse_args()


async def run_follower_mode(args):
    """
    Follower Mode - копирование сигналов из Telegram канала.

    Алгоритм работы:
    1. Подключение к Telegram через Bot API
    2. Прослушивание канала/группы с сигналами
    3. Парсинг сигналов (LONG/SHORT, ENTRY, TP, SL)
    4. Выполнение копирования через Binance API
    5. Риск-менеджмент и лимиты
    """
    logger.info("🎯 Запуск FOLLOWER MODE - копирование сигналов из Telegram")
    logger.info("=" * 60)

    # Импортируем необходимые модули
    try:
        from infra.copy_trader_follower import CopyTraderFollower
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        logger.info("Создаём standalone модуль follower...")
        # Будет создан в следующем шаге
        return

    # Загружаем конфигурацию из .env
    env_path = Path(args.env)
    if not env_path.exists():
        logger.error(f"❌ Файл конфигурации не найден: {env_path}")
        logger.info("💡 Создайте .env файл на основе .env.copy_trader.example")
        return

    load_dotenv(env_path)

    # Проверяем обязательные переменные
    required_vars = [
        'TG_BOT_TOKEN',
        'TG_SOURCE_CHANNEL',
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Отсутствуют обязательные переменные: {', '.join(missing_vars)}")
        return

    # Создаём и запускаем follower
    config = {
        'tg_bot_token': os.getenv('TG_BOT_TOKEN'),
        'tg_source_channel': os.getenv('TG_SOURCE_CHANNEL'),
        'binance_api_key': os.getenv('BINANCE_API_KEY'),
        'binance_api_secret': os.getenv('BINANCE_API_SECRET'),
        'testnet': args.testnet or os.getenv('TESTNET', 'false').lower() == 'true',
        'dry_run': args.dry_run or os.getenv('DRY_RUN', 'false').lower() == 'true',

        # Настройки копирования
        'position_size_multiplier': float(os.getenv('COPY_POSITION_SIZE_MULTIPLIER', '0.5')),
        'max_position_size': float(os.getenv('COPY_MAX_POSITION_SIZE', '100.0')),
        'min_position_size': float(os.getenv('COPY_MIN_POSITION_SIZE', '10.0')),
        'allowed_symbols': os.getenv('COPY_ALLOWED_SYMBOLS', '').split(',') if os.getenv('COPY_ALLOWED_SYMBOLS') else None,
        'max_leverage': int(os.getenv('COPY_MAX_LEVERAGE', '10')),
        'max_open_positions': int(os.getenv('COPY_MAX_OPEN_POSITIONS', '5')),
        'daily_loss_limit_pct': float(os.getenv('COPY_DAILY_LOSS_LIMIT_PCT', '5.0')),
    }

    logger.info("📋 Конфигурация:")
    logger.info(f"   Telegram канал: {config['tg_source_channel']}")
    logger.info(f"   Binance: {'TESTNET' if config['testnet'] else 'MAINNET'}")
    logger.info(f"   Режим: {'DRY RUN' if config['dry_run'] else 'LIVE'}")
    logger.info(f"   Размер позиции: {config['position_size_multiplier']*100}% от сигнала")
    logger.info(f"   Макс позиция: ${config['max_position_size']}")
    logger.info(f"   Макс открытых: {config['max_open_positions']}")
    logger.info(f"   Дневной лимит убытка: {config['daily_loss_limit_pct']}%")
    if config['allowed_symbols']:
        logger.info(f"   Разрешённые символы: {', '.join(config['allowed_symbols'])}")
    logger.info("=" * 60)

    try:
        follower = CopyTraderFollower(config)
        await follower.start()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Остановка по Ctrl+C...")
        await follower.stop()
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


async def run_leader_mode(args):
    """
    Leader Mode - публикация сигналов в Telegram канал.

    Этот режим подключается к запущенному основному боту и публикует
    его сигналы в Telegram канал для копирования follower-ами.

    Требуется запущенный основной бот с включённым API сервером.
    """
    logger.info("📢 Запуск LEADER MODE - публикация сигналов в Telegram")
    logger.info("=" * 60)

    # Загружаем конфигурацию
    env_path = Path(args.env)
    if not env_path.exists():
        logger.warning(f"⚠️  Файл {env_path} не найден, используем переменные окружения")
    else:
        load_dotenv(env_path)

    # Проверяем обязательные переменные
    tg_token = os.getenv('TG_BOT_TOKEN')
    tg_channel = os.getenv('TG_CHAT_ID')

    if not tg_token or not tg_channel:
        logger.error("❌ Не указаны TG_BOT_TOKEN или TG_CHAT_ID")
        return

    logger.info("📋 Конфигурация:")
    logger.info(f"   Telegram канал: {tg_channel}")
    logger.info(f"   Порт основного бота: {args.bot_port}")
    logger.info("=" * 60)

    # Импортируем Telegram бот
    try:
        from infra.telegram_bot import TelegramDashboardBot
        from infra.copy_trader_leader import CopyTraderLeader
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        return

    try:
        # Создаём Telegram бот для публикации
        telegram_bot = TelegramDashboardBot(tg_token, tg_channel)

        # Создаём leader
        leader = CopyTraderLeader(telegram_bot, bot_port=args.bot_port)

        logger.info("✅ Leader запущен и слушает сигналы от основного бота")
        logger.info(f"   Порт: {args.bot_port}")
        logger.info("   Нажмите Ctrl+C для остановки")

        await leader.start()

    except KeyboardInterrupt:
        logger.info("\n⚠️  Остановка по Ctrl+C...")
        await leader.stop()
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Главная функция."""
    print_banner()

    args = parse_args()

    # Устанавливаем уровень логирования
    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
            level="DEBUG"
        )

    # Запускаем соответствующий режим
    if args.mode == 'follower':
        await run_follower_mode(args)
    elif args.mode == 'leader':
        await run_leader_mode(args)
    else:
        logger.error(f"❌ Неизвестный режим: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Завершено")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
