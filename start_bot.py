#!/usr/bin/env python3
"""
Простой скрипт запуска бота с Phase 1-4 интеграцией
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 70)
    print("🤖 AI CRYPTO TRADING BOT - LAUNCH")
    print("=" * 70)
    print()

    # Выбор режима
    print("Выберите режим запуска:")
    print()
    print("1. 🔴 Live Trading (Testnet) - Безопасное тестирование")
    print("2. 🟢 Live Trading (Real) - РЕАЛЬНАЯ ТОРГОВЛЯ!")
    print("3. 📊 Backtest - Тестирование на исторических данных")
    print("4. 📄 Paper Trading - Виртуальная торговля")
    print()

    try:
        choice = input("Ваш выбор (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
        sys.exit(0)

    if choice == "1":
        print("\n🔴 Запуск Live Trading на Testnet...")
        print("=" * 70)
        run_live_testnet()
    elif choice == "2":
        print("\n⚠️  ВНИМАНИЕ: РЕАЛЬНАЯ ТОРГОВЛЯ!")
        confirm = input("Вы уверены? Введите 'YES' для подтверждения: ")
        if confirm == "YES":
            print("\n🟢 Запуск Live Trading на Real...")
            print("=" * 70)
            run_live_real()
        else:
            print("❌ Отменено")
    elif choice == "3":
        print("\n📊 Запуск Backtest...")
        print("=" * 70)
        run_backtest()
    elif choice == "4":
        print("\n📄 Запуск Paper Trading...")
        print("=" * 70)
        run_paper()
    else:
        print(f"❌ Неверный выбор: {choice}")
        sys.exit(1)

def run_live_testnet():
    """Запуск на testnet"""
    try:
        from core.config import Config
        from runner.live import run_live_trading

        # Загрузка конфигурации
        config = Config()
        config.dry_run = False  # Реальные ордера на testnet
        config.testnet = True   # Использовать testnet

        print(f"✅ Конфигурация загружена")
        print(f"   Режим: {'DRY RUN' if config.dry_run else 'LIVE'}")
        print(f"   Сеть: {'TESTNET' if config.testnet else 'MAINNET'}")
        print(f"   Символы: {config.symbols}")
        print(f"   Таймфрейм: {config.timeframe}")
        print(f"   Плечо: {config.leverage}x")
        print(f"   Риск на сделку: {config.risk_per_trade_pct}%")
        print()
        print("🚀 Запуск движка...")
        print("=" * 70)
        print()

        # Запуск
        asyncio.run(run_live_trading(config))

    except KeyboardInterrupt:
        print("\n\n⚠️  Бот остановлен пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()

def run_live_real():
    """Запуск на mainnet"""
    try:
        from core.config import Config
        from runner.live import run_live_trading

        config = Config()
        config.dry_run = False
        config.testnet = False  # MAINNET!

        print(f"⚠️  ВНИМАНИЕ: РЕАЛЬНАЯ ТОРГОВЛЯ НА MAINNET!")
        print(f"   Символы: {config.symbols}")
        print(f"   Плечо: {config.leverage}x")
        print()

        asyncio.run(run_live_trading(config))

    except KeyboardInterrupt:
        print("\n\n⚠️  Бот остановлен пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка запуска: {e}")

def run_backtest():
    """Запуск бэктеста"""
    try:
        from core.config import Config
        from runner.backtest import run_backtest as backtest

        config = Config()
        print("📊 Запуск бэктеста...")
        asyncio.run(backtest(config))

    except Exception as e:
        print(f"❌ Ошибка: {e}")

def run_paper():
    """Запуск paper trading"""
    try:
        from core.config import Config
        from runner.paper import run_paper_trading

        config = Config()
        config.dry_run = True
        print("📄 Запуск paper trading...")
        asyncio.run(run_paper_trading(config))

    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)
