#!/usr/bin/env python3
"""
Trading Bot Adapter - Integration Module
Адаптер для интеграции с существующим торговым ботом crypto_trading_bot_v12
"""

import asyncio
from typing import Dict, Any, Optional
from loguru import logger


class TradingBotAdapter:
    """
    Адаптер для подключения к существующему торговому боту.
    Перехватывает торговые сигналы от основного бота и передает их в CopyTradingEngine.
    """

    def __init__(self, existing_bot_instance):
        """
        Args:
            existing_bot_instance: Экземпляр существующего торгового бота
        """
        self.bot = existing_bot_instance
        self.signal_handlers = []
        self.running = False

    async def connect_to_existing_bot(self):
        """Подключение к существующему торговому боту."""
        # Перехват сигналов от существующего бота
        self.bot.register_signal_handler(self.on_trade_signal)

        logger.info("🔗 Connected to existing trading bot")

    async def on_trade_signal(self, signal: Dict[str, Any]):
        """
        Обработчик торговых сигналов от основного бота.

        Args:
            signal: Торговый сигнал от основного бота
        """
        try:
            # Нормализация формата сигнала
            normalized_signal = self.normalize_signal(signal)

            # Передача сигнала в систему копитрейдинга
            await self.copy_trading_engine.process_master_trade(normalized_signal)

            # Логирование
            await self.log_signal(normalized_signal)

        except Exception as e:
            logger.error(f"❌ Error processing trade signal: {e}")

    def normalize_signal(self, raw_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Нормализация сигнала в единый формат.

        Конвертирует формат сигнала от основного бота в формат,
        ожидаемый системой копитрейдинга.
        """
        return {
            'symbol': raw_signal.get('pair', '').replace('/', ''),
            'side': raw_signal.get('action', '').upper(),
            'quantity': float(raw_signal.get('amount', 0)),
            'price': float(raw_signal.get('price', 0)),
            'type': raw_signal.get('order_type', 'MARKET').upper(),
            'time': raw_signal.get('timestamp', 0),
            'orderId': raw_signal.get('order_id', '')
        }

    async def sync_positions(self):
        """Синхронизация состояния позиций."""
        await self.bot.sync_positions()
        await self.bot.sync_orders()

    async def log_signal(self, signal: Dict[str, Any]):
        """Логирование торгового сигнала."""
        await self.bot.log_signal(signal)


# Пример использования
async def main():
    """Пример использования адаптера."""

    # Заглушка для существующего бота
    class MockTradingBot:
        def __init__(self):
            self.signal_handler = None

        def register_signal_handler(self, handler):
            self.signal_handler = handler
            print("✅ Signal handler registered")

        async def sync_positions(self):
            print("🔄 Positions synced")

        async def sync_orders(self):
            print("🔄 Orders synced")

        async def log_signal(self, signal):
            print(f"📝 Signal logged: {signal}")

        async def emit_signal(self, signal):
            """Симуляция эмиссии торгового сигнала."""
            if self.signal_handler:
                await self.signal_handler(signal)

    # Создаем мок существующего бота
    existing_bot = MockTradingBot()

    # Создаем адаптер
    adapter = TradingBotAdapter(existing_bot)
    await adapter.connect_to_existing_bot()

    # Симулируем торговый сигнал от бота
    test_signal = {
        'pair': 'BTC/USDT',
        'action': 'buy',
        'amount': 0.5,
        'price': 50000.0,
        'order_type': 'market',
        'timestamp': 1234567890,
        'order_id': 'test_order_123'
    }

    await existing_bot.emit_signal(test_signal)


if __name__ == "__main__":
    asyncio.run(main())
