#!/usr/bin/env python3
"""
Copy Trading System - Core Module
Система копитрейдинга для автоматического копирования сделок мастер-трейдеров
"""

import asyncio
from asyncio import Queue, create_task
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
from loguru import logger


@dataclass
class TradeSignal:
    """Торговый сигнал от мастер-трейдера."""
    symbol: str
    side: str  # BUY/SELL
    quantity: Decimal
    price: Decimal
    order_type: str  # MARKET/LIMIT
    timestamp: float
    master_order_id: str


@dataclass
class CopierAccount:
    """Аккаунт копировщика с настройками."""
    user_id: int
    email: str
    api_key: str
    api_secret: str

    # Настройки копирования
    is_active: bool = True
    allowed_pairs: List[str] = None  # None = все пары

    # Режимы расчета позиции
    position_mode: str = 'fixed_ratio'  # fixed_ratio, fixed_amount, percentage
    copy_ratio: Decimal = Decimal('0.1')  # Для fixed_ratio
    fixed_amount: Decimal = Decimal('100')  # Для fixed_amount в USDT
    balance_percentage: Decimal = Decimal('0.05')  # Для percentage режима

    # Риск-менеджмент
    max_position_size: Decimal = Decimal('1000')  # USDT
    max_open_positions: int = 10
    max_daily_loss: Decimal = Decimal('-500')  # USDT

    # Статистика
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: Decimal = Decimal('0')


class PositionTracker:
    """Отслеживание открытых позиций."""

    def __init__(self):
        self.positions: Dict[str, Dict[str, Any]] = {}

    def update_position(self, user_id: int, symbol: str, result: Any):
        """Обновить позицию после открытия ордера."""
        key = f"{user_id}_{symbol}"
        self.positions[key] = {
            'user_id': user_id,
            'symbol': symbol,
            'result': result,
            'opened_at': datetime.now()
        }

    def close_position(self, user_id: int, symbol: str):
        """Закрыть позицию."""
        key = f"{user_id}_{symbol}"
        if key in self.positions:
            del self.positions[key]

    def get_open_positions(self, user_id: int) -> List[str]:
        """Получить список открытых позиций для пользователя."""
        return [
            pos['symbol']
            for key, pos in self.positions.items()
            if pos['user_id'] == user_id
        ]

    def get_position_count(self, user_id: int) -> int:
        """Количество открытых позиций."""
        return len(self.get_open_positions(user_id))


class CopyTradingEngine:
    """
    Основной движок копитрейдинга.
    Обрабатывает сигналы от мастер-трейдера и распределяет их копировщикам.
    """

    def __init__(self):
        self.signal_queue = Queue()
        self.active_copiers: Dict[int, CopierAccount] = {}
        self.position_tracker = PositionTracker()
        self.running = False

    def add_copier(self, copier: CopierAccount):
        """Добавить копировщика в систему."""
        self.active_copiers[copier.user_id] = copier
        logger.info(f"📋 Copier added: {copier.email} (user_id={copier.user_id})")

    def remove_copier(self, user_id: int):
        """Удалить копировщика из системы."""
        if user_id in self.active_copiers:
            email = self.active_copiers[user_id].email
            del self.active_copiers[user_id]
            logger.info(f"📋 Copier removed: {email} (user_id={user_id})")

    async def process_master_trade(self, trade: Dict[str, Any]):
        """
        Обработка сделки от мастер-трейдера.
        Конвертация в TradeSignal и добавление в очередь.
        """
        signal = TradeSignal(
            symbol=trade['symbol'],
            side=trade['side'],
            quantity=Decimal(str(trade['quantity'])),
            price=Decimal(str(trade['price'])),
            order_type=trade['type'],
            timestamp=trade['time'],
            master_order_id=trade['orderId']
        )

        logger.info(f"📡 Master trade received: {signal.side} {signal.symbol}")

        # Добавляем сигнал в очередь
        await self.signal_queue.put(signal)

    async def distribute_signal(self, signal: TradeSignal):
        """
        Распределение сигнала между копировщиками.
        Создает задачи для каждого активного копировщика.
        """
        tasks = []

        for user_id, copier in self.active_copiers.items():
            if copier.is_active and self._is_copier_allowed(copier, signal):
                task = create_task(
                    self.execute_copy_trade(copier, signal)
                )
                tasks.append(task)

        # Параллельное выполнение для всех копировщиков
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Логирование результатов
        for user_id, result in zip(self.active_copiers.keys(), results):
            if isinstance(result, Exception):
                await self.handle_copy_error(user_id, signal, result)
            else:
                await self.log_successful_copy(user_id, signal, result)

    def _is_copier_allowed(self, copier: CopierAccount, signal: TradeSignal) -> bool:
        """Проверка, разрешено ли копировщику копировать этот сигнал."""
        # Проверка списка разрешенных пар
        if copier.allowed_pairs and signal.symbol not in copier.allowed_pairs:
            return False

        return True

    async def execute_copy_trade(self, copier: CopierAccount, signal: TradeSignal):
        """Выполнение копирования сделки для конкретного копировщика."""
        # Расчет размера позиции с учетом настроек копировщика
        adjusted_quantity = await self.calculate_position_size(
            copier, signal
        )

        # Проверка рисков
        if not await self.check_risk_limits(copier, signal, adjusted_quantity):
            raise RiskLimitExceeded(f"Risk limit exceeded for user {copier.user_id}")

        # Создание ордера
        order_params = {
            'symbol': signal.symbol,
            'side': signal.side,
            'type': signal.order_type,
            'quantity': float(adjusted_quantity)
        }

        if signal.order_type == 'LIMIT':
            # Добавляем проскальзывание для лимитных ордеров
            slippage = copier.settings.get('slippage', 0.001)
            if signal.side == 'BUY':
                order_params['price'] = float(signal.price * (1 + slippage))
            else:
                order_params['price'] = float(signal.price * (1 - slippage))

        # Выполнение ордера через Binance API
        # (Здесь нужна интеграция с binance клиентом копировщика)
        # result = await copier.binance_client.create_order(**order_params)

        # Временная заглушка
        result = {
            'orderId': f'order_{copier.user_id}_{signal.master_order_id}',
            'status': 'FILLED',
            **order_params
        }

        # Обновление позиции
        await self.position_tracker.update_position(
            copier.user_id, signal.symbol, result
        )

        return result

    async def calculate_position_size(
        self, copier: CopierAccount, signal: TradeSignal
    ) -> Decimal:
        """
        Расчет размера позиции с учетом настроек копировщика.

        Поддерживаемые режимы:
        - fixed_ratio: Фиксированное соотношение к мастеру
        - fixed_amount: Фиксированная сумма в USDT
        - percentage: Процент от баланса
        """
        # Получаем баланс копировщика
        balance = await self.get_copier_balance(copier)

        if copier.position_mode == 'fixed_ratio':
            # Фиксированное соотношение к мастеру
            ratio = Decimal(str(copier.settings.get('copy_ratio', 0.1)))
            return signal.quantity * ratio

        elif copier.position_mode == 'fixed_amount':
            # Фиксированная сумма в USDT
            fixed_amount = Decimal(str(copier.settings.get('fixed_amount', 100)))
            current_price = await self.get_current_price(signal.symbol)
            return fixed_amount / current_price

        elif copier.position_mode == 'percentage':
            # Процент от баланса
            percentage = Decimal(str(copier.settings.get('balance_percentage', 0.05)))
            amount = balance * percentage
            current_price = await self.get_current_price(signal.symbol)
            return amount / current_price

        else:
            raise ValueError(f"Unknown position mode: {copier.position_mode}")

    async def check_risk_limits(
        self, copier: CopierAccount, signal: TradeSignal, quantity: Decimal
    ) -> bool:
        """Проверка лимитов риска."""
        # Максимальный размер позиции
        position_value = quantity * await self.get_current_price(signal.symbol)
        max_position = Decimal(str(copier.settings.get('max_position_size', 1000)))
        if position_value > max_position:
            logger.warning(f"Position size {position_value} exceeds max {max_position}")
            return False

        # Максимальное количество открытых позиций
        open_positions = self.position_tracker.get_position_count(copier.user_id)
        max_open = copier.settings.get('max_open_positions', 10)
        if open_positions >= max_open:
            logger.warning(f"Max open positions reached: {open_positions}/{max_open}")
            return False

        # Дневной лимит убытков
        daily_loss = await self.position_tracker.get_daily_pnl(copier.user_id)
        max_daily_loss = Decimal(str(copier.settings.get('max_daily_loss', -500)))
        if daily_loss < max_daily_loss:
            logger.warning(f"Daily loss limit reached: {daily_loss} < {max_daily_loss}")
            return False

        return True

    async def get_copier_balance(self, copier: CopierAccount) -> Decimal:
        """Получить баланс копировщика."""
        # Здесь должна быть интеграция с Binance API копировщика
        # balance = await copier.binance_client.get_available_balance()

        # Временная заглушка
        return Decimal('1000.0')

    async def get_current_price(self, symbol: str) -> Decimal:
        """Получить текущую цену символа."""
        # Здесь должна быть интеграция с Binance API
        # price = await binance_client.get_current_price(symbol)

        # Временная заглушка
        return Decimal('50000.0')

    async def handle_copy_error(self, user_id: int, signal: TradeSignal, error: Exception):
        """Обработка ошибки копирования."""
        logger.error(f"❌ Copy error for user {user_id}: {error}")

        # Уведомление пользователя через Telegram
        # await self.send_telegram_notification(user_id, f"Failed to copy trade: {error}")

    async def log_successful_copy(self, user_id: int, signal: TradeSignal, result: Any):
        """Логирование успешного копирования."""
        logger.info(f"✅ Trade copied for user {user_id}: {result}")

        # Обновление статистики копировщика
        if user_id in self.active_copiers:
            copier = self.active_copiers[user_id]
            copier.total_trades += 1

    async def start_polling(self):
        """Запуск обработки очереди сигналов."""
        self.running = True
        logger.info("🚀 Copy trading engine started")

        while self.running:
            try:
                # Получаем сигнал из очереди
                signal = await self.signal_queue.get()

                # Распределяем сигнал между копировщиками
                await self.distribute_signal(signal)

            except Exception as e:
                logger.error(f"❌ Error in copy trading loop: {e}")

            await asyncio.sleep(1)  # Пауза между обработкой сигналов

    async def stop_polling(self):
        """Остановка обработки очереди."""
        self.running = False
        logger.info("🛑 Copy trading engine stopped")


class RiskLimitExceeded(Exception):
    """Исключение при превышении лимитов риска."""
    pass


# Пример использования
async def main():
    """Пример использования системы копитрейдинга."""

    # Создаем движок
    engine = CopyTradingEngine()

    # Добавляем копировщика
    copier = CopierAccount(
        user_id=1,
        email="copier1@example.com",
        api_key="api_key_here",
        api_secret="api_secret_here",
        is_active=True,
        position_mode='fixed_ratio',
        copy_ratio=Decimal('0.1')
    )
    copier.settings = {
        'copy_ratio': 0.1,
        'max_position_size': 1000,
        'max_open_positions': 10,
        'max_daily_loss': -500
    }
    engine.add_copier(copier)

    # Запускаем движок
    polling_task = create_task(engine.start_polling())

    # Симулируем сделку мастер-трейдера
    master_trade = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 1.0,
        'price': 50000.0,
        'type': 'MARKET',
        'time': datetime.now().timestamp(),
        'orderId': 'master_order_123'
    }

    await engine.process_master_trade(master_trade)

    # Ждем обработки
    await asyncio.sleep(5)

    # Останавливаем движок
    await engine.stop_polling()


if __name__ == "__main__":
    asyncio.run(main())
