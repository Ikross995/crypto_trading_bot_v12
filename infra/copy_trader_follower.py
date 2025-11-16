#!/usr/bin/env python3
"""
Copy Trader Follower - Standalone копирование сигналов из Telegram

Функциональность:
- Подключение к Telegram каналу/группе
- Парсинг сигналов в разных форматах
- Копирование сделок через Binance Futures API
- Риск-менеджмент и защита капитала
- Отслеживание позиций и PnL
"""

import asyncio
import aiohttp
import re
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    from binance.client import AsyncClient
    from binance.enums import *
except ImportError:
    logger.warning("⚠️  python-binance не установлен. Установите: pip install python-binance")
    AsyncClient = None


@dataclass
class TradingSignal:
    """Торговый сигнал, распарсенный из Telegram сообщения."""
    symbol: str
    action: str  # OPEN или CLOSE
    side: str  # LONG или SHORT
    entry_price: Optional[float] = None
    take_profits: List[float] = None
    stop_loss: Optional[float] = None
    leverage: Optional[int] = None
    position_size_usdt: Optional[float] = None
    timestamp: datetime = None
    raw_message: str = ""

    def __post_init__(self):
        if self.take_profits is None:
            self.take_profits = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SignalParser:
    """Парсер торговых сигналов из текста Telegram."""

    @staticmethod
    def parse_signal(text: str) -> Optional[TradingSignal]:
        """
        Парсинг сигнала из текста.

        Поддерживаемые форматы:
        1. Стандартный формат:
           🟢 LONG BTCUSDT
           Entry: $41,250.00
           TP1: $42,780.00 (+3.71%)
           SL: $40,435.00 (-1.98%)

        2. Компактный формат:
           🟢 LONG BTCUSDT
           Entry: 41250 | TP: 42780 | SL: 40435

        3. Формат закрытия:
           ❌ CLOSE BTCUSDT
           Profit: +5.2%
        """
        try:
            # Определяем действие и направление
            action = "OPEN"
            if "CLOSE" in text or "EXIT" in text or "CLOSED" in text:
                action = "CLOSE"

            # Определяем сторону
            side = None
            if "LONG" in text or "🟢" in text or "BUY" in text:
                side = "LONG"
            elif "SHORT" in text or "🔴" in text or "SELL" in text:
                side = "SHORT"

            # Извлекаем символ
            symbol_match = re.search(r'([A-Z]{3,}USDT)', text)
            if not symbol_match:
                return None
            symbol = symbol_match.group(1)

            # Если это закрытие, не парсим дальше
            if action == "CLOSE":
                return TradingSignal(
                    symbol=symbol,
                    action=action,
                    side=side or "UNKNOWN",
                    raw_message=text
                )

            # Парсим entry price
            entry = None
            entry_patterns = [
                r'Entry[:\s]+\$?([0-9,]+\.?[0-9]*)',
                r'Entry[:\s]+([0-9,]+)',
                r'Price[:\s]+\$?([0-9,]+\.?[0-9]*)'
            ]
            for pattern in entry_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entry = float(match.group(1).replace(',', ''))
                    break

            # Парсим take profits
            take_profits = []
            tp_patterns = [
                r'TP\d*[:\s]+\$?([0-9,]+\.?[0-9]*)',
                r'Take\s*Profit\d*[:\s]+\$?([0-9,]+\.?[0-9]*)',
                r'Target\d*[:\s]+\$?([0-9,]+\.?[0-9]*)'
            ]
            for pattern in tp_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    tp = float(match.group(1).replace(',', ''))
                    if tp not in take_profits:
                        take_profits.append(tp)

            # Парсим stop loss
            stop_loss = None
            sl_patterns = [
                r'SL[:\s]+\$?([0-9,]+\.?[0-9]*)',
                r'Stop\s*Loss[:\s]+\$?([0-9,]+\.?[0-9]*)',
                r'Stop[:\s]+\$?([0-9,]+\.?[0-9]*)'
            ]
            for pattern in sl_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    stop_loss = float(match.group(1).replace(',', ''))
                    break

            # Парсим leverage
            leverage = None
            lev_match = re.search(r'Leverage[:\s]+(\d+)x?|(\d+)x', text, re.IGNORECASE)
            if lev_match:
                leverage = int(lev_match.group(1) or lev_match.group(2))

            # Парсим размер позиции
            position_size = None
            size_match = re.search(r'Position[:\s]+\$?([0-9,]+\.?[0-9]*)', text, re.IGNORECASE)
            if size_match:
                position_size = float(size_match.group(1).replace(',', ''))

            # Создаём сигнал только если есть основная информация
            if not side or not entry:
                logger.debug(f"Недостаточно информации для сигнала: side={side}, entry={entry}")
                return None

            return TradingSignal(
                symbol=symbol,
                action=action,
                side=side,
                entry_price=entry,
                take_profits=sorted(take_profits),
                stop_loss=stop_loss,
                leverage=leverage,
                position_size_usdt=position_size,
                raw_message=text
            )

        except Exception as e:
            logger.error(f"Ошибка парсинга сигнала: {e}")
            logger.debug(f"Текст: {text}")
            return None


class RiskManager:
    """Управление рисками для копи-трейдинга."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.daily_pnl = Decimal('0')
        self.daily_pnl_reset_time = datetime.now()
        self.open_positions: Dict[str, Dict] = {}

    async def check_can_open_position(self, signal: TradingSignal, balance: float) -> tuple[bool, str]:
        """
        Проверка возможности открытия позиции.

        Returns:
            (can_open, reason)
        """
        # Проверяем дневной лимит убытков
        await self._reset_daily_pnl_if_needed()

        daily_limit_pct = self.config.get('daily_loss_limit_pct', 5.0)
        daily_limit_usdt = balance * (daily_limit_pct / 100)

        if self.daily_pnl < -daily_limit_usdt:
            return False, f"Достигнут дневной лимит убытков: {self.daily_pnl:.2f} USDT (лимит: -{daily_limit_usdt:.2f})"

        # Проверяем количество открытых позиций
        max_positions = self.config.get('max_open_positions', 5)
        if len(self.open_positions) >= max_positions:
            return False, f"Достигнут лимит открытых позиций: {len(self.open_positions)}/{max_positions}"

        # Проверяем разрешённые символы
        allowed_symbols = self.config.get('allowed_symbols')
        if allowed_symbols and signal.symbol not in allowed_symbols:
            return False, f"Символ {signal.symbol} не в списке разрешённых"

        # Проверяем leverage
        max_leverage = self.config.get('max_leverage', 10)
        if signal.leverage and signal.leverage > max_leverage:
            return False, f"Leverage {signal.leverage}x превышает максимальный {max_leverage}x"

        return True, "OK"

    def calculate_position_size(self, signal: TradingSignal, balance: float) -> float:
        """
        Расчёт размера позиции с учётом настроек риск-менеджмента.

        Returns:
            Размер позиции в USDT
        """
        # Базовый размер из сигнала или настроек
        if signal.position_size_usdt:
            base_size = signal.position_size_usdt
        else:
            # Если размер не указан, используем фиксированный процент от баланса
            base_size = balance * 0.02  # 2% от баланса по умолчанию

        # Применяем multiplier из настроек
        multiplier = self.config.get('position_size_multiplier', 0.5)
        adjusted_size = base_size * multiplier

        # Применяем ограничения
        max_size = self.config.get('max_position_size', 100.0)
        min_size = self.config.get('min_position_size', 10.0)

        adjusted_size = max(min_size, min(adjusted_size, max_size))

        return adjusted_size

    async def _reset_daily_pnl_if_needed(self):
        """Сброс дневного PnL в полночь UTC."""
        now = datetime.now()
        if now.date() > self.daily_pnl_reset_time.date():
            logger.info(f"📊 Сброс дневного PnL: {self.daily_pnl:.2f} USDT")
            self.daily_pnl = Decimal('0')
            self.daily_pnl_reset_time = now

    def update_daily_pnl(self, pnl: float):
        """Обновление дневного PnL."""
        self.daily_pnl += Decimal(str(pnl))
        logger.info(f"💰 Дневной PnL: {self.daily_pnl:+.2f} USDT")


class CopyTraderFollower:
    """Основной класс для копирования сигналов из Telegram."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.binance_client: Optional[AsyncClient] = None
        self.risk_manager = RiskManager(config)
        self.signal_parser = SignalParser()

        # Telegram
        self.tg_bot_token = config['tg_bot_token']
        self.tg_source_channel = config['tg_source_channel']
        self.tg_base_url = f"https://api.telegram.org/bot{self.tg_bot_token}"
        self.last_update_id = 0

        # Binance
        self.testnet = config.get('testnet', True)
        self.dry_run = config.get('dry_run', False)

    async def start(self):
        """Запуск копи-трейдера."""
        logger.info("🚀 Запуск Copy Trader Follower...")

        # Инициализация Binance клиента
        await self._init_binance()

        # Проверка подключения
        if not await self._check_connectivity():
            logger.error("❌ Не удалось подключиться к Binance")
            return

        # Получаем баланс
        balance = await self._get_balance()
        logger.info(f"💰 Баланс: {balance:.2f} USDT")

        # Запускаем polling Telegram
        self.running = True
        logger.info("✅ Копи-трейдер запущен и слушает сигналы...")
        logger.info(f"   Канал: {self.tg_source_channel}")
        logger.info(f"   Режим: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info("")

        try:
            await self._polling_loop()
        finally:
            await self.stop()

    async def stop(self):
        """Остановка копи-трейдера."""
        logger.info("🛑 Остановка копи-трейдера...")
        self.running = False

        if self.binance_client:
            await self.binance_client.close_connection()

    async def _init_binance(self):
        """Инициализация Binance клиента."""
        if AsyncClient is None:
            logger.error("❌ Binance клиент не доступен. Установите: pip install python-binance")
            return

        try:
            if self.testnet:
                # Testnet endpoints
                self.binance_client = await AsyncClient.create(
                    api_key=self.config['binance_api_key'],
                    api_secret=self.config['binance_api_secret'],
                    testnet=True
                )
            else:
                # Mainnet
                self.binance_client = await AsyncClient.create(
                    api_key=self.config['binance_api_key'],
                    api_secret=self.config['binance_api_secret']
                )

            logger.info(f"✅ Binance подключен ({'TESTNET' if self.testnet else 'MAINNET'})")

        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Binance: {e}")
            raise

    async def _check_connectivity(self) -> bool:
        """Проверка подключения к Binance."""
        try:
            if not self.binance_client:
                return False

            # Проверяем через получение статуса аккаунта
            account = await self.binance_client.futures_account()
            logger.debug(f"✅ Подключение к Binance успешно")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка проверки подключения: {e}")
            return False

    async def _get_balance(self) -> float:
        """Получение баланса USDT."""
        try:
            if not self.binance_client:
                return 0.0

            account = await self.binance_client.futures_account()
            balance = float(account['availableBalance'])
            return balance

        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return 0.0

    async def _polling_loop(self):
        """Основной цикл получения обновлений из Telegram."""
        while self.running:
            try:
                updates = await self._get_telegram_updates()

                for update in updates:
                    await self._process_telegram_update(update)

            except Exception as e:
                logger.error(f"Ошибка в polling loop: {e}")

            await asyncio.sleep(1)

    async def _get_telegram_updates(self) -> List[Dict]:
        """Получение обновлений из Telegram."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.tg_base_url}/getUpdates"
                params = {
                    "offset": self.last_update_id + 1,
                    "timeout": 30,
                }

                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=35)) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = data.get("result", [])

                        if updates:
                            self.last_update_id = updates[-1]["update_id"]

                        return updates

            return []

        except Exception as e:
            logger.debug(f"Ошибка получения обновлений: {e}")
            return []

    async def _process_telegram_update(self, update: Dict):
        """Обработка обновления из Telegram."""
        try:
            # Извлекаем сообщение
            message = None
            if "message" in update:
                message = update["message"]
            elif "channel_post" in update:
                message = update["channel_post"]

            if not message or "text" not in message:
                return

            text = message["text"]
            chat = message.get("chat", {})
            chat_username = chat.get("username", "")
            chat_id = chat.get("id", "")

            # Проверяем, что сообщение из нужного канала
            source_channel = self.tg_source_channel.replace('@', '')
            if source_channel and source_channel != chat_username and str(chat_id) != source_channel:
                logger.debug(f"Пропускаем сообщение из другого канала: @{chat_username}")
                return

            logger.debug(f"📩 Новое сообщение из @{chat_username}")

            # Парсим сигнал
            signal = self.signal_parser.parse_signal(text)
            if signal:
                logger.info(f"📡 Получен сигнал: {signal.action} {signal.side} {signal.symbol}")
                await self._handle_signal(signal)

        except Exception as e:
            logger.error(f"Ошибка обработки обновления: {e}")

    async def _handle_signal(self, signal: TradingSignal):
        """Обработка торгового сигнала."""
        try:
            if signal.action == "OPEN":
                await self._open_position(signal)
            elif signal.action == "CLOSE":
                await self._close_position(signal)

        except Exception as e:
            logger.error(f"❌ Ошибка обработки сигнала: {e}")

    async def _open_position(self, signal: TradingSignal):
        """Открытие позиции по сигналу."""
        logger.info(f"📈 Открытие позиции: {signal.side} {signal.symbol} @ {signal.entry_price}")

        # Получаем баланс
        balance = await self._get_balance()

        # Проверяем риски
        can_open, reason = await self.risk_manager.check_can_open_position(signal, balance)
        if not can_open:
            logger.warning(f"⚠️  Отклонено: {reason}")
            return

        # Рассчитываем размер позиции
        position_size_usdt = self.risk_manager.calculate_position_size(signal, balance)
        quantity = position_size_usdt / signal.entry_price

        logger.info(f"   Размер позиции: ${position_size_usdt:.2f} USDT ({quantity:.4f} {signal.symbol})")

        if self.dry_run:
            logger.info("   [DRY RUN] Ордер не выполнен")
            return

        # Выполняем ордер через Binance
        try:
            # Устанавливаем leverage
            if signal.leverage:
                await self.binance_client.futures_change_leverage(
                    symbol=signal.symbol,
                    leverage=signal.leverage
                )
                logger.info(f"   Leverage установлен: {signal.leverage}x")

            # Получаем информацию о символе для форматирования
            info = await self.binance_client.futures_exchange_info()
            symbol_info = next((s for s in info['symbols'] if s['symbol'] == signal.symbol), None)

            if not symbol_info:
                logger.error(f"Символ {signal.symbol} не найден")
                return

            # Форматируем quantity с правильной точностью
            qty_precision = symbol_info['quantityPrecision']
            quantity = float(Decimal(str(quantity)).quantize(
                Decimal(10) ** -qty_precision,
                rounding=ROUND_DOWN
            ))

            # Market ордер на вход
            side = SIDE_BUY if signal.side == "LONG" else SIDE_SELL
            order = await self.binance_client.futures_create_order(
                symbol=signal.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )

            logger.info(f"✅ Позиция открыта: {order['orderId']}")

            # Сохраняем информацию о позиции
            self.risk_manager.open_positions[signal.symbol] = {
                'signal': signal,
                'order': order,
                'entry_price': float(order.get('avgPrice', signal.entry_price)),
                'quantity': quantity,
                'side': signal.side,
                'opened_at': datetime.now()
            }

            # Устанавливаем SL/TP если указаны
            if signal.stop_loss or signal.take_profits:
                await self._set_sl_tp(signal, quantity)

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения ордера: {e}")

    async def _close_position(self, signal: TradingSignal):
        """Закрытие позиции по сигналу."""
        logger.info(f"📉 Закрытие позиции: {signal.symbol}")

        if signal.symbol not in self.risk_manager.open_positions:
            logger.warning(f"⚠️  Позиция {signal.symbol} не найдена")
            return

        position_info = self.risk_manager.open_positions[signal.symbol]

        if self.dry_run:
            logger.info("   [DRY RUN] Позиция не закрыта")
            del self.risk_manager.open_positions[signal.symbol]
            return

        try:
            # Закрываем позицию market ордером
            side = SIDE_SELL if position_info['side'] == "LONG" else SIDE_BUY
            order = await self.binance_client.futures_create_order(
                symbol=signal.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=position_info['quantity']
            )

            logger.info(f"✅ Позиция закрыта: {order['orderId']}")

            # Рассчитываем PnL
            exit_price = float(order.get('avgPrice', 0))
            entry_price = position_info['entry_price']
            quantity = position_info['quantity']

            if position_info['side'] == "LONG":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            logger.info(f"💰 PnL: {pnl:+.2f} USDT")

            # Обновляем дневной PnL
            self.risk_manager.update_daily_pnl(pnl)

            # Удаляем из открытых позиций
            del self.risk_manager.open_positions[signal.symbol]

        except Exception as e:
            logger.error(f"❌ Ошибка закрытия позиции: {e}")

    async def _set_sl_tp(self, signal: TradingSignal, quantity: float):
        """Установка Stop Loss и Take Profit."""
        try:
            # Stop Loss
            if signal.stop_loss:
                sl_side = SIDE_SELL if signal.side == "LONG" else SIDE_BUY
                sl_order = await self.binance_client.futures_create_order(
                    symbol=signal.symbol,
                    side=sl_side,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    stopPrice=signal.stop_loss,
                    closePosition=True
                )
                logger.info(f"   SL установлен: ${signal.stop_loss:.2f}")

            # Take Profit (первый уровень)
            if signal.take_profits:
                tp_price = signal.take_profits[0]
                tp_side = SIDE_SELL if signal.side == "LONG" else SIDE_BUY
                tp_order = await self.binance_client.futures_create_order(
                    symbol=signal.symbol,
                    side=tp_side,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    stopPrice=tp_price,
                    closePosition=True
                )
                logger.info(f"   TP установлен: ${tp_price:.2f}")

        except Exception as e:
            logger.error(f"⚠️  Ошибка установки SL/TP: {e}")
