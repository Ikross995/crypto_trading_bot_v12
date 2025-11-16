#!/usr/bin/env python3
"""
Copy Trader Leader - публикация торговых сигналов в Telegram

Функциональность:
- Прослушивание сделок от основного бота через HTTP API
- Форматирование сигналов для копирования
- Публикация в Telegram канал
- Статистика производительности
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional
from aiohttp import web
from loguru import logger


class CopyTraderLeader:
    """
    Leader режим для публикации торговых сигналов.

    Этот класс создаёт HTTP сервер, который принимает сигналы
    от основного бота и публикует их в Telegram канал.
    """

    def __init__(self, telegram_bot, bot_port: int = 8080):
        """
        Args:
            telegram_bot: Экземпляр TelegramDashboardBot для отправки сообщений
            bot_port: Порт для HTTP сервера
        """
        self.telegram_bot = telegram_bot
        self.bot_port = bot_port
        self.app = None
        self.runner = None
        self.running = False

        # Статистика
        self.signals_published = 0
        self.started_at = datetime.now()

    async def start(self):
        """Запуск Leader сервера."""
        logger.info(f"🚀 Запуск Copy Trader Leader на порту {self.bot_port}...")

        # Создаём aiohttp приложение
        self.app = web.Application()
        self._setup_routes()

        # Запускаем сервер
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, '0.0.0.0', self.bot_port)
        await site.start()

        self.running = True
        logger.info(f"✅ Leader запущен и слушает порт {self.bot_port}")
        logger.info(f"   Endpoint: http://localhost:{self.bot_port}/signal")
        logger.info("")

        # Отправляем стартовое сообщение в канал
        await self._send_startup_message()

        # Держим сервер работающим
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass

    async def stop(self):
        """Остановка Leader сервера."""
        logger.info("🛑 Остановка Copy Trader Leader...")
        self.running = False

        if self.runner:
            await self.runner.cleanup()

    def _setup_routes(self):
        """Настройка HTTP маршрутов."""
        self.app.router.add_post('/signal', self.handle_signal)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/stats', self.handle_stats)

    async def handle_signal(self, request: web.Request) -> web.Response:
        """
        Обработка входящего торгового сигнала.

        Expected JSON:
        {
            "action": "OPEN" | "CLOSE",
            "symbol": "BTCUSDT",
            "side": "LONG" | "SHORT",
            "entry_price": 41250.0,
            "quantity": 0.01,
            "take_profits": [42780.0, 43500.0],
            "stop_loss": 40435.0,
            "leverage": 10,
            "position_size_usdt": 100.0,
            "reason": "Strong bullish signal"
        }
        """
        try:
            data = await request.json()

            # Валидация
            if 'action' not in data or 'symbol' not in data:
                return web.json_response({
                    'status': 'error',
                    'message': 'Missing required fields: action, symbol'
                }, status=400)

            # Обрабатываем сигнал
            await self._process_signal(data)

            self.signals_published += 1

            return web.json_response({
                'status': 'success',
                'message': 'Signal published to Telegram',
                'signals_count': self.signals_published
            })

        except Exception as e:
            logger.error(f"❌ Ошибка обработки сигнала: {e}")
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'uptime_seconds': (datetime.now() - self.started_at).total_seconds(),
            'signals_published': self.signals_published
        })

    async def handle_stats(self, request: web.Request) -> web.Response:
        """Статистика Leader."""
        uptime = datetime.now() - self.started_at
        return web.json_response({
            'started_at': self.started_at.isoformat(),
            'uptime_seconds': uptime.total_seconds(),
            'uptime_human': str(uptime),
            'signals_published': self.signals_published,
            'telegram_channel': self.telegram_bot.chat_id if hasattr(self.telegram_bot, 'chat_id') else 'N/A'
        })

    async def _process_signal(self, signal_data: Dict[str, Any]):
        """Обработка и публикация сигнала."""
        action = signal_data.get('action', 'OPEN')

        if action == 'OPEN':
            await self._publish_open_signal(signal_data)
        elif action == 'CLOSE':
            await self._publish_close_signal(signal_data)
        else:
            logger.warning(f"⚠️  Неизвестное действие: {action}")

    async def _publish_open_signal(self, signal: Dict[str, Any]):
        """Публикация сигнала на открытие позиции."""
        logger.info(f"📢 Публикация сигнала: {signal.get('side')} {signal.get('symbol')}")

        # Форматируем сообщение для копирования
        message = self._format_open_signal(signal)

        # Отправляем в Telegram
        try:
            await self.telegram_bot.send_message(message, parse_mode="HTML")
            logger.info(f"✅ Сигнал опубликован в Telegram")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки в Telegram: {e}")

    async def _publish_close_signal(self, signal: Dict[str, Any]):
        """Публикация сигнала на закрытие позиции."""
        logger.info(f"📢 Публикация закрытия: {signal.get('symbol')}")

        message = self._format_close_signal(signal)

        try:
            await self.telegram_bot.send_message(message, parse_mode="HTML")
            logger.info(f"✅ Сигнал закрытия опубликован")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки в Telegram: {e}")

    def _format_open_signal(self, signal: Dict[str, Any]) -> str:
        """
        Форматирование сигнала на открытие для Telegram.

        Создаёт сообщение в формате, понятном для парсера follower-ов.
        """
        side = signal.get('side', 'LONG')
        symbol = signal.get('symbol', 'UNKNOWN')
        entry = signal.get('entry_price', 0)
        leverage = signal.get('leverage', 1)
        position_size = signal.get('position_size_usdt', 0)

        # Эмодзи
        side_emoji = "🟢" if side == "LONG" else "🔴"
        direction_emoji = "📈" if side == "LONG" else "📉"

        # Базовое сообщение
        message = f"""╔════════════════════════╗
║  <b>⚡ NEW SIGNAL!</b>       ║
╚════════════════════════╝

<b>{side_emoji} {side} {symbol}  {direction_emoji}</b>
⏰ {datetime.now().strftime('%H:%M:%S UTC')}

╭─────────────────────╮
│ <b>📊 ENTRY INFO</b>       │
╰─────────────────────╯

<b>Entry:</b> ${entry:,.4f}
<b>Leverage:</b> {leverage}x
<b>Position:</b> ${position_size:,.2f}
"""

        # Добавляем Take Profit уровни
        take_profits = signal.get('take_profits', [])
        if take_profits:
            message += "\n╭─────────────────────╮\n│ <b>🎯 TARGETS</b>          │\n╰─────────────────────╯\n"
            for i, tp in enumerate(take_profits, 1):
                tp_dist = ((tp - entry) / entry * 100) if entry > 0 else 0
                message += f"\n<b>TP{i}:</b> ${tp:,.4f} (+{tp_dist:.2f}%)"

        # Добавляем Stop Loss
        stop_loss = signal.get('stop_loss')
        if stop_loss:
            sl_dist = ((stop_loss - entry) / entry * 100) if entry > 0 else 0
            message += f"\n\n╭─────────────────────╮\n│ <b>🛡️ PROTECTION</b>       │\n╰─────────────────────╯\n"
            message += f"\n<b>SL:</b> ${stop_loss:,.4f} ({sl_dist:+.2f}%)"

        # Причина/описание
        reason = signal.get('reason')
        if reason:
            message += f"\n\n<i>📡 {reason}</i>"

        # Хэштеги для поиска
        message += f"\n\n#{symbol} #{side} #CopyTrade"

        return message

    def _format_close_signal(self, signal: Dict[str, Any]) -> str:
        """Форматирование сигнала на закрытие."""
        symbol = signal.get('symbol', 'UNKNOWN')
        pnl = signal.get('pnl', 0)
        pnl_pct = signal.get('pnl_pct', 0)

        # Определяем эмодзи по результату
        if pnl >= 0:
            result_emoji = "✅"
            status = "PROFIT"
        else:
            result_emoji = "❌"
            status = "LOSS"

        message = f"""╔════════════════════════╗
║  <b>📊 POSITION CLOSED</b>  ║
╚════════════════════════╝

<b>{result_emoji} {symbol} - {status}</b>
⏰ {datetime.now().strftime('%H:%M:%S UTC')}

╭─────────────────────╮
│ <b>💰 RESULT</b>           │
╰─────────────────────╯

<b>PnL:</b> {pnl:+.2f} USDT
<b>ROI:</b> {pnl_pct:+.2f}%
"""

        reason = signal.get('reason')
        if reason:
            message += f"\n<i>📡 {reason}</i>"

        message += f"\n\n#{symbol} #Closed #CopyTrade"

        return message

    async def _send_startup_message(self):
        """Отправка стартового сообщения в канал."""
        message = f"""╔═══════════════════════════╗
║  <b>🤖 LEADER BOT STARTED</b>  ║
╚═══════════════════════════╝

Copy trading signals will be published here.

<b>⚡ Auto-copying is now active!</b>

Configure your follower bot to copy these signals automatically.

<i>Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""

        try:
            await self.telegram_bot.send_message(message, parse_mode="HTML")
        except Exception as e:
            logger.warning(f"⚠️  Не удалось отправить стартовое сообщение: {e}")


# Пример интеграции с основным ботом
class BotSignalHook:
    """
    Хук для интеграции с основным ботом.

    Устанавливается в LiveTradingEngine и отправляет сигналы в Leader.
    """

    def __init__(self, leader_url: str = "http://localhost:8080"):
        self.leader_url = leader_url

    async def on_position_opened(self, trade_info: Dict[str, Any]):
        """Вызывается когда основной бот открывает позицию."""
        signal = {
            'action': 'OPEN',
            'symbol': trade_info['symbol'],
            'side': trade_info['side'],
            'entry_price': trade_info['entry_price'],
            'quantity': trade_info['quantity'],
            'leverage': trade_info.get('leverage', 1),
            'position_size_usdt': trade_info.get('notional', 0),
            'take_profit': trade_info.get('take_profit'),
            'stop_loss': trade_info.get('stop_loss'),
            'reason': trade_info.get('reason', 'AI signal')
        }

        await self._send_signal(signal)

    async def on_position_closed(self, trade_info: Dict[str, Any]):
        """Вызывается когда основной бот закрывает позицию."""
        signal = {
            'action': 'CLOSE',
            'symbol': trade_info['symbol'],
            'pnl': trade_info.get('pnl', 0),
            'pnl_pct': trade_info.get('pnl_pct', 0),
            'reason': trade_info.get('reason', 'Position closed')
        }

        await self._send_signal(signal)

    async def _send_signal(self, signal: Dict[str, Any]):
        """Отправка сигнала в Leader."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.leader_url}/signal"
                async with session.post(url, json=signal) as response:
                    if response.status == 200:
                        logger.debug("✅ Сигнал отправлен в Leader")
                    else:
                        logger.warning(f"⚠️  Leader вернул статус {response.status}")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сигнала в Leader: {e}")
