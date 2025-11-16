#!/usr/bin/env python3
"""
WebSocket Manager for Binance Real-Time Updates
Управление WebSocket соединениями для получения обновлений в реальном времени
"""

import json
import asyncio
import aiohttp
from typing import Dict, List, Callable, Any, Optional
from loguru import logger
from binance import AsyncClient, BinanceSocketManager


class BinanceWebSocketManager:
    """Менеджер WebSocket соединений для Binance."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client: Optional[AsyncClient] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        self.active_streams: Dict[str, Any] = {}
        self.running = False

    async def initialize(self):
        """Инициализация клиента и менеджера сокетов."""
        self.client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        self.socket_manager = BinanceSocketManager(self.client)
        logger.info("✅ Binance WebSocket Manager initialized")

    async def start_user_stream(self, callback: Callable):
        """
        Запуск потока пользовательских данных.
        Получает обновления о:
        - ORDER_TRADE_UPDATE (обновления статуса ордера)
        - ACCOUNT_UPDATE (обновления баланса и позиций)
        """
        # Получение listen key
        listen_key = await self.client.stream_get_listen_key()

        # Создание user data stream
        user_stream = self.socket_manager.futures_user_socket(listen_key)

        async with user_stream as stream:
            while self.running:
                msg = await stream.recv()
                await self.process_user_update(msg, callback)

    async def process_user_update(self, msg: dict, callback: Callable):
        """Обработка обновлений пользовательских данных."""
        try:
            event_type = msg.get('e')

            if event_type == 'ORDER_TRADE_UPDATE':
                # Обновление статуса ордера
                await callback('order_update', {
                    'symbol': msg['o']['s'],
                    'order_id': msg['o']['i'],
                    'status': msg['o']['X'],
                    'executed_qty': msg['o']['z'],
                    'price': msg['o']['p'],
                    'side': msg['o']['S'],
                    'type': msg['o']['o']
                })

            elif event_type == 'ACCOUNT_UPDATE':
                # Обновление баланса и позиций
                positions = []
                for position in msg['a']['P']:
                    positions.append({
                        'symbol': position['s'],
                        'amount': position['pa'],
                        'entry_price': position['ep'],
                        'unrealized_pnl': position['up'],
                        'margin_type': position['mt']
                    })

                await callback('position_update', {
                    'positions': positions,
                    'balances': msg['a']['B']
                })

        except Exception as e:
            logger.error(f"❌ Error processing user update: {e}")

    async def subscribe_to_ticker(
        self, symbols: List[str], callback: Callable
    ):
        """
        Подписка на тикеры символов.
        Получает real-time цены.
        """
        for symbol in symbols:
            symbol_lower = symbol.lower()

            # Создание ticker stream
            ticker_stream = self.socket_manager.symbol_ticker_futures_socket(symbol)

            async with ticker_stream as stream:
                while self.running:
                    msg = await stream.recv()

                    await callback('ticker_update', {
                        'symbol': msg['s'],
                        'price': msg['c'],
                        'volume': msg['v'],
                        'high_24h': msg['h'],
                        'low_24h': msg['l'],
                        'change_24h': msg['P']
                    })

    async def subscribe_to_depth(
        self, symbol: str, callback: Callable
    ):
        """
        Подписка на стакан ордеров (order book).
        Получает топ 10 bid/ask ордеров.
        """
        depth_stream = self.socket_manager.depth_socket(symbol)

        async with depth_stream as stream:
            while self.running:
                msg = await stream.recv()

                # Топ 10 bid/ask ордеров
                await callback('depth_update', {
                    'symbol': symbol,
                    'bids': msg['b'][:10],  # Top 10 bids
                    'asks': msg['a'][:10],  # Top 10 asks
                    'timestamp': msg['E']
                })

    async def close(self):
        """Закрытие всех соединений."""
        self.running = False

        if self.client:
            await self.client.close_connection()

        logger.info("🛑 WebSocket connections closed")


# Пример использования
async def main():
    """Пример использования WebSocket менеджера."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    BOT_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BOT_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    if not BOT_API_KEY or not BOT_API_SECRET:
        print("❌ Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        return

    # Создаем менеджер
    ws_manager = BinanceWebSocketManager(BOT_API_KEY, BOT_API_SECRET)
    await ws_manager.initialize()

    # Callback для обработки обновлений
    async def handle_update(event_type: str, data: dict):
        print(f"📡 {event_type}: {data}")

    # Запускаем user stream
    ws_manager.running = True
    await ws_manager.start_user_stream(handle_update)


if __name__ == "__main__":
    asyncio.run(main())
