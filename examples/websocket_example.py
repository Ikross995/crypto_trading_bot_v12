"""
Example usage of ResilientWebSocketManager for Binance WebSocket streams
"""

import asyncio
import logging
from exchange.websocket_manager import ResilientWebSocketManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Callback functions
async def on_message(data):
    """Handle received WebSocket messages"""
    if isinstance(data, dict):
        # Binance ticker update
        if 'e' in data and data['e'] == '24hrTicker':
            symbol = data.get('s', 'UNKNOWN')
            price = data.get('c', '0')
            change = data.get('P', '0')
            logger.info(f"📊 {symbol}: ${price} ({change}%)")

        # Binance trade update
        elif 'e' in data and data['e'] == 'trade':
            symbol = data.get('s', 'UNKNOWN')
            price = data.get('p', '0')
            quantity = data.get('q', '0')
            logger.info(f"💰 Trade: {symbol} {quantity} @ ${price}")

        else:
            logger.info(f"📨 Message: {data}")


async def on_connect():
    """Called when WebSocket connects"""
    logger.info("✅ Connected to Binance WebSocket!")


async def on_disconnect():
    """Called when WebSocket disconnects"""
    logger.warning("⚠️  Disconnected from Binance WebSocket")


async def on_error(error):
    """Called when error occurs"""
    logger.error(f"❌ WebSocket error: {error}")


async def binance_ticker_stream():
    """
    Example: Subscribe to Binance ticker stream
    """
    # All symbols ticker stream
    url = "wss://stream.binance.com:9443/ws/!ticker@arr"

    ws_manager = ResilientWebSocketManager(
        url=url,
        on_message=on_message,
        on_connect=on_connect,
        on_disconnect=on_disconnect,
        on_error=on_error,
        ping_interval=20,
        ping_timeout=10
    )

    # Connect
    connected = await ws_manager.connect()

    if connected:
        logger.info("🚀 Streaming Binance tickers...")

        # Run for 60 seconds
        await asyncio.sleep(60)

        # Disconnect
        await ws_manager.disconnect()

        # Show statistics
        ws_manager.log_stats()


async def binance_trade_stream(symbols=['BTCUSDT', 'ETHUSDT']):
    """
    Example: Subscribe to specific symbol trade streams
    """
    # Multiple streams
    streams = '/'.join([f"{s.lower()}@trade" for s in symbols])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    ws_manager = ResilientWebSocketManager(
        url=url,
        on_message=on_message,
        on_connect=on_connect,
        on_disconnect=on_disconnect,
        max_retries=10  # Max 10 reconnection attempts
    )

    connected = await ws_manager.connect()

    if connected:
        logger.info(f"🚀 Streaming trades for {symbols}...")

        # Run for 60 seconds
        await asyncio.sleep(60)

        # Disconnect
        await ws_manager.disconnect()


async def test_reconnection():
    """
    Test automatic reconnection

    This example demonstrates reconnection by using an invalid URL
    that will fail initially, then switching to valid URL
    """
    logger.info("🧪 Testing reconnection logic...")

    # Start with invalid URL (will trigger reconnection)
    url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

    reconnect_count = [0]

    async def on_reconnect():
        reconnect_count[0] += 1
        logger.info(f"🔄 Reconnection attempt #{reconnect_count[0]}")

    ws_manager = ResilientWebSocketManager(
        url=url,
        on_message=on_message,
        on_connect=on_reconnect,
        max_retries=5,
        ping_interval=10
    )

    # Try to connect (may fail and retry)
    await ws_manager.connect()

    # Let it run for 30 seconds
    await asyncio.sleep(30)

    # Disconnect
    await ws_manager.disconnect()

    logger.info(f"✅ Test completed. Reconnections: {reconnect_count[0]}")


async def message_buffering_example():
    """
    Example: Send messages while disconnected (buffering)
    """
    url = "wss://stream.binance.com:9443/ws"

    ws_manager = ResilientWebSocketManager(
        url=url,
        message_buffer_size=100
    )

    # Send messages before connection (will be buffered)
    logger.info("📦 Sending messages before connection...")

    for i in range(5):
        msg = {"method": "SUBSCRIBE", "params": [f"btcusdt@ticker"], "id": i}
        await ws_manager.send(msg)

    logger.info(f"📦 {len(ws_manager.message_buffer)} messages buffered")

    # Connect
    await ws_manager.connect()

    # Messages will be flushed automatically after connection

    await asyncio.sleep(10)
    await ws_manager.disconnect()


async def production_example():
    """
    Production-ready example with full monitoring
    """
    url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

    # Message counter
    message_count = [0]

    async def count_messages(data):
        message_count[0] += 1
        if message_count[0] % 10 == 0:
            logger.info(f"📊 Received {message_count[0]} messages")

    ws_manager = ResilientWebSocketManager(
        url=url,
        on_message=count_messages,
        on_connect=lambda: logger.info("✅ Production stream connected"),
        on_disconnect=lambda: logger.warning("⚠️  Production stream disconnected"),
        ping_interval=20,
        ping_timeout=10
    )

    # Connect
    await ws_manager.connect()

    # Monitor statistics every 30 seconds
    async def monitor_stats():
        while ws_manager.is_connected:
            await asyncio.sleep(30)
            ws_manager.log_stats()

    # Run monitoring in background
    monitor_task = asyncio.create_task(monitor_stats())

    # Run for 2 minutes
    await asyncio.sleep(120)

    # Cleanup
    monitor_task.cancel()
    await ws_manager.disconnect()

    # Final stats
    stats = ws_manager.get_stats()
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"   Messages received: {stats['total_messages_received']}")
    logger.info(f"   Bytes received: {stats['total_bytes_received']:,}")
    logger.info(f"   Uptime: {stats['current_uptime_seconds']:.1f}s")


if __name__ == '__main__':
    # Choose example to run:

    # Basic ticker stream
    # asyncio.run(binance_ticker_stream())

    # Trade stream for specific symbols
    # asyncio.run(binance_trade_stream(['BTCUSDT', 'ETHUSDT', 'BNBUSDT']))

    # Test reconnection logic
    # asyncio.run(test_reconnection())

    # Message buffering
    # asyncio.run(message_buffering_example())

    # Production example with monitoring
    asyncio.run(production_example())
