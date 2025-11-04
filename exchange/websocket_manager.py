"""
🔌 Resilient WebSocket Manager
================================

Provides bulletproof WebSocket connection with:
- Automatic reconnection with exponential backoff
- Ping/Pong heartbeat monitoring
- Connection state tracking
- Event callbacks for connection changes
- Message queue for offline buffering

Prevents bot downtime due to network issues.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime, timezone
from dataclasses import dataclass
import json

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed,
        WebSocketException,
        InvalidStatusCode
    )
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets library not installed - WebSocket features disabled")

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class ConnectionStats:
    """Statistics for WebSocket connection"""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_reconnects: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    last_connected: Optional[datetime] = None
    last_disconnected: Optional[datetime] = None
    current_uptime_start: Optional[datetime] = None


class ResilientWebSocketManager:
    """
    Production-grade WebSocket manager with auto-reconnection.

    Features:
    - Exponential backoff (2s, 4s, 8s, 16s, 32s, 60s max)
    - Ping/Pong heartbeat monitoring
    - Automatic reconnection on failure
    - Message buffering during reconnection
    - Connection state callbacks
    - Comprehensive statistics

    Usage:
        ws_manager = ResilientWebSocketManager(
            url='wss://stream.binance.com:9443/ws',
            on_message=handle_message
        )

        await ws_manager.connect()

        # Manager handles reconnections automatically
        # Your on_message callback will receive messages
    """

    def __init__(
        self,
        url: str,
        on_message: Optional[Callable] = None,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        max_retries: int = None,  # None = infinite
        ping_interval: int = 20,
        ping_timeout: int = 10,
        message_buffer_size: int = 1000
    ):
        """
        Initialize WebSocket manager.

        Args:
            url: WebSocket URL
            on_message: Callback for received messages
            on_connect: Callback when connection established
            on_disconnect: Callback when connection lost
            on_error: Callback for errors
            max_retries: Maximum reconnection attempts (None = infinite)
            ping_interval: Seconds between ping messages
            ping_timeout: Seconds to wait for pong response
            message_buffer_size: Max messages to buffer when offline
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required")

        self.url = url
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error

        self.max_retries = max_retries
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.message_buffer_size = message_buffer_size

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.state = ConnectionState.DISCONNECTED

        self.retry_count = 0
        self.backoff_base = 2
        self.max_backoff = 60  # Max 60 seconds between retries

        self.stats = ConnectionStats()
        self.message_buffer: List[str] = []

        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        self._shutdown = False
        self._manual_disconnect = False

        logger.info(f"🔌 WebSocketManager initialized: {url}")

    async def connect(self) -> bool:
        """
        Connect to WebSocket with automatic retry.

        Returns:
            True if connected successfully
        """
        if self.state == ConnectionState.CONNECTED:
            logger.warning("Already connected")
            return True

        self.state = ConnectionState.CONNECTING
        self.stats.total_connections += 1

        return await self._connect_with_retry()

    async def _connect_with_retry(self) -> bool:
        """
        Internal method to connect with exponential backoff.
        """
        while not self._shutdown:
            try:
                logger.info(f"🔌 Connecting to {self.url}...")

                # Create WebSocket connection
                self.websocket = await websockets.connect(
                    self.url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=5
                )

                # Connection successful
                self.state = ConnectionState.CONNECTED
                self.retry_count = 0
                self.stats.successful_connections += 1
                self.stats.last_connected = datetime.now(timezone.utc)
                self.stats.current_uptime_start = datetime.now(timezone.utc)

                logger.info("✅ WebSocket connected successfully!")

                # Trigger connect callback
                if self.on_connect:
                    try:
                        await self.on_connect()
                    except Exception as e:
                        logger.error(f"Error in on_connect callback: {e}")

                # Flush buffered messages
                await self._flush_message_buffer()

                # Start receive and heartbeat tasks
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                return True

            except (ConnectionClosed, WebSocketException, OSError) as e:
                self.stats.failed_connections += 1
                self.retry_count += 1

                # Check if max retries exceeded
                if self.max_retries and self.retry_count > self.max_retries:
                    logger.critical(
                        f"❌ Max retries ({self.max_retries}) exceeded. Giving up."
                    )
                    self.state = ConnectionState.FAILED
                    return False

                # Calculate backoff time (exponential)
                backoff_time = min(
                    self.max_backoff,
                    self.backoff_base ** (self.retry_count - 1)
                )

                logger.warning(
                    f"⚠️  Connection failed (attempt {self.retry_count}"
                    f"{f'/{self.max_retries}' if self.max_retries else ''}): {e}"
                )
                logger.info(f"🔄 Retrying in {backoff_time}s...")

                self.state = ConnectionState.RECONNECTING

                # Wait before retry
                await asyncio.sleep(backoff_time)

            except Exception as e:
                logger.error(f"❌ Unexpected error during connection: {e}")
                self.stats.failed_connections += 1

                # Wait before retry
                await asyncio.sleep(5)

        return False

    async def _receive_loop(self):
        """
        Continuously receive messages from WebSocket.
        """
        try:
            while not self._shutdown and self.websocket:
                try:
                    message = await self.websocket.recv()

                    self.stats.total_messages_received += 1
                    self.stats.total_bytes_received += len(message)

                    # Parse JSON if possible
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        data = message

                    # Trigger message callback
                    if self.on_message:
                        try:
                            await self.on_message(data)
                        except Exception as e:
                            logger.error(f"Error in on_message callback: {e}")

                except ConnectionClosed:
                    logger.warning("🔌 WebSocket connection closed")
                    break

                except Exception as e:
                    logger.error(f"Error receiving message: {e}")

                    if self.on_error:
                        try:
                            await self.on_error(e)
                        except Exception:
                            pass

        finally:
            # Connection lost - trigger reconnection
            if not self._manual_disconnect and not self._shutdown:
                logger.warning("📡 Connection lost - initiating reconnection...")
                self._reconnect_task = asyncio.create_task(self._handle_reconnection())

    async def _heartbeat_loop(self):
        """
        Monitor connection health via ping/pong.
        """
        while not self._shutdown and self.websocket:
            try:
                await asyncio.sleep(self.ping_interval)

                if self.websocket and self.state == ConnectionState.CONNECTED:
                    # Ping is handled automatically by websockets library
                    # We just check if connection is still alive
                    if self.websocket.closed:
                        logger.warning("💔 Heartbeat detected closed connection")
                        break

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def _handle_reconnection(self):
        """
        Handle reconnection after connection loss.
        """
        if self.state == ConnectionState.RECONNECTING:
            return  # Already reconnecting

        self.state = ConnectionState.DISCONNECTED
        self.stats.total_reconnects += 1
        self.stats.last_disconnected = datetime.now(timezone.utc)

        # Trigger disconnect callback
        if self.on_disconnect:
            try:
                await self.on_disconnect()
            except Exception as e:
                logger.error(f"Error in on_disconnect callback: {e}")

        # Attempt reconnection
        logger.info("🔄 Attempting automatic reconnection...")
        await self._connect_with_retry()

    async def send(self, message: str | dict) -> bool:
        """
        Send message through WebSocket.

        Args:
            message: String or dict (will be JSON-encoded)

        Returns:
            True if sent successfully
        """
        if isinstance(message, dict):
            message = json.dumps(message)

        # Buffer message if not connected
        if self.state != ConnectionState.CONNECTED or not self.websocket:
            if len(self.message_buffer) < self.message_buffer_size:
                self.message_buffer.append(message)
                logger.debug(
                    f"📦 Message buffered (queue: {len(self.message_buffer)})"
                )
            else:
                logger.warning("⚠️  Message buffer full - message dropped!")
            return False

        try:
            await self.websocket.send(message)
            self.stats.total_messages_sent += 1
            self.stats.total_bytes_sent += len(message)
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")

            # Buffer failed message
            if len(self.message_buffer) < self.message_buffer_size:
                self.message_buffer.append(message)

            return False

    async def _flush_message_buffer(self):
        """
        Send all buffered messages after reconnection.
        """
        if not self.message_buffer:
            return

        logger.info(f"📤 Flushing {len(self.message_buffer)} buffered messages...")

        failed_messages = []

        for message in self.message_buffer:
            try:
                await self.websocket.send(message)
                self.stats.total_messages_sent += 1
                self.stats.total_bytes_sent += len(message)
            except Exception as e:
                logger.error(f"Failed to flush message: {e}")
                failed_messages.append(message)

        # Keep failed messages in buffer
        self.message_buffer = failed_messages

        if failed_messages:
            logger.warning(f"⚠️  {len(failed_messages)} messages failed to flush")
        else:
            logger.info("✅ All buffered messages sent successfully")

    async def disconnect(self):
        """
        Gracefully disconnect WebSocket.
        """
        self._manual_disconnect = True
        self._shutdown = True

        logger.info("🔌 Disconnecting WebSocket...")

        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self._reconnect_task:
            self._reconnect_task.cancel()

        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")

        self.state = ConnectionState.DISCONNECTED
        self.stats.last_disconnected = datetime.now(timezone.utc)

        logger.info("✅ WebSocket disconnected")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Dictionary with stats
        """
        uptime = None
        if self.stats.current_uptime_start:
            uptime = (
                datetime.now(timezone.utc) - self.stats.current_uptime_start
            ).total_seconds()

        return {
            'state': self.state.value,
            'total_connections': self.stats.total_connections,
            'successful_connections': self.stats.successful_connections,
            'failed_connections': self.stats.failed_connections,
            'connection_success_rate': (
                (self.stats.successful_connections / self.stats.total_connections * 100)
                if self.stats.total_connections > 0 else 0
            ),
            'total_reconnects': self.stats.total_reconnects,
            'total_messages_sent': self.stats.total_messages_sent,
            'total_messages_received': self.stats.total_messages_received,
            'total_bytes_sent': self.stats.total_bytes_sent,
            'total_bytes_received': self.stats.total_bytes_received,
            'buffered_messages': len(self.message_buffer),
            'current_uptime_seconds': uptime,
            'last_connected': (
                self.stats.last_connected.isoformat()
                if self.stats.last_connected else None
            ),
            'last_disconnected': (
                self.stats.last_disconnected.isoformat()
                if self.stats.last_disconnected else None
            )
        }

    def log_stats(self):
        """Log connection statistics"""
        stats = self.get_stats()

        logger.info("=" * 70)
        logger.info("🔌 [WEBSOCKET_STATS] Connection Statistics")
        logger.info("=" * 70)
        logger.info(f"State: {stats['state']}")
        logger.info(f"Connections: {stats['successful_connections']}/{stats['total_connections']}")
        logger.info(f"Success Rate: {stats['connection_success_rate']:.1f}%")
        logger.info(f"Reconnects: {stats['total_reconnects']}")
        logger.info(f"Messages: ⬆️ {stats['total_messages_sent']} | ⬇️ {stats['total_messages_received']}")
        logger.info(f"Bytes: ⬆️ {stats['total_bytes_sent']:,} | ⬇️ {stats['total_bytes_received']:,}")

        if stats['current_uptime_seconds']:
            logger.info(f"Uptime: {stats['current_uptime_seconds']:.1f}s")

        if stats['buffered_messages'] > 0:
            logger.warning(f"⚠️  Buffered messages: {stats['buffered_messages']}")

        logger.info("=" * 70)

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected"""
        return self.state == ConnectionState.CONNECTED and self.websocket is not None


__all__ = ['ResilientWebSocketManager', 'ConnectionState', 'ConnectionStats']
