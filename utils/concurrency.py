"""
🔒 Concurrency Safety Manager
===============================

Prevents race conditions in async trading operations.
Ensures atomic updates to balance, positions, and orders.

Critical for preventing:
- Double spending
- Position size miscalculations
- Data corruption during concurrent trades
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class LockStats:
    """Statistics for lock usage monitoring"""
    total_acquisitions: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    contentions: int = 0  # Times had to wait
    last_acquired: Optional[datetime] = None


class SafeTradingState:
    """
    Thread-safe state manager for trading operations.

    Uses asyncio.Lock to prevent race conditions when:
    - Updating account balance
    - Modifying positions
    - Managing orders
    - Recording trades

    Usage:
        state = SafeTradingState()

        # Atomic balance update
        async with state.atomic_balance_update():
            balance -= cost

        # Atomic position update
        async with state.atomic_position_update():
            position.size += quantity
    """

    def __init__(self):
        # Separate locks for different resources
        self.balance_lock = asyncio.Lock()
        self.position_lock = asyncio.Lock()
        self.order_lock = asyncio.Lock()
        self.trade_lock = asyncio.Lock()

        # Statistics for monitoring
        self.stats = {
            'balance': LockStats(),
            'position': LockStats(),
            'order': LockStats(),
            'trade': LockStats()
        }

        logger.info("🔒 SafeTradingState initialized with 4 resource locks")

    @asynccontextmanager
    async def atomic_balance_update(self):
        """
        Ensures atomic balance modifications.

        Example:
            async with state.atomic_balance_update():
                # Critical section - only one coroutine at a time
                if self.balance >= cost:
                    self.balance -= cost
                    return True
                return False
        """
        start_time = asyncio.get_event_loop().time()

        # Check if lock is already held (contention)
        if self.balance_lock.locked():
            self.stats['balance'].contentions += 1
            logger.debug("⏳ Balance lock contention - waiting...")

        async with self.balance_lock:
            # Calculate wait time
            wait_time = asyncio.get_event_loop().time() - start_time

            # Update stats
            stats = self.stats['balance']
            stats.total_acquisitions += 1
            stats.total_wait_time += wait_time
            stats.max_wait_time = max(stats.max_wait_time, wait_time)
            stats.last_acquired = datetime.now(timezone.utc)

            if wait_time > 0.1:  # Log if waited >100ms
                logger.warning(f"⏰ Balance lock waited {wait_time*1000:.1f}ms")

            try:
                yield
            except Exception as e:
                logger.error(f"❌ Error in balance critical section: {e}")
                raise

    @asynccontextmanager
    async def atomic_position_update(self, symbol: Optional[str] = None):
        """
        Ensures atomic position modifications.

        Args:
            symbol: Optional symbol for logging

        Example:
            async with state.atomic_position_update('BTCUSDT'):
                position.size += quantity
                position.entry_price = (old_value + new_value) / 2
        """
        start_time = asyncio.get_event_loop().time()

        if self.position_lock.locked():
            self.stats['position'].contentions += 1
            logger.debug(f"⏳ Position lock contention ({symbol or 'all'}) - waiting...")

        async with self.position_lock:
            wait_time = asyncio.get_event_loop().time() - start_time

            stats = self.stats['position']
            stats.total_acquisitions += 1
            stats.total_wait_time += wait_time
            stats.max_wait_time = max(stats.max_wait_time, wait_time)
            stats.last_acquired = datetime.now(timezone.utc)

            if wait_time > 0.1:
                logger.warning(
                    f"⏰ Position lock ({symbol or 'all'}) waited {wait_time*1000:.1f}ms"
                )

            try:
                yield
            except Exception as e:
                logger.error(f"❌ Error in position critical section: {e}")
                raise

    @asynccontextmanager
    async def atomic_order_update(self, order_id: Optional[str] = None):
        """
        Ensures atomic order modifications.

        Args:
            order_id: Optional order ID for logging

        Example:
            async with state.atomic_order_update(order_id):
                order.status = 'FILLED'
                order.executed_qty = quantity
        """
        start_time = asyncio.get_event_loop().time()

        if self.order_lock.locked():
            self.stats['order'].contentions += 1
            logger.debug(f"⏳ Order lock contention ({order_id or 'all'}) - waiting...")

        async with self.order_lock:
            wait_time = asyncio.get_event_loop().time() - start_time

            stats = self.stats['order']
            stats.total_acquisitions += 1
            stats.total_wait_time += wait_time
            stats.max_wait_time = max(stats.max_wait_time, wait_time)
            stats.last_acquired = datetime.now(timezone.utc)

            if wait_time > 0.1:
                logger.warning(
                    f"⏰ Order lock ({order_id or 'all'}) waited {wait_time*1000:.1f}ms"
                )

            try:
                yield
            except Exception as e:
                logger.error(f"❌ Error in order critical section: {e}")
                raise

    @asynccontextmanager
    async def atomic_trade_record(self):
        """
        Ensures atomic trade recording.

        Example:
            async with state.atomic_trade_record():
                await db.save_trade(trade)
                self.total_trades += 1
        """
        start_time = asyncio.get_event_loop().time()

        if self.trade_lock.locked():
            self.stats['trade'].contentions += 1
            logger.debug("⏳ Trade lock contention - waiting...")

        async with self.trade_lock:
            wait_time = asyncio.get_event_loop().time() - start_time

            stats = self.stats['trade']
            stats.total_acquisitions += 1
            stats.total_wait_time += wait_time
            stats.max_wait_time = max(stats.max_wait_time, wait_time)
            stats.last_acquired = datetime.now(timezone.utc)

            if wait_time > 0.1:
                logger.warning(f"⏰ Trade lock waited {wait_time*1000:.1f}ms")

            try:
                yield
            except Exception as e:
                logger.error(f"❌ Error in trade critical section: {e}")
                raise

    def get_lock_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns detailed lock statistics for monitoring.

        Returns:
            Dictionary with statistics for each lock
        """
        result = {}

        for lock_name, stats in self.stats.items():
            avg_wait = (
                stats.total_wait_time / stats.total_acquisitions
                if stats.total_acquisitions > 0
                else 0
            )

            contention_rate = (
                (stats.contentions / stats.total_acquisitions) * 100
                if stats.total_acquisitions > 0
                else 0
            )

            result[lock_name] = {
                'total_acquisitions': stats.total_acquisitions,
                'contentions': stats.contentions,
                'contention_rate_pct': round(contention_rate, 2),
                'avg_wait_time_ms': round(avg_wait * 1000, 2),
                'max_wait_time_ms': round(stats.max_wait_time * 1000, 2),
                'total_wait_time_sec': round(stats.total_wait_time, 2),
                'last_acquired': stats.last_acquired.isoformat() if stats.last_acquired else None
            }

        return result

    def log_statistics(self):
        """Log lock statistics for monitoring"""
        stats = self.get_lock_statistics()

        logger.info("=" * 70)
        logger.info("🔒 [LOCK_STATS] Concurrency Statistics")
        logger.info("=" * 70)

        for lock_name, lock_stats in stats.items():
            logger.info(f"\n📊 {lock_name.upper()} Lock:")
            logger.info(f"   Acquisitions: {lock_stats['total_acquisitions']}")
            logger.info(f"   Contentions: {lock_stats['contentions']} ({lock_stats['contention_rate_pct']}%)")
            logger.info(f"   Avg Wait: {lock_stats['avg_wait_time_ms']:.2f}ms")
            logger.info(f"   Max Wait: {lock_stats['max_wait_time_ms']:.2f}ms")

        logger.info("=" * 70)

    def reset_statistics(self):
        """Reset all statistics (useful for testing)"""
        for stats in self.stats.values():
            stats.total_acquisitions = 0
            stats.total_wait_time = 0.0
            stats.max_wait_time = 0.0
            stats.contentions = 0
            stats.last_acquired = None

        logger.info("🔄 Lock statistics reset")


class DeadlockDetector:
    """
    Detects potential deadlocks in lock acquisition.

    Simple implementation: warns if lock held for too long.
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds
        self.active_locks: Dict[str, datetime] = {}

    async def monitor_lock(self, lock_name: str, lock: asyncio.Lock):
        """Monitor a lock for potential deadlock"""
        if lock.locked():
            if lock_name not in self.active_locks:
                self.active_locks[lock_name] = datetime.now(timezone.utc)
            else:
                # Check how long lock has been held
                held_time = (datetime.now(timezone.utc) - self.active_locks[lock_name]).total_seconds()

                if held_time > self.timeout_seconds:
                    logger.critical(
                        f"🚨 DEADLOCK WARNING: {lock_name} lock held for {held_time:.1f}s "
                        f"(timeout: {self.timeout_seconds}s)"
                    )
        else:
            # Lock released
            if lock_name in self.active_locks:
                del self.active_locks[lock_name]


# Global instance for easy access
_global_safe_state: Optional[SafeTradingState] = None


def get_global_safe_state() -> SafeTradingState:
    """
    Returns global SafeTradingState instance (singleton pattern).

    Usage:
        from utils.concurrency import get_global_safe_state

        state = get_global_safe_state()
        async with state.atomic_balance_update():
            # Your code here
    """
    global _global_safe_state

    if _global_safe_state is None:
        _global_safe_state = SafeTradingState()

    return _global_safe_state


def reset_global_safe_state():
    """Reset global state (useful for testing)"""
    global _global_safe_state
    _global_safe_state = None


__all__ = [
    'SafeTradingState',
    'DeadlockDetector',
    'get_global_safe_state',
    'reset_global_safe_state'
]
