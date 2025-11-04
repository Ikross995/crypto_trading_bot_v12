"""
⏱️ Adaptive API Rate Limiter
==============================

Prevents API rate limit violations (429 errors) with:
- Token bucket algorithm
- Adaptive rate reduction after 429 errors
- Per-endpoint rate limiting
- Request queue with prioritization
- Comprehensive statistics

Essential for Binance API limits:
- 1200 requests/minute (20/second)
- Specific endpoint limits
- Weight-based limiting
"""

import asyncio
import time
import logging
from collections import deque, defaultdict
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Request priority levels"""
    CRITICAL = 0  # Account balance, positions (highest)
    HIGH = 1      # Order execution, cancellation
    NORMAL = 2    # Market data, ticker
    LOW = 3       # Historical data, optional requests


@dataclass
class RateLimitStats:
    """Statistics for rate limiter"""
    total_requests: int = 0
    requests_allowed: int = 0
    requests_throttled: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    rate_limit_hits: int = 0  # 429 errors
    adaptive_reductions: int = 0
    last_request: Optional[datetime] = None


class AdaptiveRateLimiter:
    """
    Token bucket rate limiter with adaptive adjustment.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Adaptive rate reduction after 429 errors
    - Per-endpoint limits
    - Request prioritization
    - Automatic recovery

    Usage:
        limiter = AdaptiveRateLimiter(max_calls=1200, time_window=60)

        # Before API call
        await limiter.acquire(endpoint='/api/v3/order', priority=Priority.HIGH)

        # After 429 error
        limiter.on_rate_limit_error()

    Binance Limits:
    - General: 1200 requests/minute
    - Order endpoints: 10/second (100/10s)
    - WebSocket: 5 connections, 300 messages/5min
    """

    def __init__(
        self,
        max_calls: int = 1200,
        time_window: int = 60,
        burst_size: Optional[int] = None,
        enable_adaptive: bool = True
    ):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed per time window
            time_window: Time window in seconds
            burst_size: Max burst size (default: max_calls)
            enable_adaptive: Enable adaptive rate reduction
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.burst_size = burst_size or max_calls
        self.enable_adaptive = enable_adaptive

        # Token bucket
        self.tokens = float(self.burst_size)
        self.last_refill = time.time()

        # Adaptive limits
        self.current_max_calls = max_calls
        self.adaptive_factor = 1.0
        self.recovery_task: Optional[asyncio.Task] = None

        # Request tracking
        self.calls = deque()
        self.stats = RateLimitStats()

        # Per-endpoint limits (Binance specific)
        self.endpoint_limits: Dict[str, deque] = defaultdict(deque)
        self.endpoint_max_calls: Dict[str, tuple] = {
            '/api/v3/order': (10, 1),  # 10 per 1 second
            '/api/v3/allOrders': (40, 10),  # 40 per 10 seconds
            '/fapi/v1/order': (20, 1),  # Futures: 20 per second
        }

        # Lock for thread safety
        self.lock = asyncio.Lock()

        # Priority queue
        self.pending_requests: Dict[Priority, asyncio.Queue] = {
            Priority.CRITICAL: asyncio.Queue(),
            Priority.HIGH: asyncio.Queue(),
            Priority.NORMAL: asyncio.Queue(),
            Priority.LOW: asyncio.Queue()
        }

        logger.info(
            f"⏱️ RateLimiter initialized: {max_calls} calls/{time_window}s "
            f"(burst: {self.burst_size})"
        )

    def _refill_tokens(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate tokens to add
        tokens_to_add = (self.current_max_calls / self.time_window) * elapsed

        # Add tokens up to burst size
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)

        self.last_refill = now

    async def acquire(
        self,
        endpoint: Optional[str] = None,
        weight: int = 1,
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """
        Acquire permission to make API call.

        Args:
            endpoint: API endpoint (for per-endpoint limits)
            weight: Request weight (default: 1)
            priority: Request priority

        Returns:
            True when request is allowed
        """
        start_time = time.time()
        self.stats.total_requests += 1

        async with self.lock:
            # Refill tokens
            self._refill_tokens()

            # Check if we have enough tokens
            wait_time = 0

            while self.tokens < weight:
                # Calculate wait time
                tokens_needed = weight - self.tokens
                wait_time = (tokens_needed / self.current_max_calls) * self.time_window

                # Cap wait time at time_window
                wait_time = min(wait_time, self.time_window)

                # Update stats
                self.stats.requests_throttled += 1

                logger.debug(
                    f"⏱️ Rate limit reached. Waiting {wait_time:.2f}s "
                    f"(tokens: {self.tokens:.2f}/{self.burst_size})"
                )

                # Wait and refill
                await asyncio.sleep(wait_time)
                self._refill_tokens()

            # Check endpoint-specific limits
            if endpoint and endpoint in self.endpoint_max_calls:
                endpoint_wait = await self._check_endpoint_limit(endpoint)
                wait_time += endpoint_wait

            # Consume tokens
            self.tokens -= weight

            # Update stats
            total_wait = time.time() - start_time
            self.stats.requests_allowed += 1
            self.stats.total_wait_time += total_wait
            self.stats.max_wait_time = max(self.stats.max_wait_time, total_wait)
            self.stats.last_request = datetime.now(timezone.utc)

            # Track call
            self.calls.append(time.time())
            if endpoint:
                self.endpoint_limits[endpoint].append(time.time())

            # Log if significant wait
            if total_wait > 0.5:
                logger.warning(
                    f"⏰ Rate limiter delayed request by {total_wait:.2f}s "
                    f"(endpoint: {endpoint or 'general'})"
                )

            return True

    async def _check_endpoint_limit(self, endpoint: str) -> float:
        """
        Check per-endpoint rate limit.

        Returns:
            Wait time in seconds
        """
        if endpoint not in self.endpoint_max_calls:
            return 0

        max_calls, time_window = self.endpoint_max_calls[endpoint]
        now = time.time()

        # Remove old calls
        calls = self.endpoint_limits[endpoint]
        while calls and calls[0] < now - time_window:
            calls.popleft()

        # Check if at limit
        if len(calls) >= max_calls:
            # Calculate wait time
            oldest_call = calls[0]
            wait_time = time_window - (now - oldest_call)

            if wait_time > 0:
                logger.debug(
                    f"⏱️ Endpoint limit hit ({endpoint}): "
                    f"waiting {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
                return wait_time

        return 0

    def on_rate_limit_error(self, retry_after: Optional[int] = None):
        """
        Called when 429 (rate limit) error is received.

        Args:
            retry_after: Seconds to wait (from Retry-After header)
        """
        self.stats.rate_limit_hits += 1

        logger.warning(
            f"⚠️ Rate limit hit! "
            f"(total hits: {self.stats.rate_limit_hits})"
        )

        if self.enable_adaptive:
            self._reduce_rate()

        # Honor retry_after if provided
        if retry_after:
            logger.info(f"⏰ Waiting {retry_after}s (from Retry-After header)")
            # This should be awaited by caller
            return retry_after

    def _reduce_rate(self):
        """Reduce rate adaptively after 429 error"""
        # Reduce by 20%
        self.adaptive_factor *= 0.8
        self.current_max_calls = int(self.max_calls * self.adaptive_factor)
        self.stats.adaptive_reductions += 1

        logger.warning(
            f"📉 Rate limit reduced to {self.current_max_calls} calls/{self.time_window}s "
            f"({self.adaptive_factor*100:.0f}% of original)"
        )

        # Cancel existing recovery task
        if self.recovery_task and not self.recovery_task.done():
            self.recovery_task.cancel()

        # Schedule gradual recovery
        self.recovery_task = asyncio.create_task(self._gradual_recovery())

    async def _gradual_recovery(self):
        """Gradually recover to normal rate after cooldown"""
        try:
            # Wait 5 minutes before starting recovery
            await asyncio.sleep(300)

            logger.info("📈 Starting gradual rate limit recovery...")

            # Recover in 10% increments every minute
            while self.adaptive_factor < 0.95:
                await asyncio.sleep(60)

                self.adaptive_factor = min(1.0, self.adaptive_factor + 0.1)
                self.current_max_calls = int(self.max_calls * self.adaptive_factor)

                logger.info(
                    f"📈 Rate limit recovered to {self.current_max_calls} calls/{self.time_window}s "
                    f"({self.adaptive_factor*100:.0f}%)"
                )

            # Full recovery
            self.adaptive_factor = 1.0
            self.current_max_calls = self.max_calls

            logger.info("✅ Rate limit fully recovered to normal")

        except asyncio.CancelledError:
            logger.debug("Rate recovery cancelled")

    def reset(self):
        """Reset rate limiter to initial state"""
        self.tokens = float(self.burst_size)
        self.last_refill = time.time()
        self.adaptive_factor = 1.0
        self.current_max_calls = self.max_calls
        self.calls.clear()

        for queue in self.endpoint_limits.values():
            queue.clear()

        logger.info("🔄 Rate limiter reset to initial state")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with stats
        """
        avg_wait = (
            self.stats.total_wait_time / self.stats.total_requests
            if self.stats.total_requests > 0
            else 0
        )

        throttle_rate = (
            (self.stats.requests_throttled / self.stats.total_requests) * 100
            if self.stats.total_requests > 0
            else 0
        )

        # Calculate current rate
        now = time.time()
        recent_calls = sum(1 for t in self.calls if t > now - self.time_window)
        current_rate = (recent_calls / self.time_window) * 60  # requests per minute

        return {
            'total_requests': self.stats.total_requests,
            'requests_allowed': self.stats.requests_allowed,
            'requests_throttled': self.stats.requests_throttled,
            'throttle_rate_pct': round(throttle_rate, 2),
            'avg_wait_time_ms': round(avg_wait * 1000, 2),
            'max_wait_time_ms': round(self.stats.max_wait_time * 1000, 2),
            'total_wait_time_sec': round(self.stats.total_wait_time, 2),
            'rate_limit_hits': self.stats.rate_limit_hits,
            'adaptive_reductions': self.stats.adaptive_reductions,
            'current_limit': self.current_max_calls,
            'original_limit': self.max_calls,
            'adaptive_factor': round(self.adaptive_factor, 2),
            'current_rate_per_min': round(current_rate, 2),
            'available_tokens': round(self.tokens, 2),
            'last_request': (
                self.stats.last_request.isoformat()
                if self.stats.last_request else None
            )
        }

    def log_stats(self):
        """Log rate limiter statistics"""
        stats = self.get_stats()

        logger.info("=" * 70)
        logger.info("⏱️ [RATE_LIMITER] Statistics")
        logger.info("=" * 70)
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Allowed: {stats['requests_allowed']} | Throttled: {stats['requests_throttled']}")
        logger.info(f"Throttle Rate: {stats['throttle_rate_pct']:.1f}%")
        logger.info(f"Avg Wait: {stats['avg_wait_time_ms']:.2f}ms | Max: {stats['max_wait_time_ms']:.2f}ms")
        logger.info(f"Rate Limit Hits (429): {stats['rate_limit_hits']}")

        if stats['adaptive_reductions'] > 0:
            logger.info(f"📉 Adaptive Reductions: {stats['adaptive_reductions']}")
            logger.info(f"Current Limit: {stats['current_limit']}/{stats['original_limit']} ({stats['adaptive_factor']:.0%})")

        logger.info(f"Current Rate: {stats['current_rate_per_min']:.1f} req/min")
        logger.info(f"Available Tokens: {stats['available_tokens']:.1f}")
        logger.info("=" * 70)


# Global instance for easy access
_global_rate_limiter: Optional[AdaptiveRateLimiter] = None


def get_global_rate_limiter(
    max_calls: int = 1200,
    time_window: int = 60
) -> AdaptiveRateLimiter:
    """
    Get global rate limiter instance (singleton).

    Usage:
        from utils.rate_limiter import get_global_rate_limiter

        limiter = get_global_rate_limiter()
        await limiter.acquire(endpoint='/api/v3/order')
    """
    global _global_rate_limiter

    if _global_rate_limiter is None:
        _global_rate_limiter = AdaptiveRateLimiter(
            max_calls=max_calls,
            time_window=time_window
        )

    return _global_rate_limiter


def reset_global_rate_limiter():
    """Reset global instance (useful for testing)"""
    global _global_rate_limiter
    _global_rate_limiter = None


__all__ = [
    'AdaptiveRateLimiter',
    'Priority',
    'get_global_rate_limiter',
    'reset_global_rate_limiter'
]
