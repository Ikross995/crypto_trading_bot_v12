"""
Tests for concurrency safety module
"""

import pytest
import asyncio
from decimal import Decimal
from utils.concurrency import SafeTradingState, get_global_safe_state, reset_global_safe_state


@pytest.fixture
def safe_state():
    """Create fresh SafeTradingState for each test"""
    state = SafeTradingState()
    state.reset_statistics()
    return state


@pytest.mark.asyncio
async def test_atomic_balance_update(safe_state):
    """Test that balance updates are atomic"""
    balance = Decimal('1000.0')
    results = []

    async def deduct_balance(amount):
        async with safe_state.atomic_balance_update():
            nonlocal balance
            # Simulate some processing
            await asyncio.sleep(0.01)
            if balance >= amount:
                balance -= amount
                results.append(f"Deducted {amount}")
                return True
            results.append(f"Insufficient funds for {amount}")
            return False

    # Run 10 concurrent deductions
    tasks = [deduct_balance(Decimal('100')) for _ in range(10)]
    await asyncio.gather(*tasks)

    # Balance should be 0 (all 10 deductions successful)
    assert balance == Decimal('0'), f"Expected 0, got {balance}"
    assert len([r for r in results if 'Deducted' in r]) == 10


@pytest.mark.asyncio
async def test_race_condition_without_lock():
    """Demonstrate race condition without lock"""
    balance = 1000
    results = []

    async def deduct_no_lock(amount):
        nonlocal balance
        # NO LOCK - race condition likely!
        await asyncio.sleep(0.001)  # Simulate processing
        if balance >= amount:
            old_balance = balance
            await asyncio.sleep(0.001)  # Simulate more processing
            balance = old_balance - amount
            results.append(amount)

    # Run 5 concurrent deductions
    tasks = [deduct_no_lock(100) for _ in range(5)]
    await asyncio.gather(*tasks)

    # Without lock, balance might be incorrect
    print(f"Balance without lock: {balance}")
    print(f"Successful deductions: {len(results)}")
    # This demonstrates why locks are needed!


@pytest.mark.asyncio
async def test_position_lock(safe_state):
    """Test atomic position updates"""
    position_size = Decimal('0')

    async def add_to_position(amount, symbol):
        async with safe_state.atomic_position_update(symbol):
            nonlocal position_size
            await asyncio.sleep(0.01)
            position_size += amount

    # Concurrent additions
    tasks = [
        add_to_position(Decimal('0.1'), 'BTCUSDT') for _ in range(10)
    ]
    await asyncio.gather(*tasks)

    assert position_size == Decimal('1.0')


@pytest.mark.asyncio
async def test_lock_statistics(safe_state):
    """Test lock statistics collection"""
    async def use_balance_lock():
        async with safe_state.atomic_balance_update():
            await asyncio.sleep(0.01)

    # Use lock 5 times
    for _ in range(5):
        await use_balance_lock()

    stats = safe_state.get_lock_statistics()

    assert stats['balance']['total_acquisitions'] == 5
    assert stats['balance']['contentions'] == 0  # No contention (sequential)


@pytest.mark.asyncio
async def test_lock_contention(safe_state):
    """Test lock contention detection"""
    async def long_operation():
        async with safe_state.atomic_balance_update():
            await asyncio.sleep(0.1)  # Hold lock for 100ms

    async def quick_operation():
        async with safe_state.atomic_balance_update():
            await asyncio.sleep(0.01)

    # Start long operation
    long_task = asyncio.create_task(long_operation())
    await asyncio.sleep(0.01)  # Let it acquire lock

    # Try quick operation (will have to wait)
    quick_task = asyncio.create_task(quick_operation())

    await asyncio.gather(long_task, quick_task)

    stats = safe_state.get_lock_statistics()

    # Should have 1 contention
    assert stats['balance']['contentions'] >= 1


@pytest.mark.asyncio
async def test_global_singleton():
    """Test global singleton pattern"""
    reset_global_safe_state()

    state1 = get_global_safe_state()
    state2 = get_global_safe_state()

    assert state1 is state2  # Same instance


@pytest.mark.asyncio
async def test_multiple_locks_different_resources():
    """Test that different resource locks don't block each other"""
    safe_state = SafeTradingState()

    start_time = asyncio.get_event_loop().time()

    async def balance_op():
        async with safe_state.atomic_balance_update():
            await asyncio.sleep(0.1)

    async def position_op():
        async with safe_state.atomic_position_update():
            await asyncio.sleep(0.1)

    # Run concurrently - should take ~0.1s, not 0.2s
    await asyncio.gather(balance_op(), position_op())

    elapsed = asyncio.get_event_loop().time() - start_time

    # Should be close to 0.1s (concurrent), not 0.2s (sequential)
    assert elapsed < 0.15, f"Locks blocked each other! Took {elapsed}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
