# 🔒 Concurrency Safety Integration Example

## How to Integrate SafeTradingState into Existing Code

### Example 1: Protecting Balance Updates in exchange/client.py

**Before (NOT SAFE):**
```python
class ExchangeClient:
    def __init__(self):
        self.balance = Decimal('0')

    async def execute_trade(self, symbol, amount, price):
        cost = amount * price

        # ❌ RACE CONDITION: Another coroutine could modify balance here!
        if self.balance >= cost:
            self.balance -= cost
            # Execute trade
            return True
        return False
```

**After (SAFE):**
```python
from utils.concurrency import get_global_safe_state

class ExchangeClient:
    def __init__(self):
        self.balance = Decimal('0')
        self.safe_state = get_global_safe_state()

    async def execute_trade(self, symbol, amount, price):
        cost = amount * price

        # ✅ SAFE: Lock ensures atomic check-and-update
        async with self.safe_state.atomic_balance_update():
            if self.balance >= cost:
                self.balance -= cost
                # Execute trade
                return True
            return False
```

### Example 2: Protecting Position Updates

**Before (NOT SAFE):**
```python
class PositionManager:
    def __init__(self):
        self.positions = {}  # symbol -> Position

    async def add_to_position(self, symbol, quantity, price):
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        # ❌ RACE CONDITION!
        position = self.positions[symbol]
        old_size = position.size
        old_price = position.entry_price

        # Another coroutine could modify position here!
        position.size = old_size + quantity
        position.entry_price = (old_size * old_price + quantity * price) / (old_size + quantity)
```

**After (SAFE):**
```python
from utils.concurrency import get_global_safe_state

class PositionManager:
    def __init__(self):
        self.positions = {}
        self.safe_state = get_global_safe_state()

    async def add_to_position(self, symbol, quantity, price):
        # ✅ SAFE: Lock ensures atomic update
        async with self.safe_state.atomic_position_update(symbol):
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol)

            position = self.positions[symbol]
            old_size = position.size
            old_price = position.entry_price

            position.size = old_size + quantity
            position.entry_price = (old_size * old_price + quantity * price) / (old_size + quantity)
```

### Example 3: Protecting Order State Updates

**Before (NOT SAFE):**
```python
class OrderManager:
    def __init__(self):
        self.orders = {}  # order_id -> Order

    async def update_order_status(self, order_id, new_status, filled_qty=None):
        # ❌ RACE CONDITION!
        order = self.orders[order_id]
        order.status = new_status

        if filled_qty:
            order.filled_qty = filled_qty
```

**After (SAFE):**
```python
from utils.concurrency import get_global_safe_state

class OrderManager:
    def __init__(self):
        self.orders = {}
        self.safe_state = get_global_safe_state()

    async def update_order_status(self, order_id, new_status, filled_qty=None):
        # ✅ SAFE: Lock ensures atomic update
        async with self.safe_state.atomic_order_update(order_id):
            order = self.orders[order_id]
            order.status = new_status

            if filled_qty:
                order.filled_qty = filled_qty
```

### Example 4: Complete Trading Engine Integration

```python
from utils.concurrency import get_global_safe_state
from decimal import Decimal

class TradingEngine:
    def __init__(self):
        self.balance = Decimal('10000')
        self.positions = {}
        self.orders = {}
        self.safe_state = get_global_safe_state()

    async def execute_buy_order(self, symbol, quantity, price):
        """
        Safe buy order execution with proper locking
        """
        cost = Decimal(str(quantity)) * Decimal(str(price))

        # Step 1: Check and deduct balance (atomic)
        async with self.safe_state.atomic_balance_update():
            if self.balance < cost:
                logger.warning(f"Insufficient balance: {self.balance} < {cost}")
                return False

            self.balance -= cost
            logger.info(f"Balance deducted: {cost}. Remaining: {self.balance}")

        # Step 2: Update position (atomic)
        try:
            async with self.safe_state.atomic_position_update(symbol):
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'size': Decimal('0'),
                        'entry_price': Decimal('0')
                    }

                position = self.positions[symbol]
                old_size = position['size']
                old_price = position['entry_price']

                # Calculate new average entry price
                total_cost = old_size * old_price + quantity * price
                position['size'] = old_size + quantity
                position['entry_price'] = total_cost / position['size']

                logger.info(
                    f"Position updated: {symbol} "
                    f"Size: {position['size']} @ {position['entry_price']}"
                )

            return True

        except Exception as e:
            # Rollback balance on error
            async with self.safe_state.atomic_balance_update():
                self.balance += cost
                logger.error(f"Order failed, balance restored: {e}")

            return False

    async def process_concurrent_signals(self, signals):
        """
        Process multiple signals concurrently (safe with locks)
        """
        tasks = [
            self.execute_buy_order(
                signal.symbol,
                signal.quantity,
                signal.price
            )
            for signal in signals
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r is True)
        logger.info(f"Processed {len(signals)} signals: {successful} successful")

        return results

    def get_statistics(self):
        """Get lock usage statistics"""
        return self.safe_state.get_lock_statistics()

    def log_statistics(self):
        """Log lock statistics for monitoring"""
        self.safe_state.log_statistics()
```

### Example 5: Monitoring Lock Performance

```python
import asyncio
from utils.concurrency import get_global_safe_state

async def monitor_locks():
    """Periodic lock statistics monitoring"""
    safe_state = get_global_safe_state()

    while True:
        await asyncio.sleep(300)  # Every 5 minutes

        stats = safe_state.get_lock_statistics()

        # Check for high contention
        for lock_name, lock_stats in stats.items():
            if lock_stats['contention_rate_pct'] > 30:
                logger.warning(
                    f"⚠️ High contention on {lock_name} lock: "
                    f"{lock_stats['contention_rate_pct']:.1f}%"
                )

            if lock_stats['max_wait_time_ms'] > 100:
                logger.warning(
                    f"⚠️ Long wait time on {lock_name} lock: "
                    f"{lock_stats['max_wait_time_ms']:.1f}ms"
                )

        # Log summary
        safe_state.log_statistics()

# Start monitoring in background
asyncio.create_task(monitor_locks())
```

## When to Use Each Lock

### `atomic_balance_update()`
Use when:
- Deducting cost for trades
- Adding profits from closed positions
- Any balance modification

### `atomic_position_update(symbol)`
Use when:
- Opening new position
- Adding to existing position (DCA)
- Closing position
- Updating position entry price

### `atomic_order_update(order_id)`
Use when:
- Creating new order
- Updating order status (filled/cancelled)
- Modifying order quantity/price

### `atomic_trade_record()`
Use when:
- Saving trade to database
- Updating trade statistics
- Recording trade history

## Best Practices

1. **Keep Critical Sections Small**
   ```python
   # ✅ GOOD: Minimal time in lock
   async with safe_state.atomic_balance_update():
       if balance >= cost:
           balance -= cost
           return True
       return False

   # ❌ BAD: Long operation in lock
   async with safe_state.atomic_balance_update():
       if balance >= cost:
           balance -= cost
           await send_notification()  # Slow!
           await update_database()     # Slow!
           return True
   ```

2. **Avoid Nested Locks (Deadlock Risk)**
   ```python
   # ❌ BAD: Nested locks can deadlock
   async with safe_state.atomic_balance_update():
       async with safe_state.atomic_position_update():
           # Deadlock risk!
           pass

   # ✅ GOOD: Sequential locks
   async with safe_state.atomic_balance_update():
       balance -= cost

   async with safe_state.atomic_position_update():
       position.size += quantity
   ```

3. **Handle Exceptions Properly**
   ```python
   async with safe_state.atomic_balance_update():
       try:
           balance -= cost
           # Critical operations
       except Exception as e:
           # Rollback changes
           balance += cost
           raise
   ```

4. **Monitor Lock Statistics**
   ```python
   # Periodically check performance
   stats = safe_state.get_lock_statistics()

   if stats['balance']['contention_rate_pct'] > 50:
       logger.critical("High lock contention - optimize code!")
   ```

## Testing Your Integration

```python
import pytest

@pytest.mark.asyncio
async def test_concurrent_trades():
    """Test that concurrent trades don't corrupt balance"""
    engine = TradingEngine()
    engine.balance = Decimal('1000')

    # Execute 10 concurrent trades of $100 each
    signals = [
        Signal('BTCUSDT', Decimal('0.001'), Decimal('50000'))
        for _ in range(10)
    ]

    results = await engine.process_concurrent_signals(signals)

    # All should succeed, balance should be 0
    assert engine.balance == Decimal('0')
    assert all(results)
```

## Migration Checklist

- [ ] Install utils/concurrency.py
- [ ] Add `from utils.concurrency import get_global_safe_state` to modules
- [ ] Identify all balance modification points
- [ ] Wrap with `async with safe_state.atomic_balance_update()`
- [ ] Identify all position modification points
- [ ] Wrap with `async with safe_state.atomic_position_update(symbol)`
- [ ] Identify all order modification points
- [ ] Wrap with `async with safe_state.atomic_order_update(order_id)`
- [ ] Run tests/test_concurrency.py
- [ ] Monitor lock statistics in production
- [ ] Check for high contention (>30%)
- [ ] Optimize critical sections if needed

## Performance Impact

- **Lock acquisition:** <1ms (negligible)
- **Contention wait:** 0-100ms depending on critical section size
- **Memory:** <1KB per lock
- **CPU:** Minimal overhead

**Bottom line:** The safety benefit far outweighs minimal performance cost!
