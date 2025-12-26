"""
Trailing Stop Loss Manager with Take Profit Integration.

SOFTWARE STOP LOSS IMPLEMENTATION (December 2025):
Binance Futures no longer supports STOP_MARKET or STOP orders on the standard
/fapi/v1/order endpoint - they require the Algo Order API.
This manager now implements SOFTWARE stop loss monitoring:
- Monitors mark price in real-time
- When SL price is hit, closes position with MARKET order
- No exchange-side stop orders needed

Features:
- Monitors TP order fills and adjusts SL level
- SOFTWARE stop loss - monitors price and executes MARKET order when hit
- Prevents spam with cooldown logic
- Supports both LONG and SHORT positions
"""

import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

from core.types import Position
from .client import BinanceClient
from .orders import OrderManager
from .precision import adjust_price

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop behavior."""
    # Move SL after nth TP fill (1-indexed)
    move_sl_after_tp: List[int] = None  # [1, 2] means move after 1st and 2nd TP
    
    # SL adjustment rules per TP level
    # 1st TP: Move to 75% between entry and current price (AGGRESSIVE profit protection)
    # 2nd TP: Move to break-even (entry price)
    # 3rd TP: Keep current or close
    sl_adjustments: Dict[int, str] = None  # {1: "75%", 2: "breakeven", 3: "keep"}

    # Cooldown between SL updates (seconds)
    update_cooldown: float = 5.0

    # Safety buffer (percentage) to avoid immediate SL trigger
    safety_buffer_pct: float = 0.1  # 0.1% buffer

    def __post_init__(self):
        if self.move_sl_after_tp is None:
            self.move_sl_after_tp = [1, 2]  # Default: move after 1st and 2nd TP

        if self.sl_adjustments is None:
            self.sl_adjustments = {
                1: "75%",       # After 1st TP: move to 75% protection (FIXED: more aggressive)
                2: "breakeven",  # After 2nd TP: move to entry (break-even)
                3: "keep"        # After 3rd TP: keep current SL
            }


class TrailingStopManager:
    """
    Manages trailing stop loss based on take profit fills.
    
    Workflow:
    1. Position opened with entry price and 3 TP levels
    2. Monitor TP order fills via exchange API
    3. When TP fills detected → calculate new SL level
    4. Cancel old SL order
    5. Place new SL order at adjusted level
    6. Update internal state
    """
    
    def __init__(self, client: BinanceClient, order_manager: OrderManager, 
                 config: Optional[TrailingStopConfig] = None):
        """
        Initialize trailing stop manager.
        
        Args:
            client: Binance client for API calls
            order_manager: Order manager for placing/canceling orders
            config: Configuration for trailing behavior
        """
        self.client = client
        self.order_manager = order_manager
        self.config = config or TrailingStopConfig()
        
        # Track state per symbol
        # Format: {symbol: {"entry": float, "side": str, "tp_filled": int, "last_sl": float, "last_update": float}}
        self._positions: Dict[str, dict] = {}
        
        # Cache filled TP orders to avoid re-processing
        self._filled_tps: Dict[str, set] = {}  # {symbol: {order_id1, order_id2, ...}}
        
        logger.info("TrailingStopManager initialized")
        logger.info(f"  Move SL after TP levels: {self.config.move_sl_after_tp}")
        logger.info(f"  SL adjustments: {self.config.sl_adjustments}")
    
    def register_position(self, symbol: str, entry_price: float, side: str,
                          tp_levels: List[float], initial_sl: float) -> None:
        """
        Register a new position for trailing stop management.

        Args:
            symbol: Trading pair
            entry_price: Entry price of position
            side: Position side ("BUY" or "SELL")
            tp_levels: List of take profit prices
            initial_sl: Initial stop loss price
        """
        # Check if position already exists to preserve tp_filled state
        existing_tp_filled = 0
        existing_last_update = time.time()
        if symbol in self._positions:
            existing_tp_filled = self._positions[symbol].get("tp_filled", 0)
            existing_last_update = self._positions[symbol].get("last_update", time.time())
            logger.debug(f"[TRAIL_SL] {symbol}: Re-registering position, preserving tp_filled={existing_tp_filled}")

        self._positions[symbol] = {
            "entry": entry_price,
            "side": side.upper(),
            "tp_levels": tp_levels,
            "tp_filled": existing_tp_filled,  # Preserve existing state or start at 0
            "last_sl": initial_sl,
            "last_update": existing_last_update,  # Preserve last update time
            "total_tps": len(tp_levels)
        }

        # Only reset filled TP cache for completely new positions
        if existing_tp_filled == 0 and symbol not in self._filled_tps:
            self._filled_tps[symbol] = set()

        logger.info(f"[TRAIL_SL] Registered position for {symbol}: "
                   f"entry={entry_price:.4f}, side={side}, "
                   f"TPs={len(tp_levels)}, initial_sl={initial_sl:.4f}, tp_filled={existing_tp_filled}")
    
    def unregister_position(self, symbol: str) -> None:
        """Remove position from tracking (when closed)."""
        if symbol in self._positions:
            del self._positions[symbol]
        if symbol in self._filled_tps:
            del self._filled_tps[symbol]
        logger.info(f"[TRAIL_SL] Unregistered position for {symbol}")
    
    async def check_and_update(self, symbol: str) -> bool:
        """
        Check if TP orders filled and update SL if needed.
        
        Args:
            symbol: Trading pair to check
        
        Returns:
            True if SL was updated, False otherwise
        """
        if symbol not in self._positions:
            return False
        
        pos_data = self._positions[symbol]
        
        # Check cooldown
        now = time.time()
        if now - pos_data["last_update"] < self.config.update_cooldown:
            return False
        
        # Fetch open and filled orders from exchange
        try:
            filled_count = await self._count_filled_tps(symbol)
            
            if filled_count > pos_data["tp_filled"]:
                logger.info(f"[TRAIL_SL] {symbol}: Detected {filled_count} TP fills "
                           f"(was {pos_data['tp_filled']})")

                # ALWAYS update tp_filled to prevent repeated detection
                pos_data["tp_filled"] = filled_count

                # Update SL based on filled count
                updated = await self._update_stop_loss(symbol, filled_count)

                if updated:
                    pos_data["last_update"] = now
                    return True
                else:
                    # Even if SL wasn't updated, mark as checked to avoid spam
                    pos_data["last_update"] = now
            
        except Exception as e:
            logger.error(f"[TRAIL_SL] Failed to check/update {symbol}: {e}")
        
        return False
    
    async def _count_filled_tps(self, symbol: str) -> int:
        """
        Count how many TP orders have been filled.
        
        Returns:
            Number of filled TP orders
        """
        # CRITICAL FIX: Try to get TP orders from order manager, fallback to simplified logic
        tp_orders = []
        
        # Try order manager first
        if hasattr(self.order_manager, '_exit_orders'):
            exit_orders = self.order_manager._exit_orders.get(symbol, {})
            tp_orders = exit_orders.get("take_profits", [])
        
        # Fallback: If no TP orders from order manager, use simplified logic 
        if not tp_orders:
            logger.debug(f"[TRAIL_SL] {symbol}: No TP orders from order manager, using fallback logic")
            
            # Get position data and count TP orders from open orders
            pos_data = self._positions.get(symbol, {})
            total_expected_tps = len(pos_data.get("tp_levels", []))
            
            if total_expected_tps == 0:
                return 0
            
            try:
                # Count remaining TP orders (reduce-only LIMIT orders)
                open_orders = self.client.get_open_orders(symbol)
                remaining_tp_orders = len([
                    order for order in open_orders 
                    if (order.get('type') == 'LIMIT' and 
                        order.get('reduceOnly', False) and
                        order.get('side') != pos_data.get('side', 'BUY'))  # opposite side
                ])
                
                # Filled TPs = Total expected - Still open
                filled_count = max(0, total_expected_tps - remaining_tp_orders)
                
                logger.debug(f"[TRAIL_SL] {symbol}: Fallback count - Expected TPs: {total_expected_tps}, "
                           f"Remaining: {remaining_tp_orders}, Filled: {filled_count}")
                
                return filled_count
                
            except Exception as e:
                logger.warning(f"[TRAIL_SL] {symbol}: Fallback TP counting failed: {e}")
                return 0
        
        filled_count = 0
        
        # Check each TP order status
        for tp in tp_orders:
            order_id = tp["order_id"]
            
            # Skip if already counted
            if order_id in self._filled_tps.get(symbol, set()):
                filled_count += 1
                continue
            
            # CRITICAL FIX: Check if order is still open (if not, it's likely filled)
            try:
                open_orders = self.client.get_open_orders(symbol)
                order_still_open = any(order.get('orderId') == order_id for order in open_orders)
                
                if not order_still_open:
                    # Order not in open orders - likely filled
                    self._filled_tps.setdefault(symbol, set()).add(order_id)
                    filled_count += 1
                    logger.info(f"🎯 [TRAIL_SL] {symbol}: TP level {tp['level']} FILLED "
                               f"@ {tp['price']:.4f} (order {order_id})")
                else:
                    logger.debug(f"[TRAIL_SL] {symbol}: TP order {order_id} still open")
            
            except Exception as e:
                logger.warning(f"[TRAIL_SL] Failed to check order {order_id} for {symbol}: {e}")
        
        return filled_count
    
    async def _update_stop_loss(self, symbol: str, tp_filled_count: int) -> bool:
        """
        Update stop loss based on TP fill count.
        
        Args:
            symbol: Trading pair
            tp_filled_count: Number of TPs that have filled
        
        Returns:
            True if SL was updated successfully
        """
        if tp_filled_count not in self.config.move_sl_after_tp:
            logger.debug(f"[TRAIL_SL] {symbol}: No SL adjustment needed for TP count {tp_filled_count}")
            return False
        
        pos_data = self._positions[symbol]
        entry = pos_data["entry"]
        side = pos_data["side"]
        old_sl = pos_data["last_sl"]
        
        # Get current market price
        try:
            ticker = self.client.get_ticker_price(symbol)
            current_price = float(ticker.get("price", entry))
        except Exception as e:
            logger.warning(f"[TRAIL_SL] Failed to get current price for {symbol}, using entry: {e}")
            current_price = entry
        
        # Calculate new SL based on adjustment rule
        adjustment_rule = self.config.sl_adjustments.get(tp_filled_count, "keep")
        
        if adjustment_rule == "keep":
            logger.info(f"[TRAIL_SL] {symbol}: Keeping current SL (rule: keep)")
            return False
        
        new_sl = self._calculate_new_sl(entry, current_price, side, adjustment_rule)
        
        if new_sl is None:
            logger.warning(f"[TRAIL_SL] {symbol}: Failed to calculate new SL")
            return False
        
        # Add safety buffer to avoid immediate trigger
        new_sl = self._apply_safety_buffer(new_sl, side)
        
        # Validate new SL is better than old SL
        if not self._is_better_sl(old_sl, new_sl, side):
            logger.warning(f"[TRAIL_SL] {symbol}: New SL {new_sl:.4f} is not better than old {old_sl:.4f}")
            return False
        
        # Update SL: Try Algo API first, fallback to software monitoring
        try:
            logger.info(f"[TRAIL_SL] {symbol}: Updating SL: {old_sl:.4f} → {new_sl:.4f} "
                       f"(after {tp_filled_count} TP fills, rule: {adjustment_rule})")

            # Cancel existing SL orders (both regular and algo)
            await self._cancel_existing_sl_orders(symbol)

            # Place new SL via Algo API
            close_side = "SELL" if side == "BUY" else "BUY"
            await self._place_algo_stop_loss(symbol, close_side, new_sl)

            # Update internal state for software monitoring backup
            pos_data["last_sl"] = new_sl

            logger.info(f"✅ [TRAIL_SL] {symbol}: SL updated to {new_sl:.4f}")
            return True

        except Exception as e:
            logger.error(f"[TRAIL_SL] {symbol}: Failed to update SL: {e}")
            # At least update internal state for software monitoring
            pos_data["last_sl"] = new_sl
            return False
    
    def _calculate_new_sl(self, entry: float, current: float, side: str, rule: str) -> Optional[float]:
        """
        Calculate new SL price based on adjustment rule.

        Args:
            entry: Entry price
            current: Current market price
            side: Position side ("BUY" or "SELL")
            rule: Adjustment rule ("50%", "75%", "breakeven", "close")

        Returns:
            New SL price or None if invalid
        """
        if rule == "breakeven":
            # Move to entry price (break-even)
            return entry

        elif rule == "50%":
            # Move to 50% between entry and current price (50% of profit protected)
            if side == "BUY":
                # Long: SL moves up to protect 50% of gains
                # FIXED: Was 0.5 * 0.5 = 25%, now correctly 50%
                return entry + (current - entry) * 0.5
            else:
                # Short: SL moves down to protect 50% of gains
                return entry - (entry - current) * 0.5

        elif rule == "75%":
            # Move to 75% between entry and current price (75% of profit protected)
            if side == "BUY":
                # Long: SL moves up to protect 75% of gains (more aggressive)
                return entry + (current - entry) * 0.75
            else:
                # Short: SL moves down to protect 75% of gains
                return entry - (entry - current) * 0.75

        elif rule == "close":
            # Close position (return None to indicate close)
            return None

        else:
            logger.warning(f"Unknown SL adjustment rule: {rule}")
            return None
    
    def _apply_safety_buffer(self, price: float, side: str) -> float:
        """
        Apply safety buffer to SL price to avoid immediate trigger.
        
        Args:
            price: Original SL price
            side: Position side
        
        Returns:
            Adjusted SL price with buffer
        """
        buffer = self.config.safety_buffer_pct / 100.0
        
        if side == "BUY":
            # Long: reduce SL slightly (move down)
            return price * (1 - buffer)
        else:
            # Short: increase SL slightly (move up)
            return price * (1 + buffer)
    
    def _is_better_sl(self, old_sl: float, new_sl: float, side: str) -> bool:
        """
        Check if new SL is better (tighter) than old SL.
        
        Args:
            old_sl: Current stop loss price
            new_sl: Proposed new stop loss price
            side: Position side
        
        Returns:
            True if new SL is better (tighter protection)
        """
        if side == "BUY":
            # Long: better SL is higher (closer to current price)
            return new_sl > old_sl
        else:
            # Short: better SL is lower
            return new_sl < old_sl
    
    async def monitor_all_positions(self) -> None:
        """Monitor all registered positions and update SL if needed."""
        for symbol in list(self._positions.keys()):
            try:
                # CRITICAL: Check if stop loss should trigger (SOFTWARE SL)
                await self._check_software_stop_loss(symbol)

                # Then check for TP fills and update SL accordingly
                await self.check_and_update(symbol)
            except Exception as e:
                logger.error(f"[TRAIL_SL] Error monitoring {symbol}: {e}")

    async def _check_software_stop_loss(self, symbol: str) -> bool:
        """
        SOFTWARE STOP LOSS: Check if current price hit stop loss level.

        Since Binance no longer supports STOP orders on standard endpoint,
        we monitor price ourselves and close with MARKET order when SL hit.

        Returns:
            True if SL was triggered and position closed, False otherwise
        """
        if symbol not in self._positions:
            return False

        pos_data = self._positions[symbol]
        side = pos_data["side"]
        last_sl = pos_data.get("last_sl", 0)

        if last_sl <= 0:
            logger.warning(f"[SOFT_SL] {symbol}: No SL price registered")
            return False

        try:
            # Get current mark price
            mark_price = float(self.client.get_mark_price(symbol))

            if mark_price <= 0:
                return False

            # Check if SL triggered
            sl_triggered = False

            if side == "BUY":  # Long position - SL triggers when price falls below SL
                if mark_price <= last_sl:
                    sl_triggered = True
                    logger.warning(f"🛑 [SOFT_SL] {symbol} LONG: Mark {mark_price:.4f} <= SL {last_sl:.4f} - TRIGGERED!")
            else:  # Short position - SL triggers when price rises above SL
                if mark_price >= last_sl:
                    sl_triggered = True
                    logger.warning(f"🛑 [SOFT_SL] {symbol} SHORT: Mark {mark_price:.4f} >= SL {last_sl:.4f} - TRIGGERED!")

            if sl_triggered:
                # Close position with MARKET order
                await self._close_position_market(symbol, side)
                return True

            return False

        except Exception as e:
            logger.error(f"[SOFT_SL] {symbol}: Failed to check SL: {e}")
            return False

    async def _close_position_market(self, symbol: str, position_side: str) -> bool:
        """
        Close position immediately with MARKET order.

        Args:
            symbol: Trading pair
            position_side: Current position side ("BUY" for long, "SELL" for short)

        Returns:
            True if position closed successfully
        """
        try:
            # Get position quantity
            positions = self.client.get_positions()
            position_qty = 0.0
            for pos in positions:
                if pos.get('symbol') == symbol.upper():
                    position_qty = abs(float(pos.get('positionAmt', 0)))
                    break

            if position_qty <= 0:
                logger.warning(f"[SOFT_SL] {symbol}: No position found to close")
                self.unregister_position(symbol)
                return False

            # Determine close side (opposite of position)
            close_side = "SELL" if position_side == "BUY" else "BUY"

            logger.info(f"🛑 [SOFT_SL] {symbol}: Closing position with MARKET order: {close_side} {position_qty}")

            # Place market order to close
            order_result = self.client.place_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=position_qty,
                reduceOnly="true"
            )

            order_id = order_result.get('orderId', 'N/A')
            logger.info(f"✅ [SOFT_SL] {symbol}: Position closed! Order ID: {order_id}")

            # Unregister position
            self.unregister_position(symbol)

            return True

        except Exception as e:
            logger.error(f"[SOFT_SL] {symbol}: Failed to close position: {e}")
            return False
    
    async def _cancel_existing_sl_orders(self, symbol: str) -> None:
        """Cancel existing stop loss orders (both regular and algo orders)."""
        try:
            # Cancel regular stop orders
            open_orders = self.client.get_open_orders(symbol)
            for order in open_orders:
                order_type = order.get('type', '').upper()
                if order_type in ('STOP_MARKET', 'STOP', 'STOP_LOSS_LIMIT'):
                    try:
                        self.client.cancel_order(symbol=symbol, orderId=order['orderId'])
                        logger.info(f"[TRAIL_SL] Cancelled regular SL order {order['orderId']} for {symbol}")
                    except Exception as e:
                        logger.debug(f"[TRAIL_SL] Failed to cancel order {order['orderId']}: {e}")

            # Cancel algo stop orders
            try:
                algo_orders = self.client.get_algo_orders(symbol=symbol)
                for algo in algo_orders:
                    if algo.get('algoStatus') == 'NEW' and algo.get('type') in ('STOP_MARKET', 'STOP'):
                        try:
                            self.client.cancel_algo_order(symbol=symbol, algoId=algo['algoId'])
                            logger.info(f"[TRAIL_SL] Cancelled algo SL order {algo['algoId']} for {symbol}")
                        except Exception as e:
                            logger.debug(f"[TRAIL_SL] Failed to cancel algo order {algo['algoId']}: {e}")
            except Exception as e:
                logger.debug(f"[TRAIL_SL] Failed to get/cancel algo orders for {symbol}: {e}")

        except Exception as e:
            logger.warning(f"[TRAIL_SL] Failed to cancel existing SL orders for {symbol}: {e}")

    async def _place_algo_stop_loss(self, symbol: str, side: str, stop_price: float) -> None:
        """Place new stop loss order via Algo Order API."""
        try:
            rounded_stop_price = adjust_price(symbol, stop_price)

            # Place STOP_MARKET via Algo API with closePosition=true
            result = self.client.place_algo_order(
                algoType="CONDITIONAL",
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                triggerPrice=str(rounded_stop_price),
                closePosition="true",
                workingType="MARK_PRICE"
            )

            algo_id = result.get('algoId', 'N/A')
            logger.info(f"🎯 [TRAIL_SL] Placed new SL via Algo API for {symbol}: {side} @ {rounded_stop_price:.4f} (algoId={algo_id})")

        except Exception as e:
            logger.error(f"[TRAIL_SL] Failed to place algo SL order for {symbol}: {e}")
            raise

    def get_status(self, symbol: str) -> Optional[dict]:
        """Get current trailing stop status for a symbol."""
        return self._positions.get(symbol)

    async def sync_positions_from_exchange(self) -> None:
        """
        Sync positions from exchange and ensure all have stop losses.

        CRITICAL: Call this on startup and periodically to catch positions
        without registered trailing stops.
        """
        try:
            positions = self.client.get_positions()
            synced = 0

            for pos in positions:
                symbol = pos.get('symbol', '')
                position_amt = float(pos.get('positionAmt', 0))

                if position_amt == 0:
                    continue  # No position

                # Check if already registered
                if symbol in self._positions:
                    continue  # Already tracking

                # New position found - register it
                entry_price = float(pos.get('entryPrice', 0))
                side = "BUY" if position_amt > 0 else "SELL"

                if entry_price <= 0:
                    continue

                # Calculate default SL (2% from entry)
                sl_distance = 0.02
                if side == "BUY":
                    default_sl = entry_price * (1 - sl_distance)
                else:
                    default_sl = entry_price * (1 + sl_distance)

                logger.warning(f"⚠️ [TRAIL_SL] {symbol}: Found untracked position! "
                              f"Side={side}, Entry={entry_price:.4f}, Registering with SL={default_sl:.4f}")

                self.register_position(
                    symbol=symbol,
                    entry_price=entry_price,
                    side=side,
                    tp_levels=[],  # Unknown TP levels
                    initial_sl=default_sl
                )
                synced += 1

            if synced > 0:
                logger.info(f"[TRAIL_SL] Synced {synced} positions from exchange")

        except Exception as e:
            logger.error(f"[TRAIL_SL] Failed to sync positions from exchange: {e}")
