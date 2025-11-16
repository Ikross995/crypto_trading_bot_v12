"""
Trailing Stop Loss Manager with Take Profit Integration.

Automatically adjusts stop loss after each take profit fill:
- After 1st TP: Move SL to 50% between entry and current price (partial protection)
- After 2nd TP: Move SL to break-even (entry price)
- After 3rd TP: Keep trailing or close remaining position

Features:
- Monitors TP order fills in real-time
- Cancels old SL before placing new one
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
                # ALWAYS update tp_filled to prevent repeated detection
                old_filled = pos_data["tp_filled"]
                pos_data["tp_filled"] = filled_count

                # Update SL based on filled count
                updated = await self._update_stop_loss(symbol, filled_count)

                if updated:
                    logger.info(f"✅ [TRAIL_SL] {symbol}: TP{filled_count} filled → SL adjusted")
                    pos_data["last_update"] = now
                    return True
                else:
                    # Log only if this is actual progress (not no-SL-adjustment case)
                    if filled_count < 3:  # Don't log when all TPs done
                        logger.debug(f"[TRAIL_SL] {symbol}: TP{filled_count} filled (no SL adjustment needed)")
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
            # Reduced log level to avoid spam - only log once per symbol
            if not hasattr(self, '_fallback_logged'):
                self._fallback_logged = set()
            if symbol not in self._fallback_logged:
                logger.debug(f"[TRAIL_SL] {symbol}: Using fallback TP counting logic")
                self._fallback_logged.add(symbol)
            
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

                # Only log if count changed or first time
                if not hasattr(self, '_last_filled_count'):
                    self._last_filled_count = {}
                if self._last_filled_count.get(symbol, -1) != filled_count:
                    logger.debug(f"[TRAIL_SL] {symbol}: TP status - Expected: {total_expected_tps}, "
                               f"Remaining: {remaining_tp_orders}, Filled: {filled_count}")
                    self._last_filled_count[symbol] = filled_count
                
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
        
        # Cancel old SL and place new one
        try:
            logger.info(f"[TRAIL_SL] {symbol}: Updating SL: {old_sl:.4f} → {new_sl:.4f} "
                       f"(after {tp_filled_count} TP fills, rule: {adjustment_rule})")
            
            # CRITICAL FIX: Cancel existing SL orders directly via client
            await self._cancel_existing_sl_orders(symbol)
            
            # Place new SL order directly via client
            close_side = "SELL" if side == "BUY" else "BUY"
            await self._place_stop_loss_order(symbol, close_side, new_sl)
            
            # Update internal state
            pos_data["last_sl"] = new_sl
            
            logger.info(f"[TRAIL_SL] {symbol}: SL updated successfully to {new_sl:.4f}")
            return True
        
        except Exception as e:
            logger.error(f"[TRAIL_SL] {symbol}: Failed to update SL: {e}")
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
                await self.check_and_update(symbol)
            except Exception as e:
                logger.error(f"[TRAIL_SL] Error monitoring {symbol}: {e}")
    
    async def _cancel_existing_sl_orders(self, symbol: str) -> None:
        """Cancel existing stop loss orders for symbol."""
        try:
            open_orders = self.client.get_open_orders(symbol)
            
            for order in open_orders:
                order_type = order.get('type', '').upper()
                is_sl_order = (
                    order_type in ('STOP_MARKET', 'STOP', 'STOP_LOSS_LIMIT') or
                    order.get('closePosition', False)
                )
                
                if is_sl_order:
                    try:
                        self.client.cancel_order(symbol=symbol, orderId=order['orderId'])
                        logger.info(f"🎯 [TRAIL_SL] Cancelled existing SL order {order['orderId']} for {symbol}")
                    except Exception as e:
                        logger.warning(f"[TRAIL_SL] Failed to cancel SL order {order['orderId']}: {e}")
        
        except Exception as e:
            logger.warning(f"[TRAIL_SL] Failed to cancel existing SL orders for {symbol}: {e}")
    
    async def _place_stop_loss_order(self, symbol: str, side: str, stop_price: float) -> None:
        """Place new stop loss order."""
        try:
            # Round stop price to proper precision using exchange filters
            rounded_stop_price = adjust_price(symbol, stop_price)

            # Place STOP_MARKET order with closePosition=true
            order_result = self.client.place_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=rounded_stop_price,
                closePosition="true",  # Close entire position
                workingType="MARK_PRICE"  # Use mark price to avoid manipulation
            )

            logger.info(f"🎯 [TRAIL_SL] Placed new SL order for {symbol}: {side} @ {rounded_stop_price:.4f} "
                       f"(order: {order_result.get('orderId', 'N/A')})")

        except Exception as e:
            logger.error(f"[TRAIL_SL] Failed to place SL order for {symbol}: {e}")
            raise
    
    def get_status(self, symbol: str) -> Optional[dict]:
        """Get current trailing stop status for a symbol."""
        return self._positions.get(symbol)
