"""
🎯 ADAPTIVE CONFIDENCE THRESHOLD SYSTEM
========================================

Dynamically adjusts signal confidence thresholds based on:
- ML model training state (cold/warm/production)
- Recent performance metrics
- Market conditions
- Symbol-specific win rates

This solves the problem of fixed thresholds that are either:
- Too high → No signals, no learning
- Too low → Too many bad trades

"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AdaptiveThresholdManager:
    """
    Manages dynamic confidence thresholds for trading signals.

    Adjusts thresholds based on ML state and performance.
    """

    def __init__(self, config=None):
        self.config = config

        # Default thresholds for each phase
        self.COLD_START_THRESHOLD = 0.6   # Gather data phase
        self.WARM_START_THRESHOLD = 0.75  # Learning phase
        self.PRODUCTION_THRESHOLD = 0.9   # Optimized phase

        # Phase transitions
        self.COLD_TO_WARM_SAMPLES = 50
        self.WARM_TO_PRODUCTION_SAMPLES = 200

        # Performance tracking (per symbol)
        self.symbol_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0.0,
            'recent_trades': deque(maxlen=20)  # Last 20 trades
        })

        # Global ML state
        self.total_ml_samples = 0
        self.current_phase = "cold_start"

        logger.info("🎯 [ADAPTIVE_THRESHOLD] System initialized")

    def update_ml_state(self, total_samples: int, models_fitted: int):
        """
        Update ML training state.

        Args:
            total_samples: Total samples across all models
            models_fitted: Number of fitted models
        """
        self.total_ml_samples = total_samples

        # Determine phase
        if total_samples < self.COLD_TO_WARM_SAMPLES:
            new_phase = "cold_start"
        elif total_samples < self.WARM_TO_PRODUCTION_SAMPLES:
            new_phase = "warm_start"
        else:
            new_phase = "production"

        if new_phase != self.current_phase:
            logger.info(
                f"🎯 [PHASE_CHANGE] {self.current_phase.upper()} → {new_phase.upper()} "
                f"(samples: {total_samples}, fitted: {models_fitted})"
            )
            self.current_phase = new_phase

    def get_threshold(
        self,
        symbol: str,
        signal_strength: float,
        ml_confidence: Optional[float] = None
    ) -> float:
        """
        Get adaptive threshold for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            signal_strength: Raw signal strength from IMBA
            ml_confidence: ML model confidence (0-1)

        Returns:
            Adjusted confidence threshold
        """
        # Base threshold from current phase
        if self.current_phase == "cold_start":
            base_threshold = self.COLD_START_THRESHOLD
        elif self.current_phase == "warm_start":
            base_threshold = self.WARM_START_THRESHOLD
        else:
            base_threshold = self.PRODUCTION_THRESHOLD

        # Adjust based on symbol performance
        perf = self.symbol_performance[symbol]

        if perf['trades'] >= 10:  # Enough data
            win_rate = perf['wins'] / perf['trades']

            # Good performance → lower threshold (take more trades)
            if win_rate > 0.65:
                adjustment = -0.1
            elif win_rate > 0.55:
                adjustment = -0.05
            # Bad performance → raise threshold (be selective)
            elif win_rate < 0.45:
                adjustment = +0.15
            elif win_rate < 0.50:
                adjustment = +0.1
            else:
                adjustment = 0.0

            adjusted_threshold = base_threshold + adjustment

            logger.debug(
                f"🎯 [THRESHOLD] {symbol}: base={base_threshold:.2f}, "
                f"win_rate={win_rate:.1%}, adj={adjustment:+.2f}, "
                f"final={adjusted_threshold:.2f}"
            )

        else:
            adjusted_threshold = base_threshold
            logger.debug(
                f"🎯 [THRESHOLD] {symbol}: base={base_threshold:.2f} "
                f"(insufficient data: {perf['trades']} trades)"
            )

        # Apply hard limits
        adjusted_threshold = max(0.5, min(1.5, adjusted_threshold))

        return adjusted_threshold

    def record_trade_result(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        signal_strength: float
    ):
        """
        Record trade result for threshold adaptation.

        Args:
            symbol: Trading pair
            pnl: Profit/loss in USD
            pnl_pct: Profit/loss percentage
            signal_strength: Original signal strength
        """
        perf = self.symbol_performance[symbol]

        # Update counters
        perf['trades'] += 1
        if pnl > 0:
            perf['wins'] += 1
        perf['total_pnl'] += pnl

        # Store recent trade
        perf['recent_trades'].append({
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'signal_strength': signal_strength,
            'timestamp': datetime.now()
        })

        # Calculate recent win rate
        recent_wins = sum(1 for t in perf['recent_trades'] if t['pnl'] > 0)
        recent_wr = recent_wins / len(perf['recent_trades']) if perf['recent_trades'] else 0.5

        logger.info(
            f"🎯 [TRADE_RESULT] {symbol}: PnL={pnl:+.2f} ({pnl_pct:+.2%}), "
            f"Total: {perf['trades']} trades, WR={recent_wr:.1%} (recent 20)"
        )

    def get_phase_info(self) -> Dict:
        """Get current phase information."""
        return {
            'phase': self.current_phase,
            'total_samples': self.total_ml_samples,
            'base_threshold': self.get_base_threshold(),
            'next_phase_samples': self._get_next_phase_target(),
            'symbols_tracked': len(self.symbol_performance)
        }

    def get_base_threshold(self) -> float:
        """Get current base threshold for the phase."""
        if self.current_phase == "cold_start":
            return self.COLD_START_THRESHOLD
        elif self.current_phase == "warm_start":
            return self.WARM_START_THRESHOLD
        else:
            return self.PRODUCTION_THRESHOLD

    def _get_next_phase_target(self) -> int:
        """Get sample target for next phase."""
        if self.current_phase == "cold_start":
            return self.COLD_TO_WARM_SAMPLES
        elif self.current_phase == "warm_start":
            return self.WARM_TO_PRODUCTION_SAMPLES
        else:
            return self.WARM_TO_PRODUCTION_SAMPLES  # Already at max

    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get performance statistics for a symbol."""
        perf = self.symbol_performance[symbol]

        if perf['trades'] == 0:
            return {
                'trades': 0,
                'win_rate': 0.5,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }

        win_rate = perf['wins'] / perf['trades']
        avg_pnl = perf['total_pnl'] / perf['trades']

        return {
            'trades': perf['trades'],
            'win_rate': win_rate,
            'total_pnl': perf['total_pnl'],
            'avg_pnl': avg_pnl,
            'recent_count': len(perf['recent_trades'])
        }

    def reset_symbol_stats(self, symbol: str):
        """Reset statistics for a symbol (use with caution)."""
        if symbol in self.symbol_performance:
            del self.symbol_performance[symbol]
            logger.warning(f"🎯 [RESET] Statistics reset for {symbol}")


# Global instance
_adaptive_threshold_manager = None


def get_adaptive_threshold_manager(config=None) -> AdaptiveThresholdManager:
    """Get or create global adaptive threshold manager."""
    global _adaptive_threshold_manager

    if _adaptive_threshold_manager is None:
        _adaptive_threshold_manager = AdaptiveThresholdManager(config)

    return _adaptive_threshold_manager
