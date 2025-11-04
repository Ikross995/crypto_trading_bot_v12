"""
📊 Market Regime Detector
==========================

Intelligent market regime detection using multiple indicators.

Identifies 5 market regimes:
1. STRONG_TREND - Clear directional movement (ADX >25, low volatility)
2. VOLATILE_TREND - Trending but high volatility (ADX >25, high ATR)
3. TIGHT_RANGE - Low movement, low volatility (ADX <20, narrow BB)
4. CHOPPY - Indecisive market (ADX <20, high volatility)
5. TRANSITIONAL - Between regimes

Research показывает: 40-60% reduction in false signals
when strategy adapts to regime!
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    STRONG_TREND = "strong_trend"
    VOLATILE_TREND = "volatile_trend"
    TIGHT_RANGE = "tight_range"
    CHOPPY = "choppy"
    TRANSITIONAL = "transitional"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Metrics used for regime detection"""
    adx: float
    atr_pct: float  # ATR as % of price
    bb_width: float
    volume_ratio: float
    price_momentum: float
    trend_consistency: float


@dataclass
class RegimeInfo:
    """Information about detected regime"""
    regime: MarketRegime
    confidence: float  # 0.0 - 1.0
    metrics: RegimeMetrics
    timestamp: datetime
    duration_candles: int = 0  # How long in this regime


class MarketRegimeDetector:
    """
    Detects market regime using technical indicators.

    Uses combination of:
    - ADX (Average Directional Index) - trend strength
    - ATR (Average True Range) - volatility
    - Bollinger Bands Width - volatility measure
    - Volume - market participation
    - Price momentum - directional strength

    Research-based thresholds (optimized for crypto 2024-2025):
    - STRONG_TREND: ADX >25, ATR% <3%
    - VOLATILE_TREND: ADX >25, ATR% ≥3%
    - TIGHT_RANGE: ADX <20, BB_width <1%
    - CHOPPY: ADX <20, BB_width ≥1%
    """

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_flat_threshold: float = 20.0,
        atr_high_threshold: float = 3.0,  # % of price
        bb_tight_threshold: float = 0.01,  # 1%
        regime_min_duration: int = 5  # Min candles to confirm regime
    ):
        """
        Initialize regime detector.

        Args:
            adx_trend_threshold: ADX above = trending
            adx_flat_threshold: ADX below = ranging
            atr_high_threshold: ATR% above = high volatility
            bb_tight_threshold: BB width below = tight range
            regime_min_duration: Minimum candles to confirm regime
        """
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_flat_threshold = adx_flat_threshold
        self.atr_high_threshold = atr_high_threshold
        self.bb_tight_threshold = bb_tight_threshold
        self.regime_min_duration = regime_min_duration

        # State tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: List[RegimeInfo] = []
        self.regime_start_candle = 0
        self.candle_count = 0

        logger.info(
            f"📊 RegimeDetector initialized: "
            f"ADX trend={adx_trend_threshold}, flat={adx_flat_threshold}"
        )

    async def detect_regime(
        self,
        candles_df: pd.DataFrame,
        adx_column: str = 'adx',
        atr_column: str = 'atr_14',
        bb_upper_column: str = 'bb_upper',
        bb_lower_column: str = 'bb_lower',
        volume_column: str = 'volume'
    ) -> RegimeInfo:
        """
        Detect current market regime.

        Args:
            candles_df: DataFrame with OHLCV and indicators
            adx_column: ADX column name
            atr_column: ATR column name
            bb_upper_column: Bollinger Band upper column
            bb_lower_column: Bollinger Band lower column
            volume_column: Volume column name

        Returns:
            RegimeInfo with detected regime and metrics
        """
        self.candle_count += 1

        # Extract latest values
        latest = candles_df.iloc[-1]
        current_price = latest['close']

        # Calculate metrics
        adx = latest.get(adx_column, 0)
        atr = latest.get(atr_column, 0)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        bb_upper = latest.get(bb_upper_column, current_price * 1.02)
        bb_lower = latest.get(bb_lower_column, current_price * 0.98)
        bb_width = ((bb_upper - bb_lower) / current_price) if current_price > 0 else 0

        # Volume ratio (current vs average)
        volume = latest.get(volume_column, 0)
        avg_volume = candles_df[volume_column].rolling(20).mean().iloc[-1]
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Price momentum (change over last 20 candles)
        price_20_ago = candles_df['close'].iloc[-20] if len(candles_df) >= 20 else current_price
        price_momentum = ((current_price - price_20_ago) / price_20_ago) * 100 if price_20_ago > 0 else 0

        # Trend consistency (how consistent is the trend direction)
        if len(candles_df) >= 20:
            returns = candles_df['close'].pct_change().iloc[-20:]
            trend_consistency = abs(returns.mean() / (returns.std() + 1e-10))
        else:
            trend_consistency = 0

        metrics = RegimeMetrics(
            adx=adx,
            atr_pct=atr_pct,
            bb_width=bb_width,
            volume_ratio=volume_ratio,
            price_momentum=price_momentum,
            trend_consistency=trend_consistency
        )

        # Detect regime
        regime, confidence = self._classify_regime(metrics)

        # Check if regime changed
        if regime != self.current_regime:
            # Require minimum duration for regime confirmation
            if self.candle_count - self.regime_start_candle < self.regime_min_duration:
                # Not enough candles to confirm new regime
                regime = MarketRegime.TRANSITIONAL
                confidence *= 0.5

        # Update regime if changed
        if regime != self.current_regime and regime != MarketRegime.TRANSITIONAL:
            logger.info(
                f"🔄 [REGIME_CHANGE] {self.current_regime.value} → {regime.value} "
                f"(confidence: {confidence:.2f})"
            )

            self.current_regime = regime
            self.regime_start_candle = self.candle_count

        # Calculate duration in current regime
        duration = self.candle_count - self.regime_start_candle

        regime_info = RegimeInfo(
            regime=regime,
            confidence=confidence,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc),
            duration_candles=duration
        )

        # Store in history
        self.regime_history.append(regime_info)

        # Keep only recent history (last 1000 regimes)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        return regime_info

    def _classify_regime(
        self,
        metrics: RegimeMetrics
    ) -> tuple[MarketRegime, float]:
        """
        Classify regime based on metrics.

        Returns:
            (regime, confidence)
        """
        adx = metrics.adx
        atr_pct = metrics.atr_pct
        bb_width = metrics.bb_width

        # Confidence factors
        confidence_factors = []

        # 1. STRONG_TREND: High ADX, Low volatility
        if adx > self.adx_trend_threshold and atr_pct < self.atr_high_threshold:
            # Additional checks for confidence
            if metrics.trend_consistency > 0.5:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.7)

            if metrics.volume_ratio > 1.2:  # Good volume
                confidence_factors.append(0.8)

            confidence = np.mean(confidence_factors) if confidence_factors else 0.7

            return MarketRegime.STRONG_TREND, confidence

        # 2. VOLATILE_TREND: High ADX, High volatility
        elif adx > self.adx_trend_threshold and atr_pct >= self.atr_high_threshold:
            if abs(metrics.price_momentum) > 2:  # Strong momentum despite volatility
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)

            confidence = np.mean(confidence_factors) if confidence_factors else 0.6

            return MarketRegime.VOLATILE_TREND, confidence

        # 3. TIGHT_RANGE: Low ADX, Narrow BB
        elif adx < self.adx_flat_threshold and bb_width < self.bb_tight_threshold:
            if atr_pct < 1.5:  # Very low volatility
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.7)

            if metrics.volume_ratio < 0.8:  # Low volume confirms ranging
                confidence_factors.append(0.8)

            confidence = np.mean(confidence_factors) if confidence_factors else 0.7

            return MarketRegime.TIGHT_RANGE, confidence

        # 4. CHOPPY: Low ADX, Wide BB (volatile but no trend)
        elif adx < self.adx_flat_threshold and bb_width >= self.bb_tight_threshold:
            confidence = 0.8 if atr_pct > self.atr_high_threshold else 0.6

            return MarketRegime.CHOPPY, confidence

        # 5. TRANSITIONAL: Between regimes
        else:
            return MarketRegime.TRANSITIONAL, 0.5

    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about regime history.

        Returns:
            Dictionary with statistics
        """
        if not self.regime_history:
            return {}

        # Count regimes
        regime_counts = {}
        for info in self.regime_history:
            regime_name = info.regime.value
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

        # Calculate percentages
        total = len(self.regime_history)
        regime_percentages = {
            regime: (count / total) * 100
            for regime, count in regime_counts.items()
        }

        # Average confidence by regime
        regime_confidences = {}
        for regime in MarketRegime:
            regime_infos = [info for info in self.regime_history if info.regime == regime]
            if regime_infos:
                avg_confidence = np.mean([info.confidence for info in regime_infos])
                regime_confidences[regime.value] = round(avg_confidence, 3)

        # Current regime duration
        current_duration = self.candle_count - self.regime_start_candle

        return {
            'current_regime': self.current_regime.value,
            'current_duration_candles': current_duration,
            'regime_counts': regime_counts,
            'regime_percentages': {k: round(v, 1) for k, v in regime_percentages.items()},
            'regime_avg_confidences': regime_confidences,
            'total_regimes_detected': len(self.regime_history)
        }

    def log_statistics(self):
        """Log regime statistics"""
        stats = self.get_regime_statistics()

        if not stats:
            logger.info("No regime history available")
            return

        logger.info("=" * 70)
        logger.info("📊 [REGIME_STATS] Market Regime Statistics")
        logger.info("=" * 70)
        logger.info(f"Current Regime: {stats['current_regime'].upper()}")
        logger.info(f"Duration: {stats['current_duration_candles']} candles")
        logger.info("\nRegime Distribution:")

        for regime, pct in stats['regime_percentages'].items():
            count = stats['regime_counts'][regime]
            logger.info(f"  {regime:20s}: {pct:5.1f}% ({count} times)")

        logger.info("\nAverage Confidence:")
        for regime, conf in stats['regime_avg_confidences'].items():
            logger.info(f"  {regime:20s}: {conf:.2f}")

        logger.info("=" * 70)


__all__ = ['MarketRegimeDetector', 'MarketRegime', 'RegimeInfo', 'RegimeMetrics']
