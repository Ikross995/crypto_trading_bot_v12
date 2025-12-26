"""
🚀 EXTENDED IMBA SIGNALS
========================

Additional high-quality signals using new volatility indicators:
13. SuperTrend (trend following + volatility)
14. Keltner Channel Breakout (volatility expansion)
15. Ichimoku Cloud (multi-timeframe confirmation)

These can be integrated into IMBA system by adding to IMBASignalAggregator.

"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

from data.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Signal structure matching IMBA format."""
    name: str
    direction: str  # 'buy', 'sell', 'wait'
    strength: float  # 0.0-1.0

    def to_dict(self):
        return {
            'name': self.name,
            'direction': self.direction,
            'strength': self.strength
        }


class ExtendedIMBASignals:
    """Extended signal collection with volatility indicators."""

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Signal:
        """
        SuperTrend signal - trend following with volatility adaptation.

        Strong signal when trend flips (1.0 strength).
        Medium signal when riding existing trend (0.5-0.7).

        Args:
            df: OHLCV DataFrame
            period: ATR period
            multiplier: ATR multiplier

        Returns:
            Signal object
        """
        try:
            if len(df) < period + 5:
                return Signal("supertrend", "wait", 0.0)

            # Calculate SuperTrend
            st_values, st_trend = TechnicalIndicators.supertrend(
                df['high'], df['low'], df['close'],
                period=period, multiplier=multiplier
            )

            if st_values.isna().all() or st_trend.isna().all():
                return Signal("supertrend", "wait", 0.0)

            # Current and previous trend
            current_trend = st_trend.iloc[-1]
            prev_trend = st_trend.iloc[-2] if len(st_trend) > 1 else current_trend

            current_price = df['close'].iloc[-1]
            st_current = st_values.iloc[-1]

            # Trend flip = strong signal
            if current_trend == 1 and prev_trend == -1:
                # Bullish flip
                distance_from_st = abs(current_price - st_current) / current_price
                strength = min(1.0, 0.9 - distance_from_st * 10)  # Closer = stronger
                return Signal("supertrend", "buy", max(0.7, strength))

            elif current_trend == -1 and prev_trend == 1:
                # Bearish flip
                distance_from_st = abs(current_price - st_current) / current_price
                strength = min(1.0, 0.9 - distance_from_st * 10)
                return Signal("supertrend", "sell", max(0.7, strength))

            # Riding existing trend (weaker signal)
            elif current_trend == 1:
                # In uptrend, price bouncing off SuperTrend
                if current_price > st_current:
                    distance = (current_price - st_current) / st_current
                    if distance < 0.005:  # Very close to ST line
                        return Signal("supertrend", "buy", 0.6)

            elif current_trend == -1:
                # In downtrend, price bouncing off SuperTrend
                if current_price < st_current:
                    distance = (st_current - current_price) / current_price
                    if distance < 0.005:
                        return Signal("supertrend", "sell", 0.6)

            return Signal("supertrend", "wait", 0.0)

        except Exception as e:
            logger.error(f"[SUPERTREND] Error: {e}")
            return Signal("supertrend", "wait", 0.0)

    @staticmethod
    def keltner_breakout(df: pd.DataFrame, kc_period: int = 20, kc_multiplier: float = 2.0) -> Signal:
        """
        Keltner Channel Breakout - volatility expansion signal.

        Fires when price breaks out of Keltner Channels with volume confirmation.
        High-quality signal, but rare.

        Args:
            df: OHLCV DataFrame
            kc_period: KC period
            kc_multiplier: ATR multiplier

        Returns:
            Signal object
        """
        try:
            if len(df) < kc_period + 5:
                return Signal("keltner_breakout", "wait", 0.0)

            # Calculate Keltner Channels
            atr_values = TechnicalIndicators.atr(df['high'], df['low'], df['close'], kc_period)
            kc_middle = df['close'].rolling(window=kc_period).mean()
            kc_upper = kc_middle + (kc_multiplier * atr_values)
            kc_lower = kc_middle - (kc_multiplier * atr_values)

            if kc_upper.isna().all():
                return Signal("keltner_breakout", "wait", 0.0)

            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            upper = kc_upper.iloc[-1]
            lower = kc_lower.iloc[-1]

            # Volume confirmation
            volume_sma = df['volume'].rolling(window=20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0

            # Upper breakout
            if prev_price <= upper and current_price > upper and volume_ratio > 1.5:
                # Strong volume = higher strength
                strength = min(1.0, 0.7 + (volume_ratio - 1.5) * 0.2)
                return Signal("keltner_breakout", "buy", strength)

            # Lower breakout
            elif prev_price >= lower and current_price < lower and volume_ratio > 1.5:
                strength = min(1.0, 0.7 + (volume_ratio - 1.5) * 0.2)
                return Signal("keltner_breakout", "sell", strength)

            return Signal("keltner_breakout", "wait", 0.0)

        except Exception as e:
            logger.error(f"[KELTNER_BREAKOUT] Error: {e}")
            return Signal("keltner_breakout", "wait", 0.0)

    @staticmethod
    def ichimoku_cloud(df: pd.DataFrame) -> Signal:
        """
        Ichimoku Cloud signal - multi-timeframe trend confirmation.

        Strong signal when:
        - Price crosses cloud
        - Tenkan-Kijun cross occurs
        - Cloud is thick (strong trend)

        Args:
            df: OHLCV DataFrame

        Returns:
            Signal object
        """
        try:
            if len(df) < 52 + 26 + 5:  # Need enough data for Ichimoku
                return Signal("ichimoku", "wait", 0.0)

            # Calculate Ichimoku
            ichimoku = TechnicalIndicators.ichimoku_cloud(df['high'], df['low'], df['close'])

            tenkan = ichimoku['tenkan_sen']
            kijun = ichimoku['kijun_sen']
            senkou_a = ichimoku['senkou_span_a']
            senkou_b = ichimoku['senkou_span_b']

            if tenkan.isna().all():
                return Signal("ichimoku", "wait", 0.0)

            current_price = df['close'].iloc[-1]
            tenkan_current = tenkan.iloc[-1]
            kijun_current = kijun.iloc[-1]

            # Cloud boundaries (current position)
            cloud_top = max(senkou_a.iloc[-1], senkou_b.iloc[-1]) if not senkou_a.isna().iloc[-1] else None
            cloud_bottom = min(senkou_a.iloc[-1], senkou_b.iloc[-1]) if not senkou_a.isna().iloc[-1] else None

            if cloud_top is None or cloud_bottom is None:
                return Signal("ichimoku", "wait", 0.0)

            # Cloud thickness (trend strength)
            cloud_thickness = abs(cloud_top - cloud_bottom) / current_price if current_price > 0 else 0

            # Tenkan-Kijun cross
            prev_tenkan = tenkan.iloc[-2] if len(tenkan) > 1 else tenkan_current
            prev_kijun = kijun.iloc[-2] if len(kijun) > 1 else kijun_current

            bullish_cross = prev_tenkan < prev_kijun and tenkan_current > kijun_current
            bearish_cross = prev_tenkan > prev_kijun and tenkan_current < kijun_current

            # Price position relative to cloud
            price_above_cloud = current_price > cloud_top
            price_below_cloud = current_price < cloud_bottom
            price_in_cloud = not price_above_cloud and not price_below_cloud

            # Bullish signals
            if bullish_cross and (price_above_cloud or price_in_cloud):
                # Strength based on cloud thickness
                strength = 0.7 + min(0.3, cloud_thickness * 30)
                return Signal("ichimoku", "buy", strength)

            # Bearish signals
            elif bearish_cross and (price_below_cloud or price_in_cloud):
                strength = 0.7 + min(0.3, cloud_thickness * 30)
                return Signal("ichimoku", "sell", strength)

            # Price breaking through cloud
            prev_price = df['close'].iloc[-2]
            if prev_price <= cloud_top and current_price > cloud_top:
                # Breaking above cloud
                return Signal("ichimoku", "buy", 0.8)

            elif prev_price >= cloud_bottom and current_price < cloud_bottom:
                # Breaking below cloud
                return Signal("ichimoku", "sell", 0.8)

            return Signal("ichimoku", "wait", 0.0)

        except Exception as e:
            logger.error(f"[ICHIMOKU] Error: {e}")
            return Signal("ichimoku", "wait", 0.0)


# Signal weights for extended signals
EXTENDED_SIGNAL_WEIGHTS = {
    "supertrend": 1.3,         # Very reliable trend following
    "keltner_breakout": 1.4,   # Rare, high-quality breakouts
    "ichimoku": 1.2,           # Multi-timeframe confirmation
}


def get_all_extended_signals(df: pd.DataFrame) -> List[Signal]:
    """
    Get all extended signals for a DataFrame.

    Args:
        df: OHLCV DataFrame

    Returns:
        List of Signal objects
    """
    signals = [
        ExtendedIMBASignals.supertrend(df),
        ExtendedIMBASignals.keltner_breakout(df),
        ExtendedIMBASignals.ichimoku_cloud(df),
    ]

    return signals


def integrate_into_imba(imba_aggregator):
    """
    Integrate extended signals into existing IMBA aggregator.

    Usage:
        from strategy.imba_signals import IMBASignalAggregator
        from strategy.imba_signals_extended import integrate_into_imba

        aggregator = IMBASignalAggregator()
        integrate_into_imba(aggregator)

    Args:
        imba_aggregator: IMBASignalAggregator instance
    """
    # This would require modifying IMBASignalAggregator.aggregate()
    # to call get_all_extended_signals() and include them in voting

    logger.info("🚀 [EXTENDED_SIGNALS] Ready to integrate 3 new signals:")
    logger.info("   • SuperTrend (weight: 1.3)")
    logger.info("   • Keltner Breakout (weight: 1.4)")
    logger.info("   • Ichimoku Cloud (weight: 1.2)")

    return EXTENDED_SIGNAL_WEIGHTS
