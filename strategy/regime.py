"""
Market Regime Detection System.

Detects market regimes (trend, flat, volatile) based on ADX and Bollinger Bands Width.
Used to adjust signal weights and filter trades based on market conditions.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

from data.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class Regime:
    """Market regime information."""
    kind: str  # "trend" | "flat" | "volatile"
    adx: float
    bbw: float
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "kind": self.kind,
            "adx": self.adx,
            "bbw": self.bbw,
            "confidence": self.confidence
        }


class RegimeDetector:
    """
    Detects market regime based on technical indicators.
    
    Regime Types:
    - Trend: Strong directional movement (ADX >= 25, BB Width > 0.01)
    - Flat: Low volatility range (ADX < 20, BB Width < 0.005)
    - Volatile: High volatility without clear direction
    """
    
    def __init__(self, 
                 trend_adx_threshold: float = 25.0,
                 flat_adx_threshold: float = 20.0,
                 trend_bbw_threshold: float = 0.01,
                 flat_bbw_threshold: float = 0.005):
        """
        Initialize regime detector.
        
        Args:
            trend_adx_threshold: Minimum ADX for trend regime
            flat_adx_threshold: Maximum ADX for flat regime
            trend_bbw_threshold: Minimum BB Width for trend regime
            flat_bbw_threshold: Maximum BB Width for flat regime
        """
        self.trend_adx_threshold = trend_adx_threshold
        self.flat_adx_threshold = flat_adx_threshold
        self.trend_bbw_threshold = trend_bbw_threshold
        self.flat_bbw_threshold = flat_bbw_threshold
    
    def detect_regime(self, df: pd.DataFrame) -> Regime:
        """
        Detect current market regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Regime object with detected regime information
        """
        if len(df) < 50:
            return Regime(kind="unknown", adx=0.0, bbw=0.0, confidence=0.0)
        
        try:
            # Calculate ADX if not present
            if 'adx' not in df.columns:
                df['adx'] = TechnicalIndicators.adx(
                    df['high'], df['low'], df['close'], period=14
                )
            
            # Calculate BB Width if not present
            if 'bb_width' not in df.columns:
                bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                    df['close'], period=20, std_dev=2.0
                )
                df['bb_width'] = (bb_upper - bb_lower) / df['close']
            
            # Get latest values
            last = df.iloc[-1]
            adx = float(last['adx']) if not pd.isna(last['adx']) else 0.0
            bbw = float(last['bb_width']) if not pd.isna(last['bb_width']) else 0.0
            
            # Detect regime
            regime_kind = "volatile"
            confidence = 0.5
            
            # Trend regime: Strong ADX + Wide BB
            if adx >= self.trend_adx_threshold and bbw > self.trend_bbw_threshold:
                regime_kind = "trend"
                confidence = min(0.9, (adx / 50.0) * (bbw / 0.03))
            
            # Flat regime: Weak ADX + Narrow BB
            elif adx < self.flat_adx_threshold and bbw < self.flat_bbw_threshold:
                regime_kind = "flat"
                confidence = min(0.9, (1.0 - adx / 30.0) * (1.0 - bbw / 0.01))
            
            # Volatile regime: Everything else
            else:
                regime_kind = "volatile"
                confidence = 0.6
            
            logger.info(
                f"Regime detected: {regime_kind} (ADX={adx:.2f}, BBW={bbw:.4f}, "
                f"confidence={confidence:.2f})"
            )
            
            return Regime(
                kind=regime_kind,
                adx=adx,
                bbw=bbw,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return Regime(kind="unknown", adx=0.0, bbw=0.0, confidence=0.0)
    
    def regime_filter(self, regime: Regime, signal_name: str, signal_direction: str) -> bool:
        """
        Filter signals based on regime compatibility.

        ⚠️ КРИТИЧНО: Mean-reversion сигналы БЛОКИРУЮТСЯ в трендах!
        Они открывают ПРОТИВ тренда и приводят к убыткам.

        Args:
            regime: Current market regime
            signal_name: Name of the signal
            signal_direction: Signal direction ('buy', 'sell', 'wait')

        Returns:
            True if signal should be allowed, False to filter out
        """
        if signal_direction == "wait":
            return False

        # ⚠️ MEAN-REVERSION СИГНАЛЫ - ОПАСНЫ В ТРЕНДАХ!
        # Они открывают ПРОТИВ движения цены
        mean_reversion_signals = [
            "rsi_mr",           # RSI oversold/overbought
            "vwap_bands_mr",    # VWAP band bounces
            "sfp",              # Swing Failure Pattern
            "volume_profile",   # POC mean reversion
            "fvg",              # Fair Value Gap fills
            "market_stress",    # Stress mean reversion
        ]

        # TREND-FOLLOWING СИГНАЛЫ - безопасны, следуют за трендом
        trend_following_signals = [
            "ema_pinch",        # EMA convergence + breakout
            "atr_momentum",     # Strong momentum candles
            "breakout_retest",  # Breakout + retest
            "bb_squeeze",       # BB squeeze breakout
            "cvd",              # Volume delta divergence
            "trend_structure",  # Higher highs/lows
            "consolidation_breakout",
            "price_momentum",
            "ema_slope_trend",
        ]

        # ========== TREND REGIME (ADX >= 25) ==========
        if regime.kind == "trend":
            # 🚫 БЛОКИРОВАТЬ mean-reversion сигналы в ЛЮБОМ тренде!
            if signal_name in mean_reversion_signals:
                logger.warning(
                    f"🚫 BLOCKED {signal_name} in TREND regime (ADX={regime.adx:.1f}) - "
                    f"mean-reversion signals are DANGEROUS in trends!"
                )
                return False

            # ✅ Разрешить trend-following сигналы
            return True

        # ========== FLAT REGIME (ADX < 20) ==========
        elif regime.kind == "flat":
            # В флэте mean-reversion работает лучше, но с осторожностью
            # Блокируем только самые агрессивные breakout сигналы
            if signal_name == "atr_momentum" and regime.adx < 15:
                logger.debug(f"Filtering {signal_name} in very flat market (ADX={regime.adx:.2f})")
                return False
            return True

        # ========== VOLATILE REGIME ==========
        else:
            # В волатильности блокируем mean-reversion - слишком рискованно
            if signal_name in mean_reversion_signals and regime.adx > 22:
                logger.warning(
                    f"🚫 BLOCKED {signal_name} in VOLATILE regime (ADX={regime.adx:.1f})"
                )
                return False
            return True
    
    def get_regime_multiplier(self, regime: Regime, signal_name: str) -> float:
        """
        Get signal strength multiplier based on regime.

        ⚠️ КРИТИЧНО: Mean-reversion получает ШТРАФ в трендах!

        Args:
            regime: Current market regime
            signal_name: Name of the signal

        Returns:
            Multiplier for signal strength (e.g., 1.2 = +20%, 0.5 = -50%)
        """
        # TREND-FOLLOWING сигналы - бонус в тренде
        trend_signals = [
            "atr_momentum", "bb_squeeze", "breakout_retest",
            "ema_pinch", "trend_structure", "consolidation_breakout",
            "price_momentum", "ema_slope_trend", "cvd"
        ]

        # MEAN-REVERSION сигналы - бонус только во флэте, штраф в тренде
        mean_reversion_signals = [
            "rsi_mr", "sfp", "vwap_bands_mr", "vwap_pullback",
            "volume_profile", "fvg", "market_stress"
        ]

        # Orderbook imbalance: +5% always
        if signal_name == "obi":
            return 1.05

        # ========== TREND REGIME ==========
        if regime.kind == "trend":
            # Trend-following сигналы получают +20% бонус
            if signal_name in trend_signals:
                return 1.2

            # Mean-reversion получает ШТРАФ -50% (если вообще дошёл сюда)
            if signal_name in mean_reversion_signals:
                return 0.5

        # ========== FLAT REGIME ==========
        elif regime.kind == "flat":
            # Mean-reversion получает +15% бонус во флэте
            if signal_name in mean_reversion_signals:
                return 1.15

            # Trend-following без бонуса во флэте
            if signal_name in trend_signals:
                return 0.9

        # ========== VOLATILE REGIME ==========
        else:
            # В волатильности все сигналы слабее
            if signal_name in mean_reversion_signals:
                return 0.6  # Mean-reversion рискован
            return 0.9  # Остальные немного слабее

        # Default: no bonus
        return 1.0
    
    def regime_stats(self, df: pd.DataFrame, lookback: int = 100) -> dict:
        """
        Calculate regime statistics over a period.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of candles to analyze
            
        Returns:
            Dictionary with regime statistics
        """
        if len(df) < lookback:
            return {"error": "Insufficient data"}
        
        recent_df = df.tail(lookback).copy()
        regimes = []
        
        for i in range(len(recent_df)):
            sample_df = recent_df.iloc[:i+1]
            if len(sample_df) >= 50:
                regime = self.detect_regime(sample_df)
                regimes.append(regime.kind)
        
        if not regimes:
            return {"error": "No regimes detected"}
        
        # Count regime occurrences
        regime_counts = {
            "trend": regimes.count("trend"),
            "flat": regimes.count("flat"),
            "volatile": regimes.count("volatile"),
            "unknown": regimes.count("unknown")
        }
        
        total = len(regimes)
        regime_pct = {k: (v / total * 100) for k, v in regime_counts.items()}
        
        return {
            "total_periods": total,
            "counts": regime_counts,
            "percentages": regime_pct,
            "current_regime": regimes[-1] if regimes else "unknown"
        }


def detect_regime(df: pd.DataFrame, 
                 trend_adx: float = 25.0,
                 flat_adx: float = 20.0) -> Regime:
    """
    Convenience function to detect regime.
    
    Args:
        df: DataFrame with OHLCV data
        trend_adx: ADX threshold for trend
        flat_adx: ADX threshold for flat
        
    Returns:
        Regime object
    """
    detector = RegimeDetector(
        trend_adx_threshold=trend_adx,
        flat_adx_threshold=flat_adx
    )
    return detector.detect_regime(df)
