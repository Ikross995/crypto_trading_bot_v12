"""
🧠 IMBA SIGNALS FROM ML FEATURES
=================================

Новые торговые сигналы на основе признаков из MLFeatures:
- Price velocity & acceleration
- EMA slope patterns
- Higher highs / lower lows
- Consolidation detection
- Market stress indicators

Эти сигналы дополняют существующие 11 IMBA сигналов.

"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import logging

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


class MLFeatureSignals:
    """Сигналы на основе ML признаков."""

    @staticmethod
    def price_momentum_signal(df: pd.DataFrame, velocity_threshold: float = 0.5) -> Signal:
        """
        Сигнал на основе price velocity & acceleration.

        Идея: Сильное ускорение цены = импульсный сигнал
        """
        try:
            if len(df) < 10:
                return Signal("price_momentum", "wait", 0.0)

            close = df['close']

            # Velocity (скорость изменения цены)
            returns = close.pct_change()
            velocity = returns.rolling(window=5).mean().iloc[-1] * 100  # %

            # Acceleration (ускорение)
            velocity_series = returns.rolling(window=5).mean()
            acceleration = velocity_series.diff().iloc[-1] * 1000

            # Сигнал при сильном положительном ускорении
            if acceleration > velocity_threshold and velocity > 0.1:
                strength = min(1.0, abs(acceleration) / 2.0)
                return Signal("price_momentum", "buy", max(0.5, strength))

            # Сигнал при сильном отрицательном ускорении
            elif acceleration < -velocity_threshold and velocity < -0.1:
                strength = min(1.0, abs(acceleration) / 2.0)
                return Signal("price_momentum", "sell", max(0.5, strength))

            return Signal("price_momentum", "wait", 0.0)

        except Exception as e:
            logger.error(f"[PRICE_MOMENTUM] Error: {e}")
            return Signal("price_momentum", "wait", 0.0)

    @staticmethod
    def trend_structure_signal(df: pd.DataFrame) -> Signal:
        """
        Сигнал на основе структуры тренда (higher highs / lower lows).

        Идея: Чистая трендовая структура = продолжение тренда
        """
        try:
            if len(df) < 20:
                return Signal("trend_structure", "wait", 0.0)

            highs = df['high']
            lows = df['low']

            # Проверяем последние N максимумов/минимумов
            lookback = 10
            recent_highs = highs.tail(lookback)
            recent_lows = lows.tail(lookback)

            # Higher highs pattern
            higher_highs_count = 0
            for i in range(1, len(recent_highs)):
                if recent_highs.iloc[i] > recent_highs.iloc[i-1]:
                    higher_highs_count += 1

            higher_highs_ratio = higher_highs_count / (lookback - 1)

            # Lower lows pattern
            lower_lows_count = 0
            for i in range(1, len(recent_lows)):
                if recent_lows.iloc[i] < recent_lows.iloc[i-1]:
                    lower_lows_count += 1

            lower_lows_ratio = lower_lows_count / (lookback - 1)

            # Сильный восходящий тренд
            if higher_highs_ratio > 0.7:
                return Signal("trend_structure", "buy", 0.7 + higher_highs_ratio * 0.3)

            # Сильный нисходящий тренд
            elif lower_lows_ratio > 0.7:
                return Signal("trend_structure", "sell", 0.7 + lower_lows_ratio * 0.3)

            return Signal("trend_structure", "wait", 0.0)

        except Exception as e:
            logger.error(f"[TREND_STRUCTURE] Error: {e}")
            return Signal("trend_structure", "wait", 0.0)

    @staticmethod
    def consolidation_breakout_signal(df: pd.DataFrame) -> Signal:
        """
        Сигнал пробоя консолидации.

        Идея: Цена долго в узком диапазоне → пробой = сильный сигнал
        """
        try:
            if len(df) < 30:
                return Signal("consolidation_breakout", "wait", 0.0)

            high = df['high']
            low = df['low']
            close = df['close']

            # Ищем консолидацию (низкая волатильность)
            lookback = 20
            recent_high = high.tail(lookback).max()
            recent_low = low.tail(lookback).min()
            range_pct = ((recent_high - recent_low) / recent_low) * 100

            # Консолидация = узкий диапазон
            is_consolidating = range_pct < 3.0  # Менее 3% диапазон

            if not is_consolidating:
                return Signal("consolidation_breakout", "wait", 0.0)

            # Проверяем пробой
            current_price = close.iloc[-1]
            prev_price = close.iloc[-2]

            # Пробой вверх
            if prev_price <= recent_high * 0.99 and current_price > recent_high:
                # Сила = насколько сильный пробой
                breakout_strength = ((current_price - recent_high) / recent_high) * 100
                strength = min(1.0, 0.7 + breakout_strength * 10)
                return Signal("consolidation_breakout", "buy", strength)

            # Пробой вниз
            elif prev_price >= recent_low * 1.01 and current_price < recent_low:
                breakout_strength = ((recent_low - current_price) / recent_low) * 100
                strength = min(1.0, 0.7 + breakout_strength * 10)
                return Signal("consolidation_breakout", "sell", strength)

            return Signal("consolidation_breakout", "wait", 0.0)

        except Exception as e:
            logger.error(f"[CONSOLIDATION_BREAKOUT] Error: {e}")
            return Signal("consolidation_breakout", "wait", 0.0)

    @staticmethod
    def ema_slope_trend_signal(df: pd.DataFrame, ema_period: int = 50) -> Signal:
        """
        Сигнал на основе наклона EMA.

        Идея: Крутой наклон EMA = сильный тренд
        """
        try:
            if len(df) < ema_period + 10:
                return Signal("ema_slope_trend", "wait", 0.0)

            close = df['close']

            # Вычисляем EMA
            ema = close.ewm(span=ema_period).mean()

            # Наклон EMA (изменение за последние 5 периодов)
            ema_change = ((ema.iloc[-1] - ema.iloc[-6]) / ema.iloc[-6]) * 100

            # Проверяем согласованность наклона (не флэт)
            ema_std = ema.tail(10).pct_change().std()

            # Сильный восходящий наклон
            if ema_change > 1.0 and ema_std > 0.001:
                strength = min(1.0, abs(ema_change) / 3.0)
                return Signal("ema_slope_trend", "buy", max(0.6, strength))

            # Сильный нисходящий наклон
            elif ema_change < -1.0 and ema_std > 0.001:
                strength = min(1.0, abs(ema_change) / 3.0)
                return Signal("ema_slope_trend", "sell", max(0.6, strength))

            return Signal("ema_slope_trend", "wait", 0.0)

        except Exception as e:
            logger.error(f"[EMA_SLOPE_TREND] Error: {e}")
            return Signal("ema_slope_trend", "wait", 0.0)

    @staticmethod
    def volatility_regime_signal(df: pd.DataFrame) -> Signal:
        """
        Сигнал на основе режима волатильности.

        Идея: Переход из низкой в высокую волатильность = сигнал входа
        """
        try:
            if len(df) < 50:
                return Signal("volatility_regime", "wait", 0.0)

            close = df['close']
            returns = close.pct_change()

            # Текущая волатильность (5-периодная)
            current_vol = returns.tail(5).std()

            # Историческая волатильность (50-периодная)
            hist_vol = returns.tail(50).std()

            # Процентиль текущей волатильности
            vol_percentile = (returns.tail(50).rolling(5).std() < current_vol).sum() / 50

            # Переход из низкой в высокую волатильность
            prev_vol = returns.iloc[-10:-5].std()

            # Expanding volatility (низкая → высокая) = готовность к движению
            if prev_vol < hist_vol * 0.8 and current_vol > hist_vol * 1.2:
                # Определяем направление по последнему движению
                recent_direction = close.iloc[-1] - close.iloc[-5]

                if recent_direction > 0:
                    return Signal("volatility_regime", "buy", 0.7)
                elif recent_direction < 0:
                    return Signal("volatility_regime", "sell", 0.7)

            return Signal("volatility_regime", "wait", 0.0)

        except Exception as e:
            logger.error(f"[VOLATILITY_REGIME] Error: {e}")
            return Signal("volatility_regime", "wait", 0.0)

    @staticmethod
    def market_stress_signal(df: pd.DataFrame) -> Signal:
        """
        Сигнал рыночного стресса (mean reversion opportunity).

        Идея: Экстремальный стресс = возможность разворота
        """
        try:
            if len(df) < 30:
                return Signal("market_stress", "wait", 0.0)

            close = df['close']
            high = df['high']
            low = df['low']

            # Стресс = комбинация волатильности + широких свечей
            atr = (high - low).tail(14).mean()
            current_range = high.iloc[-1] - low.iloc[-1]

            # Стресс индекс
            stress = current_range / atr if atr > 0 else 0

            # RSI для определения направления
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            # Высокий стресс + oversold = buy
            if stress > 2.0 and rsi < 30:
                strength = min(1.0, (35 - rsi) / 20)
                return Signal("market_stress", "buy", max(0.6, strength))

            # Высокий стресс + overbought = sell
            elif stress > 2.0 and rsi > 70:
                strength = min(1.0, (rsi - 65) / 20)
                return Signal("market_stress", "sell", max(0.6, strength))

            return Signal("market_stress", "wait", 0.0)

        except Exception as e:
            logger.error(f"[MARKET_STRESS] Error: {e}")
            return Signal("market_stress", "wait", 0.0)


# Signal weights for ML feature signals
ML_FEATURE_SIGNAL_WEIGHTS = {
    "price_momentum": 1.2,         # Импульсные движения точны
    "trend_structure": 1.3,        # Структура тренда надёжна
    "consolidation_breakout": 1.4, # Пробой консолидации сильный
    "ema_slope_trend": 1.1,        # EMA тренд стабилен
    "volatility_regime": 1.0,      # Волатильность средняя точность
    "market_stress": 0.9,          # Mean reversion рискованный
}


def get_all_ml_feature_signals(df: pd.DataFrame) -> list:
    """
    Получить все сигналы на основе ML признаков.

    Args:
        df: OHLCV DataFrame

    Returns:
        List of Signal objects
    """
    signals = [
        MLFeatureSignals.price_momentum_signal(df),
        MLFeatureSignals.trend_structure_signal(df),
        MLFeatureSignals.consolidation_breakout_signal(df),
        MLFeatureSignals.ema_slope_trend_signal(df),
        MLFeatureSignals.volatility_regime_signal(df),
        MLFeatureSignals.market_stress_signal(df),
    ]

    return signals
