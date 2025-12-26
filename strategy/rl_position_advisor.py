"""
RL Position Advisor - Умное управление позициями через RL Agent

Работает ВМЕСТЕ с IMBA стратегией:
- IMBA ставит тейки и начальные стопы
- После TP2 RL Agent берет управление
- Делает trailing stop (двигает стоп за ценой)
- Закрывает досрочно если видит разворот

Автор: Claude (Anthropic)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Состояние открытой позиции"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    current_price: float
    position_size: float
    entry_time: datetime

    # Стадия позиции
    tp1_hit: bool = False  # Сработал TP1
    tp2_hit: bool = False  # Сработал TP2

    # Стопы и тейки
    current_stop: float = 0.0
    trailing_stop: Optional[float] = None
    highest_price: Optional[float] = None  # Для LONG
    lowest_price: Optional[float] = None   # Для SHORT

    # Метрики
    unrealized_pnl_pct: float = 0.0
    max_pnl_pct: float = 0.0  # Максимальная прибыль

    def update_price(self, new_price: float):
        """Обновить текущую цену и метрики"""
        self.current_price = new_price

        # Обновить highest/lowest
        if self.side == 'LONG':
            if self.highest_price is None or new_price > self.highest_price:
                self.highest_price = new_price
        else:  # SHORT
            if self.lowest_price is None or new_price < self.lowest_price:
                self.lowest_price = new_price

        # Пересчитать PnL
        if self.side == 'LONG':
            self.unrealized_pnl_pct = (new_price - self.entry_price) / self.entry_price * 100
        else:  # SHORT
            self.unrealized_pnl_pct = (self.entry_price - new_price) / self.entry_price * 100

        # Обновить максимум
        if self.unrealized_pnl_pct > self.max_pnl_pct:
            self.max_pnl_pct = self.unrealized_pnl_pct


class RLPositionAdvisor:
    """
    RL Agent как советник для управления позициями.

    Функции:
    1. Досрочное закрытие (если видит разворот)
    2. Trailing stop (после TP2)
    3. Защита прибыли
    """

    def __init__(self, config):
        """
        Инициализация RL Position Advisor.

        Args:
            config: Конфигурация бота
        """
        self.config = config
        self.positions: Dict[str, PositionState] = {}

        # COMBO интеграция
        self.combo_integration = None
        if getattr(config, 'use_combo_signals', False):
            try:
                from strategy.combo_integration import COMBOSignalIntegration
                self.combo_integration = COMBOSignalIntegration(config)
                logger.info("✅ RL Position Advisor initialized with COMBO models")
            except Exception as e:
                logger.warning(f"Failed to load COMBO for advisor: {e}")

        # Параметры
        self.close_confidence_min = getattr(config, 'rl_close_confidence_min', 0.75)
        self.emergency_confidence = getattr(config, 'rl_emergency_confidence', 0.95)
        self.trailing_distance_pct = getattr(config, 'rl_trailing_distance_pct', 3.0)  # -3% от максимума

        logger.info(f"RL Position Advisor settings:")
        logger.info(f"  Close confidence: {self.close_confidence_min:.0%}")
        logger.info(f"  Emergency confidence: {self.emergency_confidence:.0%}")
        logger.info(f"  Trailing distance: {self.trailing_distance_pct}%")

    def register_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        position_size: float,
        initial_stop: float
    ):
        """
        Зарегистрировать новую позицию для отслеживания.

        Args:
            symbol: Торговый символ
            side: 'LONG' или 'SHORT'
            entry_price: Цена входа
            position_size: Размер позиции
            initial_stop: Начальный стоп-лосс
        """
        position = PositionState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            position_size=position_size,
            entry_time=datetime.now(timezone.utc),
            current_stop=initial_stop
        )

        self.positions[symbol] = position

        logger.info(f"📍 Registered position: {symbol} {side} @ ${entry_price:,.2f}")
        logger.info(f"   Initial stop: ${initial_stop:,.2f}")

    def mark_tp_hit(self, symbol: str, tp_level: int):
        """
        Отметить срабатывание тейк-профита.

        Args:
            symbol: Торговый символ
            tp_level: Уровень тейка (1, 2, 3)
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        if tp_level == 1:
            position.tp1_hit = True
            logger.info(f"✅ TP1 hit for {symbol}, stop unchanged")

        elif tp_level == 2:
            position.tp2_hit = True
            # Стоп в безубыток
            position.current_stop = position.entry_price
            position.trailing_stop = position.entry_price
            logger.info(f"✅ TP2 hit for {symbol}, stop → breakeven ${position.entry_price:,.2f}")
            logger.info(f"🎯 RL Trailing Stop ACTIVATED for {symbol}")

    def update_and_advise(
        self,
        symbol: str,
        current_price: float,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Обновить позицию и получить совет от RL Agent.

        Args:
            symbol: Торговый символ
            current_price: Текущая цена
            market_data: Данные рынка (свечи + индикаторы)

        Returns:
            {
                'action': 'hold' | 'close' | 'update_stop',
                'confidence': 0.0-1.0,
                'new_stop': float (если action='update_stop'),
                'reason': str
            }
        """
        if symbol not in self.positions:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No position tracked'}

        position = self.positions[symbol]
        position.update_price(current_price)

        # До TP2 - не вмешиваемся, только наблюдаем
        if not position.tp2_hit:
            return self._check_early_close(position, market_data)

        # После TP2 - активируем trailing stop + проверяем разворот
        return self._manage_trailing_stop(position, market_data)

    def _check_early_close(
        self,
        position: PositionState,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Проверить нужно ли закрыть досрочно (до TP2).

        Args:
            position: Состояние позиции
            market_data: Данные рынка

        Returns:
            Совет по действию
        """
        if self.combo_integration is None:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'COMBO not available'}

        try:
            # Получаем решение RL Agent
            rl_signal = self.combo_integration.generate_signal_from_df(
                df=market_data,
                symbol=position.symbol
            )

            rl_direction = rl_signal.get('direction', 'wait')
            rl_confidence = rl_signal.get('confidence', 0.0)

            # Проверяем противоположный сигнал
            should_close = False
            if position.side == 'LONG' and rl_direction == 'sell':
                should_close = True
            elif position.side == 'SHORT' and rl_direction == 'buy':
                should_close = True

            if should_close:
                if rl_confidence >= self.emergency_confidence:
                    # ЭКСТРЕННОЕ закрытие!
                    return {
                        'action': 'close',
                        'confidence': rl_confidence,
                        'reason': f'🚨 EMERGENCY: RL sees strong reversal (conf={rl_confidence:.0%})'
                    }
                elif rl_confidence >= self.close_confidence_min:
                    # Обычное досрочное закрытие
                    return {
                        'action': 'close',
                        'confidence': rl_confidence,
                        'reason': f'⚠️ EARLY CLOSE: RL predicts reversal (conf={rl_confidence:.0%})'
                    }

            return {
                'action': 'hold',
                'confidence': rl_confidence,
                'reason': 'RL supports holding'
            }

        except Exception as e:
            logger.error(f"Error in early close check: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}

    def _manage_trailing_stop(
        self,
        position: PositionState,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Управление trailing stop после TP2.

        Args:
            position: Состояние позиции
            market_data: Данные рынка

        Returns:
            Совет по действию
        """
        # Сначала проверяем - нужно ли закрывать досрочно
        early_close = self._check_early_close(position, market_data)
        if early_close['action'] == 'close':
            return early_close

        # Теперь обновляем trailing stop
        new_stop = self._calculate_trailing_stop(position)

        if new_stop is None:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': 'Trailing stop unchanged'
            }

        # Проверяем - нужно ли двигать стоп
        if position.trailing_stop is None or new_stop != position.trailing_stop:
            # Стоп должен двигаться только ВВЕРХ (для LONG) или ВНИЗ (для SHORT)
            should_update = False

            if position.side == 'LONG' and new_stop > position.trailing_stop:
                should_update = True
            elif position.side == 'SHORT' and new_stop < position.trailing_stop:
                should_update = True

            if should_update:
                old_stop = position.trailing_stop
                position.trailing_stop = new_stop
                position.current_stop = new_stop

                distance_from_entry = (new_stop - position.entry_price) / position.entry_price * 100

                logger.info(
                    f"📈 Trailing stop moved for {position.symbol}: "
                    f"${old_stop:,.2f} → ${new_stop:,.2f} "
                    f"({distance_from_entry:+.1f}% from entry)"
                )

                return {
                    'action': 'update_stop',
                    'confidence': 1.0,
                    'new_stop': new_stop,
                    'reason': f'Trailing stop: ${old_stop:,.2f} → ${new_stop:,.2f}'
                }

        return {
            'action': 'hold',
            'confidence': 0.0,
            'reason': 'Trailing stop optimal'
        }

    def _calculate_trailing_stop(self, position: PositionState) -> Optional[float]:
        """
        Рассчитать новый trailing stop.

        Args:
            position: Состояние позиции

        Returns:
            Новый уровень стопа или None
        """
        if position.side == 'LONG':
            if position.highest_price is None:
                return None

            # Стоп на X% ниже максимума
            new_stop = position.highest_price * (1 - self.trailing_distance_pct / 100)

            # Но не ниже безубытка
            if new_stop < position.entry_price:
                new_stop = position.entry_price

            return new_stop

        else:  # SHORT
            if position.lowest_price is None:
                return None

            # Стоп на X% выше минимума
            new_stop = position.lowest_price * (1 + self.trailing_distance_pct / 100)

            # Но не выше безубытка
            if new_stop > position.entry_price:
                new_stop = position.entry_price

            return new_stop

    def remove_position(self, symbol: str):
        """
        Удалить позицию из отслеживания (после закрытия).

        Args:
            symbol: Торговый символ
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            logger.info(
                f"📊 Position closed: {symbol} "
                f"Final PnL: {position.unrealized_pnl_pct:+.2f}% "
                f"(Max: {position.max_pnl_pct:+.2f}%)"
            )
            del self.positions[symbol]

    def get_position_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получить статистику по позиции.

        Args:
            symbol: Торговый символ

        Returns:
            Словарь со статистикой или None
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        return {
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'unrealized_pnl_pct': position.unrealized_pnl_pct,
            'max_pnl_pct': position.max_pnl_pct,
            'tp1_hit': position.tp1_hit,
            'tp2_hit': position.tp2_hit,
            'current_stop': position.current_stop,
            'trailing_stop': position.trailing_stop,
            'highest_price': position.highest_price,
            'lowest_price': position.lowest_price,
        }
