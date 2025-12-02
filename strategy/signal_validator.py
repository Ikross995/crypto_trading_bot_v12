"""
🎓 SIGNAL VALIDATION & ML LEARNING SYSTEM
=========================================

Обучает ML модели на паттернах сигналов БЕЗ входа в реальные сделки.

Идея:
- IMBA генерирует сигнал (например, BUY @ $50,000)
- Даже если порог не пройден - запоминаем сигнал
- Через N свечей проверяем: правильный был сигнал или нет
- Обучаем ML на этих "виртуальных" сделках

Преимущества:
✅ ML учится на ВСЕХ сигналах (не только на реальных сделках)
✅ Собирает 10-100x больше обучающих данных
✅ Может обучаться даже при высоком пороге
✅ Учится распознавать хорошие/плохие паттерны

"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from collections import deque
import json
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VirtualSignal:
    """Виртуальный сигнал для обучения ML."""
    signal_id: str
    timestamp: datetime
    symbol: str
    direction: str  # 'buy' or 'sell'
    strength: float
    price: float

    # Контекст рынка на момент сигнала
    market_context: Dict

    # IMBA детали
    imba_votes: Dict  # {'buy': 0.8, 'sell': 0.0}
    active_signals: List[str]  # Какие индикаторы сработали

    # Результаты (заполняется позже)
    outcome_validated: bool = False
    max_favorable_move: float = 0.0  # Максимальное движение в нужную сторону
    max_adverse_move: float = 0.0    # Максимальное движение против
    outcome_pnl_pct: float = 0.0     # % PnL если бы вошли
    was_correct: bool = False        # Правильный ли был сигнал
    validation_price: float = 0.0    # Цена через N свечей


@dataclass
class ValidationConfig:
    """Конфигурация валидации сигналов."""
    validation_periods: List[int] = None  # Через сколько свечей проверять [5, 10, 20]
    min_profit_threshold: float = 0.5     # Минимум % для "правильного" сигнала
    max_drawdown_threshold: float = -2.0  # Максимум % просадки

    def __post_init__(self):
        if self.validation_periods is None:
            self.validation_periods = [5, 10, 20]  # 5, 10, 20 свечей


class SignalValidator:
    """
    Валидатор сигналов для обучения ML.

    Отслеживает все сигналы и проверяет их корректность.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()

        # Очередь непроверенных сигналов
        self.pending_signals = deque(maxlen=1000)

        # История валидированных сигналов
        self.validated_signals = deque(maxlen=10000)

        # Кеш цен для валидации
        self.price_history = {}  # {symbol: deque([(timestamp, ohlc), ...])}

        # Статистика
        self.stats = {
            'total_signals': 0,
            'validated_signals': 0,
            'correct_signals': 0,
            'incorrect_signals': 0,
            'win_rate': 0.0
        }

        # Путь для сохранения
        self.data_dir = Path("ml_learning_data/signal_validation")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🎓 [SIGNAL_VALIDATOR] Initialized")
        self._load_historical_data()

    def record_signal(
        self,
        symbol: str,
        direction: str,
        strength: float,
        price: float,
        market_context: Dict,
        imba_votes: Dict,
        active_signals: List[str]
    ) -> str:
        """
        Записывает сигнал для последующей валидации.

        Args:
            symbol: Торговая пара
            direction: 'buy' или 'sell'
            strength: Сила сигнала (0-10)
            price: Цена на момент сигнала
            market_context: Контекст рынка (MarketContext.dict())
            imba_votes: Голоса IMBA
            active_signals: Список сработавших индикаторов

        Returns:
            signal_id: ID созданного сигнала
        """
        timestamp = datetime.now(timezone.utc)
        signal_id = f"{symbol}_{timestamp.timestamp()}_{direction}"

        virtual_signal = VirtualSignal(
            signal_id=signal_id,
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strength=strength,
            price=price,
            market_context=market_context,
            imba_votes=imba_votes,
            active_signals=active_signals
        )

        self.pending_signals.append(virtual_signal)
        self.stats['total_signals'] += 1

        logger.debug(
            f"🎓 [SIGNAL_RECORDED] {symbol} {direction.upper()} @ ${price:.2f} "
            f"(strength: {strength:.2f}, votes: {imba_votes})"
        )

        return signal_id

    def update_price_history(self, symbol: str, timestamp: datetime, ohlc: Dict):
        """
        Обновляет историю цен для валидации.

        Args:
            symbol: Торговая пара
            timestamp: Временная метка
            ohlc: {'open': x, 'high': x, 'low': x, 'close': x}
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)

        self.price_history[symbol].append((timestamp, ohlc))

    def validate_pending_signals(self, current_time: datetime = None):
        """
        Валидирует ожидающие сигналы на основе истории цен.

        Проверяет сигналы через N свечей и определяет были ли они правильными.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        signals_to_remove = []
        newly_validated = 0

        for signal in self.pending_signals:
            # Проверяем готов ли сигнал к валидации
            time_since_signal = (current_time - signal.timestamp).total_seconds()

            # Пропускаем слишком свежие сигналы
            if time_since_signal < 60:  # Минимум 1 минута
                continue

            # Валидируем
            validation_result = self._validate_signal(signal, current_time)

            if validation_result is not None:
                # Обновляем сигнал результатами
                signal.outcome_validated = True
                signal.max_favorable_move = validation_result['max_favorable']
                signal.max_adverse_move = validation_result['max_adverse']
                signal.outcome_pnl_pct = validation_result['pnl_pct']
                signal.was_correct = validation_result['was_correct']
                signal.validation_price = validation_result['final_price']

                # Перемещаем в валидированные
                self.validated_signals.append(signal)
                signals_to_remove.append(signal)
                newly_validated += 1

                # Обновляем статистику
                self.stats['validated_signals'] += 1
                if signal.was_correct:
                    self.stats['correct_signals'] += 1
                else:
                    self.stats['incorrect_signals'] += 1

                # Обновляем win rate
                if self.stats['validated_signals'] > 0:
                    self.stats['win_rate'] = (
                        self.stats['correct_signals'] / self.stats['validated_signals']
                    )

                logger.debug(
                    f"✅ [VALIDATED] {signal.symbol} {signal.direction.upper()} "
                    f"@ ${signal.price:.2f} → "
                    f"{'CORRECT' if signal.was_correct else 'INCORRECT'} "
                    f"(PnL: {signal.outcome_pnl_pct:+.2f}%)"
                )

        # Удаляем валидированные из очереди
        for signal in signals_to_remove:
            self.pending_signals.remove(signal)

        if newly_validated > 0:
            logger.info(
                f"🎓 [VALIDATION_BATCH] Validated {newly_validated} signals. "
                f"Win rate: {self.stats['win_rate']:.1%} "
                f"({self.stats['correct_signals']}/{self.stats['validated_signals']})"
            )

        return newly_validated

    def _validate_signal(
        self,
        signal: VirtualSignal,
        current_time: datetime
    ) -> Optional[Dict]:
        """
        Валидирует один сигнал на основе движения цены.

        Returns:
            Dict с результатами или None если недостаточно данных
        """
        symbol = signal.symbol

        # Проверяем есть ли история цен
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            return None

        # Фильтруем цены после сигнала
        future_prices = [
            (ts, ohlc) for ts, ohlc in self.price_history[symbol]
            if ts > signal.timestamp
        ]

        if len(future_prices) < 3:  # Минимум 3 свечи для валидации
            return None

        # Берём период для валидации (например, 10 свечей)
        validation_period = min(len(future_prices), 10)
        prices_to_check = future_prices[:validation_period]

        # Вычисляем максимальное движение
        entry_price = signal.price
        max_favorable = 0.0
        max_adverse = 0.0

        for ts, ohlc in prices_to_check:
            high = ohlc['high']
            low = ohlc['low']

            if signal.direction == 'buy':
                # Для BUY: favorable = рост, adverse = падение
                favorable_move = ((high - entry_price) / entry_price) * 100
                adverse_move = ((low - entry_price) / entry_price) * 100
            else:
                # Для SELL: favorable = падение, adverse = рост
                favorable_move = ((entry_price - low) / entry_price) * 100
                adverse_move = ((entry_price - high) / entry_price) * 100

            max_favorable = max(max_favorable, favorable_move)
            max_adverse = min(max_adverse, adverse_move)

        # Финальная цена (close последней свечи)
        final_price = prices_to_check[-1][1]['close']

        if signal.direction == 'buy':
            pnl_pct = ((final_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - final_price) / entry_price) * 100

        # Определяем правильность сигнала
        was_correct = (
            max_favorable >= self.config.min_profit_threshold and
            max_adverse >= self.config.max_drawdown_threshold
        )

        return {
            'max_favorable': max_favorable,
            'max_adverse': max_adverse,
            'pnl_pct': pnl_pct,
            'was_correct': was_correct,
            'final_price': final_price
        }

    def get_training_data_for_ml(self, min_samples: int = 30) -> Optional[Tuple]:
        """
        Возвращает данные для обучения ML моделей.

        Returns:
            Tuple of (features, labels) или None если мало данных
        """
        if len(self.validated_signals) < min_samples:
            logger.debug(
                f"🎓 [ML_DATA] Not enough validated signals: "
                f"{len(self.validated_signals)}/{min_samples}"
            )
            return None

        features_list = []
        labels_list = []

        for signal in self.validated_signals:
            # Извлекаем признаки из market_context
            if not signal.market_context:
                continue

            # Создаём feature vector
            features = self._extract_features_from_signal(signal)

            # Label: 1 если сигнал правильный, 0 если нет
            label = 1 if signal.was_correct else 0

            features_list.append(features)
            labels_list.append(label)

        if len(features_list) < min_samples:
            return None

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(
            f"🎓 [ML_DATA] Prepared {len(X)} samples for training "
            f"(Win rate: {y.mean():.1%})"
        )

        return X, y

    def _extract_features_from_signal(self, signal: VirtualSignal) -> List[float]:
        """Извлекает признаки из сигнала для ML."""
        ctx = signal.market_context

        features = [
            signal.strength,
            signal.imba_votes.get('buy', 0.0),
            signal.imba_votes.get('sell', 0.0),
            ctx.get('rsi_14', 50.0),
            ctx.get('macd', 0.0),
            ctx.get('atr_14', 0.0),
            ctx.get('volatility_percentile', 50.0),
            ctx.get('trend_strength', 0.0),
            ctx.get('volume_ratio', 1.0),
            len(signal.active_signals),  # Количество активных индикаторов
        ]

        return features

    def save_data(self):
        """Сохраняет валидированные сигналы."""
        try:
            signals_data = [asdict(s) for s in self.validated_signals]

            with open(self.data_dir / "validated_signals.json", 'w') as f:
                json.dump(signals_data, f, indent=2, default=str)

            with open(self.data_dir / "stats.json", 'w') as f:
                json.dump(self.stats, f, indent=2)

            logger.info(
                f"💾 [SIGNAL_VALIDATOR] Saved {len(self.validated_signals)} "
                f"validated signals"
            )

        except Exception as e:
            logger.error(f"❌ [SAVE] Error: {e}")

    def _load_historical_data(self):
        """Загружает исторические валидированные сигналы."""
        try:
            signals_file = self.data_dir / "validated_signals.json"
            stats_file = self.data_dir / "stats.json"

            if signals_file.exists():
                with open(signals_file, 'r') as f:
                    signals_data = json.load(f)

                for sig_dict in signals_data:
                    # Восстанавливаем datetime
                    if 'timestamp' in sig_dict:
                        sig_dict['timestamp'] = datetime.fromisoformat(
                            sig_dict['timestamp'].replace('Z', '+00:00')
                        )
                    signal = VirtualSignal(**sig_dict)
                    self.validated_signals.append(signal)

                logger.info(
                    f"📚 [LOAD] Loaded {len(self.validated_signals)} "
                    f"historical signals"
                )

            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats.update(json.load(f))

        except Exception as e:
            logger.error(f"❌ [LOAD] Error: {e}")

    def get_stats(self) -> Dict:
        """Возвращает статистику валидации."""
        return {
            **self.stats,
            'pending_signals': len(self.pending_signals),
            'validated_signals_count': len(self.validated_signals)
        }


# Глобальный экземпляр
_signal_validator = None


def get_signal_validator(config: ValidationConfig = None) -> SignalValidator:
    """Получить или создать глобальный валидатор."""
    global _signal_validator

    if _signal_validator is None:
        _signal_validator = SignalValidator(config)

    return _signal_validator
