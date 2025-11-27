"""
🧠 REAL MACHINE LEARNING SYSTEM FOR CRYPTO TRADING
===================================================

Реальная система машинного обучения с:
- Feature Engineering (техническая и фундаментальная информация)
- Online Learning (обучение в реальном времени)
- Multi-objective Optimization (не только PnL)
- Ensemble Methods (комбинирование моделей)
- Contextual Learning (учет рыночных условий)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import deque
import json
import logging
from pathlib import Path

# ML библиотеки
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import SGDRegressor, LogisticRegression, SGDClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

# 🔢 MODEL VERSION - увеличивайте при изменении логики обучения!
# v1: Initial implementation
# v2: Added target clipping (+/-20% PnL, 0-24h hold_time, 0-10% risk)
# v3: Added trend direction classifier and trend strength predictor
# v4: Separate models per trading pair (symbol-specific learning)
MODEL_VERSION = 4

@dataclass
class MarketContext:
    """Контекст рынка во время сделки"""
    timestamp: datetime
    symbol: str
    
    # Технические индикаторы
    rsi_14: float
    rsi_7: float
    macd: float
    macd_signal: float
    bb_position: float  # Позиция относительно Bollinger Bands
    sma_20: float
    ema_50: float
    atr_14: float
    volume_ratio: float  # Соотношение текущего объема к среднему
    
    # Рыночные условия
    volatility_percentile: float  # Процентиль волатильности (0-100)
    trend_strength: float  # Сила тренда
    market_regime: str  # "trending", "ranging", "volatile"
    fear_greed_index: int
    btc_dominance: float
    
    # Временные факторы
    hour_of_day: int
    day_of_week: int
    session: str  # "asian", "european", "american"
    
    # Ценовые уровни
    support_distance: float  # Расстояние до ближайшей поддержки (%)
    resistance_distance: float  # Расстояние до ближайшего сопротивления (%)
    
    # Спреды и ликвидность
    bid_ask_spread: float
    order_book_imbalance: float

@dataclass
class TradeOutcome:
    """Результат сделки с дополнительными метриками"""
    trade_id: str
    pnl: float
    pnl_pct: float
    hold_time_minutes: float
    exit_reason: str
    
    # Качественные метрики
    sharpe_ratio: float
    max_favorable_excursion: float  # MFE
    max_adverse_excursion: float   # MAE
    win_probability: float  # Вероятность успеха на момент входа
    
    # Эмоциональные факторы
    stress_level: float  # Уровень "стресса" позиции
    confidence_decay: float  # Как менялась уверенность

@dataclass
class MLFeatures:
    """Набор признаков для обучения ML моделей"""

    # Технические признаки
    rsi_momentum: float
    macd_divergence: float
    volume_surge: float
    price_momentum: float
    volatility_regime: float

    # Рыночные признаки
    market_stress: float
    trend_alignment: float
    support_strength: float

    # Временные признаки
    session_volatility: float
    day_performance: float

    # Мета-признаки
    signal_confluence: float
    historical_accuracy: float

    # 🆕 НОВЫЕ признаки для распознавания трендов
    price_velocity: float  # Скорость изменения цены
    price_acceleration: float  # Ускорение изменения цены
    ema_slope: float  # Наклон EMA (тренд)
    higher_highs: float  # Паттерн растущих максимумов (0-1)
    lower_lows: float  # Паттерн падающих минимумов (0-1)
    consolidation_score: float  # Индикатор консолидации/флэта (0-1)

class OnlineLearningModel:
    """Модель онлайн-обучения для регрессии"""

    def __init__(self, name: str):
        self.name = name
        self.model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000,
            tol=1e-3
        ) if ML_AVAILABLE else None
        self.scaler = RobustScaler() if ML_AVAILABLE else None
        self.is_fitted = False
        self.samples_seen = 0
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Обучение на новых данных"""
        if not ML_AVAILABLE:
            return
            
        try:
            if not self.is_fitted:
                # Первоначальное обучение
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.is_fitted = True
            else:
                # Онлайн обновление
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y)
                
            self.samples_seen += len(X)
            logger.debug(f"🧠 [ML_{self.name}] Updated with {len(X)} samples, total: {self.samples_seen}")
            
        except Exception as e:
            logger.error(f"❌ [ML_{self.name}] Training error: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if not ML_AVAILABLE or not self.is_fitted:
            return np.zeros(len(X))

        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"❌ [ML_{self.name}] Prediction error: {e}")
            return np.zeros(len(X))


class OnlineLearningClassifier:
    """Модель онлайн-обучения для классификации (направление тренда)"""

    def __init__(self, name: str):
        self.name = name
        self.model = SGDClassifier(
            loss='log_loss',  # Логистическая регрессия для вероятностей
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000,
            tol=1e-3,
            random_state=42
        ) if ML_AVAILABLE else None
        self.scaler = RobustScaler() if ML_AVAILABLE else None
        self.is_fitted = False
        self.samples_seen = 0
        self.classes_ = None

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Обучение на новых данных"""
        if not ML_AVAILABLE:
            return

        try:
            if not self.is_fitted:
                # Первоначальное обучение
                # Для классификатора нужно указать все возможные классы
                # 0 = DOWN, 1 = SIDEWAYS, 2 = UP
                self.classes_ = np.array([0, 1, 2])
                X_scaled = self.scaler.fit_transform(X)
                self.model.partial_fit(X_scaled, y, classes=self.classes_)
                self.is_fitted = True
            else:
                # Онлайн обновление
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y)

            self.samples_seen += len(X)
            logger.debug(f"🧠 [ML_CLASSIFIER_{self.name}] Updated with {len(X)} samples, total: {self.samples_seen}")

        except Exception as e:
            logger.error(f"❌ [ML_CLASSIFIER_{self.name}] Training error: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание класса"""
        if not ML_AVAILABLE or not self.is_fitted:
            return np.ones(len(X))  # Возвращаем SIDEWAYS по умолчанию

        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"❌ [ML_CLASSIFIER_{self.name}] Prediction error: {e}")
            return np.ones(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей классов"""
        if not ML_AVAILABLE or not self.is_fitted:
            # Возвращаем равномерное распределение
            return np.array([[0.33, 0.34, 0.33]] * len(X))

        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            logger.error(f"❌ [ML_CLASSIFIER_{self.name}] Prediction error: {e}")
            return np.array([[0.33, 0.34, 0.33]] * len(X))

class AdvancedMLLearningSystem:
    """Продвинутая система машинного обучения"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path("ml_learning_data")
        self.data_dir.mkdir(exist_ok=True)

        # История данных
        self.market_contexts = deque(maxlen=10000)
        self.trade_outcomes = deque(maxlen=10000)
        self.feature_history = deque(maxlen=5000)

        # 🆕 v4: ML модели разделены по торговым парам (symbol)
        # Структура: {symbol: {model_name: model_instance}}
        self.models_by_symbol = {}  # Регрессоры для каждой пары
        self.classifiers_by_symbol = {}  # Классификаторы для каждой пары

        # Список поддерживаемых моделей (шаблон)
        self.model_template = {
            'pnl_predictor': 'PnL',
            'win_probability': 'WinProb',
            'hold_time_predictor': 'HoldTime',
            'risk_estimator': 'Risk',
            'trend_strength': 'TrendStrength'
        }

        self.classifier_template = {
            'trend_direction': 'TrendDirection'
        }

        # Ensemble модели (для более сложных предсказаний)
        self.ensemble_models = {}

        # Статистика производительности (по символам)
        self.model_performance_by_symbol = {}

        logger.info("🧠 [ADVANCED_ML] System initialized with per-symbol models")
        self._load_historical_data()

    def _get_or_create_models(self, symbol: str) -> Tuple[Dict, Dict]:
        """
        Получает или создает модели для конкретного символа.

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')

        Returns:
            Tuple of (models_dict, classifiers_dict) for the symbol
        """
        # Создаем модели для символа если их еще нет
        if symbol not in self.models_by_symbol:
            logger.info(f"🆕 [ML_INIT] Creating new models for {symbol}")

            # Создаем регрессоры
            self.models_by_symbol[symbol] = {}
            for model_name, display_name in self.model_template.items():
                self.models_by_symbol[symbol][model_name] = OnlineLearningModel(
                    f"{display_name}_{symbol}"
                )

            # Создаем классификаторы
            self.classifiers_by_symbol[symbol] = {}
            for clf_name, display_name in self.classifier_template.items():
                self.classifiers_by_symbol[symbol][clf_name] = OnlineLearningClassifier(
                    f"{display_name}_{symbol}"
                )

            # Инициализируем статистику производительности
            all_model_names = list(self.model_template.keys()) + list(self.classifier_template.keys())
            self.model_performance_by_symbol[symbol] = {name: [] for name in all_model_names}

            logger.info(f"✅ [ML_INIT] Created {len(self.models_by_symbol[symbol])} regressors "
                       f"and {len(self.classifiers_by_symbol[symbol])} classifiers for {symbol}")

        return self.models_by_symbol[symbol], self.classifiers_by_symbol[symbol]

    def extract_features(self, market_context: MarketContext,
                        signal_strength: float,
                        recent_performance: Dict) -> MLFeatures:
        """Извлечение признаков для ML модели"""

        try:
            # 🛡️ Защита: проверяем что market_context правильного типа
            if market_context is None or isinstance(market_context, dict):
                logger.warning(f"⚠️ [FEATURE_EXTRACTION] Invalid market_context type: {type(market_context)}. Returning zero features.")
                return MLFeatures(**{field: 0.0 for field in MLFeatures.__annotations__})

            # Технические признаки
            rsi_momentum = (market_context.rsi_14 - 50) / 50  # Нормализованный RSI
            macd_divergence = market_context.macd - market_context.macd_signal
            volume_surge = max(0, market_context.volume_ratio - 1)  # Превышение среднего объема
            
            # Ценовые моменты
            price_momentum = (market_context.ema_50 - market_context.sma_20) / market_context.sma_20
            volatility_regime = market_context.volatility_percentile / 100
            
            # Рыночный стресс
            market_stress = (100 - market_context.fear_greed_index) / 100
            
            # Тренд
            trend_alignment = market_context.trend_strength * (1 if price_momentum > 0 else -1)
            
            # Поддержка/сопротивление
            support_strength = 1 / (1 + market_context.support_distance)
            
            # Временные факторы
            session_multiplier = {
                'american': 1.2,  # Высокая активность
                'european': 1.0,
                'asian': 0.8      # Низкая активность
            }.get(market_context.session, 1.0)
            
            session_volatility = volatility_regime * session_multiplier
            
            # Дневная производительность
            day_performance = recent_performance.get('today_pnl_pct', 0) / 100
            
            # Мета-признаки
            signal_confluence = signal_strength  # Как много индикаторов согласны
            historical_accuracy = recent_performance.get('recent_accuracy', 0.5)

            # 🆕 НОВЫЕ признаки для распознавания трендов
            # Price velocity - скорость изменения цены (производная)
            price_velocity = (market_context.ema_50 - market_context.sma_20) / market_context.sma_20

            # Price acceleration - ускорение (вторая производная)
            # Используем разницу между краткосрочным и долгосрочным momentum
            price_acceleration = market_context.trend_strength * price_momentum

            # EMA slope - наклон экспоненциальной скользящей средней
            ema_slope = (market_context.ema_50 - market_context.sma_20) / market_context.sma_20

            # Higher highs pattern - паттерн растущих максимумов
            # Если resistance далеко и цена растет -> higher highs
            higher_highs = max(0, min(1,
                (market_context.resistance_distance / 10) * (1 if price_momentum > 0 else 0)
            ))

            # Lower lows pattern - паттерн падающих минимумов
            # Если support далеко и цена падает -> lower lows
            lower_lows = max(0, min(1,
                (market_context.support_distance / 10) * (1 if price_momentum < 0 else 0)
            ))

            # Consolidation score - индикатор флэта/консолидации
            # Высокий когда: низкая волатильность + цена между support/resistance
            consolidation_score = max(0, min(1,
                (1 - volatility_regime) * (1 - abs(price_momentum))
            ))

            return MLFeatures(
                rsi_momentum=rsi_momentum,
                macd_divergence=macd_divergence,
                volume_surge=volume_surge,
                price_momentum=price_momentum,
                volatility_regime=volatility_regime,
                market_stress=market_stress,
                trend_alignment=trend_alignment,
                support_strength=support_strength,
                session_volatility=session_volatility,
                day_performance=day_performance,
                signal_confluence=signal_confluence,
                historical_accuracy=historical_accuracy,
                # Новые признаки для трендов
                price_velocity=price_velocity,
                price_acceleration=price_acceleration,
                ema_slope=ema_slope,
                higher_highs=higher_highs,
                lower_lows=lower_lows,
                consolidation_score=consolidation_score
            )
            
        except Exception as e:
            logger.error(f"❌ [FEATURE_EXTRACTION] Error: {e}")
            # Возвращаем нулевые признаки в случае ошибки
            return MLFeatures(**{field: 0.0 for field in MLFeatures.__annotations__})
    
    async def predict_trade_outcome(self, market_context: MarketContext,
                                  signal_strength: float,
                                  recent_performance: Dict) -> Dict[str, float]:
        """Предсказывает результат сделки перед входом"""

        try:
            # 🛡️ Защита: проверяем валидность market_context
            if market_context is None or isinstance(market_context, dict):
                logger.warning(f"⚠️ [PREDICT_TRADE] Invalid market context - returning neutral predictions")
                return {
                    'expected_pnl_pct': 0.0,
                    'win_probability': 0.5,
                    'expected_hold_time_minutes': 30.0,
                    'risk_score': 0.5,
                    'confidence': 0.0
                }

            # 🆕 v4: Получаем модели для конкретного символа
            symbol = market_context.symbol
            models, classifiers = self._get_or_create_models(symbol)

            # Извлекаем признаки
            features = self.extract_features(market_context, signal_strength, recent_performance)
            feature_array = np.array([list(asdict(features).values())])

            # Получаем предсказания от всех моделей (регрессоров) КОНКРЕТНОЙ ПАРЫ
            predictions = {}

            for name, model in models.items():
                if model.is_fitted:
                    pred = model.predict(feature_array)[0]
                    predictions[name] = float(pred)
                else:
                    predictions[name] = 0.0

            # 🆕 Предсказания от классификаторов КОНКРЕТНОЙ ПАРЫ
            trend_predictions = {}

            for name, classifier in classifiers.items():
                if classifier.is_fitted:
                    # Получаем класс и вероятности
                    pred_class = classifier.predict(feature_array)[0]
                    pred_proba = classifier.predict_proba(feature_array)[0]
                    trend_predictions[name] = {
                        'class': int(pred_class),  # 0=DOWN, 1=SIDEWAYS, 2=UP
                        'probabilities': pred_proba.tolist(),  # [P(DOWN), P(SIDEWAYS), P(UP)]
                        'confidence': float(max(pred_proba))  # Максимальная вероятность
                    }
                else:
                    trend_predictions[name] = {
                        'class': 1,  # SIDEWAYS по умолчанию
                        'probabilities': [0.33, 0.34, 0.33],
                        'confidence': 0.34
                    }

            # Метрики качества предсказания (для данного символа)
            all_samples = sum(model.samples_seen for model in models.values()) + \
                         sum(clf.samples_seen for clf in classifiers.values())
            prediction_confidence = min(1.0, max(0.1, all_samples / 1000))

            # 🆕 Интерпретация направления тренда
            trend_info = trend_predictions.get('trend_direction', {})
            trend_class = trend_info.get('class', 1)
            trend_proba = trend_info.get('probabilities', [0.33, 0.34, 0.33])
            trend_conf = trend_info.get('confidence', 0.34)

            trend_direction_str = ['DOWN ⬇️', 'SIDEWAYS ↔️', 'UP ⬆️'][trend_class]

            result = {
                'expected_pnl_pct': predictions.get('pnl_predictor', 0.0),
                'win_probability': max(0.1, min(0.9, predictions.get('win_probability', 0.5))),
                'expected_hold_time': max(5, predictions.get('hold_time_predictor', 30)),  # минуты
                'risk_score': predictions.get('risk_estimator', 0.5),
                'prediction_confidence': prediction_confidence,
                'feature_importance': self._get_feature_importance(symbol),
                # 🆕 Предсказания тренда
                'trend_direction': trend_direction_str,
                'trend_direction_class': trend_class,
                'trend_probabilities': {
                    'down': trend_proba[0],
                    'sideways': trend_proba[1],
                    'up': trend_proba[2]
                },
                'trend_confidence': trend_conf,
                'trend_strength': max(0, min(1, predictions.get('trend_strength', 0.5)))
            }

            logger.info(f"🎯 [ML_PREDICTION] Trend: {trend_direction_str} ({trend_conf:.0%} conf), "
                       f"Strength: {result['trend_strength']:.2f}, "
                       f"Expected PnL: {result['expected_pnl_pct']:+.2f}%, "
                       f"Win prob: {result['win_probability']:.0%}")

            return result
            
        except Exception as e:
            logger.error(f"❌ [ML_PREDICTION] Error: {e}")
            return {
                'expected_pnl_pct': 0.0,
                'win_probability': 0.5,
                'expected_hold_time': 30.0,
                'risk_score': 0.5,
                'prediction_confidence': 0.0,
                'feature_importance': {}
            }
    
    async def learn_from_trade(self, market_context: MarketContext,
                             trade_outcome: TradeOutcome,
                             signal_strength: float,
                             recent_performance: Dict):
        """Обучение на завершенной сделке"""

        try:
            # 🛡️ Защита: проверяем валидность market_context
            if market_context is None or isinstance(market_context, dict):
                logger.warning(f"⚠️ [LEARN_FROM_TRADE] Invalid market context - skipping learning")
                return

            # 🆕 v4: Получаем модели для конкретного символа
            symbol = market_context.symbol
            models, classifiers = self._get_or_create_models(symbol)

            # Извлекаем признаки
            features = self.extract_features(market_context, signal_strength, recent_performance)
            feature_array = np.array([list(asdict(features).values())])

            # 🔧 ИСПРАВЛЕНИЕ: Целевые переменные В ПРОЦЕНТАХ (не абсолютные значения!)
            # Clip extreme values to prevent model from learning outliers
            pnl_pct = np.clip(trade_outcome.pnl_pct, -20, 20)  # Ограничим +/-20%

            # Нормализуем hold_time в часах (0-24 часа)
            hold_time_hours = np.clip(trade_outcome.hold_time_minutes / 60, 0, 24)

            # MAE в процентах от entry price (не абсолютное значение!)
            # Используем pnl_pct как прокси для риска
            risk_pct = abs(pnl_pct) if trade_outcome.pnl < 0 else 0

            targets = {
                'pnl_predictor': pnl_pct,
                'win_probability': 1.0 if trade_outcome.pnl > 0 else 0.0,
                'hold_time_predictor': hold_time_hours,
                'risk_estimator': np.clip(risk_pct, 0, 10),  # Максимум 10% риск
                'trend_strength': np.clip(abs(pnl_pct) / 10, 0, 1)  # 🆕 Сила тренда (0-1, нормализованная)
            }

            # Обучаем все модели (регрессоры) КОНКРЕТНОЙ ПАРЫ
            for name, target in targets.items():
                if name in models:
                    models[name].partial_fit(feature_array, np.array([target]))

            # 🆕 Обучаем классификаторы КОНКРЕТНОЙ ПАРЫ
            # Определяем направление тренда на основе PnL
            if pnl_pct > 1.0:  # Значительный рост
                trend_class = 2  # UP
            elif pnl_pct < -1.0:  # Значительное падение
                trend_class = 0  # DOWN
            else:  # Минимальное движение
                trend_class = 1  # SIDEWAYS

            # Обучаем классификатор тренда
            if 'trend_direction' in classifiers:
                classifiers['trend_direction'].partial_fit(
                    feature_array,
                    np.array([trend_class])
                )

            # Сохраняем данные
            self.market_contexts.append(market_context)
            self.trade_outcomes.append(trade_outcome)
            self.feature_history.append(features)

            # Периодически оцениваем качество моделей
            if len(self.trade_outcomes) % 50 == 0:
                await self._evaluate_model_performance()

            logger.info(f"🧠 [ML_LEARNING] Learned from {symbol} trade: {trade_outcome.pnl_pct:+.2f}% PnL")

        except Exception as e:
            logger.error(f"❌ [ML_LEARNING] Error: {e}")
    
    async def get_intelligent_recommendations(self, current_market: MarketContext,
                                            recent_performance: Dict) -> Dict[str, Any]:
        """Получить рекомендации от AI системы"""

        try:
            if current_market is None or isinstance(current_market, dict):
                logger.warning(f"⚠️ [ML_RECOMMENDATIONS] Invalid market context type: {type(current_market)}")
                return {'confidence': 0.0, 'recommendations': []}

            # 🆕 v4: Получаем модели для конкретного символа
            symbol = current_market.symbol
            models, classifiers = self._get_or_create_models(symbol)

            # 🛡️ Защита: проверяем наличие обученных моделей
            if not models['pnl_predictor'].is_fitted:
                return {'confidence': 0.0, 'recommendations': []}

            # Анализируем текущие рыночные условия
            features = self.extract_features(current_market, 1.0, recent_performance)
            feature_array = np.array([list(asdict(features).values())])

            # Получаем предсказания от моделей КОНКРЕТНОЙ ПАРЫ
            expected_pnl = models['pnl_predictor'].predict(feature_array)[0]
            win_prob = models['win_probability'].predict(feature_array)[0]
            risk_score = models['risk_estimator'].predict(feature_array)[0]
            
            # Генерируем рекомендации
            recommendations = []
            
            if expected_pnl > 0.5 and win_prob > 0.6:
                recommendations.append({
                    'action': 'increase_position_size',
                    'confidence': min(0.9, win_prob),
                    'reason': f'High win probability ({win_prob:.1%}) and positive expected return'
                })
            
            if risk_score > 0.7:
                recommendations.append({
                    'action': 'tighten_stop_loss',
                    'confidence': 0.8,
                    'reason': f'High risk environment detected (score: {risk_score:.2f})'
                })
            
            if features.volatility_regime > 0.8:
                recommendations.append({
                    'action': 'reduce_exposure',
                    'confidence': 0.7,
                    'reason': 'High volatility regime - reduce risk'
                })
            
            if features.trend_alignment > 0.5 and features.support_strength > 0.7:
                recommendations.append({
                    'action': 'extend_targets',
                    'confidence': 0.6,
                    'reason': 'Strong trend with solid support - ride the momentum'
                })
            
            confidence = min(1.0, sum(model.samples_seen for model in self.models.values()) / 2000)
            
            return {
                'confidence': confidence,
                'expected_pnl': expected_pnl,
                'win_probability': win_prob,
                'risk_score': risk_score,
                'recommendations': recommendations,
                'market_regime': current_market.market_regime,
                'feature_summary': {
                    'trend_strength': features.trend_alignment,
                    'volatility': features.volatility_regime,
                    'market_stress': features.market_stress
                }
            }
            
        except Exception as e:
            logger.error(f"❌ [ML_RECOMMENDATIONS] Error: {e}")
            return {'confidence': 0.0, 'recommendations': []}
    
    def _get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Получить важность признаков для конкретного символа"""
        try:
            # Получаем модели для символа
            if symbol not in self.models_by_symbol:
                return {}

            models = self.models_by_symbol[symbol]

            if not ML_AVAILABLE or not models['pnl_predictor'].is_fitted:
                return {}

            # Для SGD модели используем коэффициенты как важность
            coef = models['pnl_predictor'].model.coef_
            feature_names = list(MLFeatures.__annotations__.keys())

            importance = {}
            for i, name in enumerate(feature_names):
                if i < len(coef):
                    importance[name] = abs(float(coef[i]))

            return importance

        except Exception as e:
            logger.error(f"❌ [FEATURE_IMPORTANCE] Error: {e}")
            return {}
    
    async def _evaluate_model_performance(self):
        """Оценка производительности моделей (для каждого символа отдельно)"""
        try:
            if len(self.trade_outcomes) < 30:
                return

            # Берем последние N сделок для оценки
            recent_outcomes = list(self.trade_outcomes)[-50:]
            recent_features = list(self.feature_history)[-50:]

            if len(recent_features) != len(recent_outcomes):
                return

            # Группируем сделки по символам
            from collections import defaultdict
            outcomes_by_symbol = defaultdict(list)
            features_by_symbol = defaultdict(list)

            for outcome, feature in zip(recent_outcomes, recent_features):
                # Получаем символ из trade_id (format: "symbol_timestamp_...")
                if '_' in outcome.trade_id:
                    symbol = outcome.trade_id.split('_')[0]
                    outcomes_by_symbol[symbol].append(outcome)
                    features_by_symbol[symbol].append(feature)

            # Оцениваем модели для каждого символа
            for symbol in outcomes_by_symbol.keys():
                if symbol not in self.models_by_symbol:
                    continue

                symbol_outcomes = outcomes_by_symbol[symbol]
                symbol_features = features_by_symbol[symbol]

                if len(symbol_features) < 10:  # Минимум 10 сделок для оценки
                    continue

                # Создаем массивы для оценки
                X = np.array([list(asdict(f).values()) for f in symbol_features])

                models = self.models_by_symbol[symbol]

                # Оцениваем каждую модель
                for name, model in models.items():
                    if not model.is_fitted:
                        continue

                    if name == 'pnl_predictor':
                        y_true = [outcome.pnl_pct for outcome in symbol_outcomes]
                    elif name == 'win_probability':
                        y_true = [1.0 if outcome.pnl > 0 else 0.0 for outcome in symbol_outcomes]
                    elif name == 'hold_time_predictor':
                        y_true = [outcome.hold_time_minutes for outcome in symbol_outcomes]
                    else:
                        continue

                    y_pred = model.predict(X)
                    mse = mean_squared_error(y_true, y_pred)

                    perf_key = f"{symbol}_{name}"
                    self.model_performance[perf_key].append({
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'mse': float(mse),
                        'samples': len(y_true)
                    })

                    logger.info(f"📊 [ML_PERFORMANCE] {symbol} {name}: MSE = {mse:.4f}")

        except Exception as e:
            logger.error(f"❌ [ML_EVALUATION] Error: {e}")
    
    def _validate_model_sanity(self, model_name: str, model: 'OnlineLearningModel') -> bool:
        """
        🛡️ Проверяет что модель дает вменяемые предсказания
        Отклоняет модели которые предсказывают абсурдные значения
        """
        try:
            if not model.is_fitted:
                return True  # Новая модель - ок

            # Создаем тестовый вектор с нормальными значениями
            test_features = np.array([[0.5] * 12])  # 12 features, все средние значения

            prediction = model.predict(test_features)[0]

            # Проверяем диапазоны в зависимости от типа модели
            if model_name == 'pnl_predictor':
                # PnL должен быть в пределах +/-50% (даже +/-20% слишком много для одного trade)
                if abs(prediction) > 50:
                    logger.warning(f"⚠️ [VALIDATION] Model '{model_name}' predicts absurd PnL: {prediction:.2f}% (expected +/-50%)")
                    return False

            elif model_name == 'win_probability':
                # Вероятность должна быть 0-1
                if prediction < 0 or prediction > 1.5:  # Небольшой запас на случай экстраполяции
                    logger.warning(f"⚠️ [VALIDATION] Model '{model_name}' predicts absurd probability: {prediction:.4f} (expected 0-1)")
                    return False

            elif model_name == 'hold_time_predictor':
                # Время удержания в часах, должно быть разумным (0-48 часов)
                if prediction < 0 or prediction > 48:
                    logger.warning(f"⚠️ [VALIDATION] Model '{model_name}' predicts absurd hold time: {prediction:.2f}h (expected 0-48h)")
                    return False

            elif model_name == 'risk_estimator':
                # Риск должен быть 0-20%
                if prediction < 0 or prediction > 20:
                    logger.warning(f"⚠️ [VALIDATION] Model '{model_name}' predicts absurd risk: {prediction:.2f}% (expected 0-20%)")
                    return False

            return True

        except Exception as e:
            logger.warning(f"⚠️ [VALIDATION] Error validating model '{model_name}': {e}")
            return False

    def _warm_start_training(self, historical_contexts: list, historical_outcomes: list):
        """🔥 WARM START: Переобучает модели на исторических сделках"""
        try:
            if not ML_AVAILABLE:
                return

            # Фильтруем только завершенные сделки с ненулевым PnL
            valid_pairs = []
            for ctx, outcome in zip(historical_contexts, historical_outcomes):
                # Пропускаем сделки с 0% PnL (незавершенные)
                if abs(outcome.pnl_pct) > 0.01:  # Минимум 0.01% движения
                    valid_pairs.append((ctx, outcome))

            if len(valid_pairs) < 5:
                logger.info(f"⚠️ [WARM_START] Not enough valid historical trades ({len(valid_pairs)}) - skipping warm start")
                return

            # 🆕 v4: Группируем сделки по символам
            trades_by_symbol = {}
            for ctx, outcome in valid_pairs:
                symbol = ctx.symbol
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append((ctx, outcome))

            logger.info(f"🔥 [WARM_START] Pretraining on {len(valid_pairs)} historical trades across {len(trades_by_symbol)} symbols...")

            # Переобучаем модели для каждого символа отдельно
            for symbol, symbol_trades in trades_by_symbol.items():
                # Получаем или создаем модели для символа
                models, classifiers = self._get_or_create_models(symbol)

                logger.info(f"  📊 [WARM_START] Training {symbol}: {len(symbol_trades)} trades")

                for ctx, outcome in symbol_trades:
                    try:
                        # Создаем признаки
                        features = self.extract_features(ctx, 1.0, {})
                        feature_array = np.array([list(asdict(features).values())])

                        # Clip target values (как в learn_from_trade)
                        pnl_pct = np.clip(outcome.pnl_pct, -20, 20)
                        hold_time_hours = np.clip(outcome.hold_time_minutes / 60, 0, 24)
                        risk_pct = abs(pnl_pct) if outcome.pnl < 0 else 0

                        targets = {
                            'pnl_predictor': pnl_pct,
                            'win_probability': 1.0 if outcome.pnl > 0 else 0.0,
                            'hold_time_predictor': hold_time_hours,
                            'risk_estimator': np.clip(risk_pct, 0, 10),
                            'trend_strength': np.clip(abs(pnl_pct) / 10, 0, 1)
                        }

                        # Обучаем регрессоры КОНКРЕТНОГО СИМВОЛА
                        for name, target in targets.items():
                            if name in models:
                                models[name].partial_fit(feature_array, np.array([target]))

                        # Обучаем классификаторы КОНКРЕТНОГО СИМВОЛА
                        if pnl_pct > 1.0:
                            trend_class = 2  # UP
                        elif pnl_pct < -1.0:
                            trend_class = 0  # DOWN
                        else:
                            trend_class = 1  # SIDEWAYS

                        if 'trend_direction' in classifiers:
                            classifiers['trend_direction'].partial_fit(
                                feature_array,
                                np.array([trend_class])
                            )

                    except Exception as e:
                        logger.debug(f"⚠️ [WARM_START] Failed to train on one trade: {e}")

                # Логируем результат для символа
                total_samples = sum(model.samples_seen for model in models.values())
                logger.info(f"  ✅ [WARM_START] {symbol}: {total_samples} total samples")

            logger.info(f"🧠 [WARM_START] Completed pretraining for all symbols")

            # Сохраняем обновленные модели
            self.save_data()

        except Exception as e:
            logger.error(f"❌ [WARM_START] Error during warm start training: {e}")

    def _load_historical_data(self):
        """Загружает исторические данные"""
        try:
            # Загружаем сохраненные данные если есть
            contexts_file = self.data_dir / "market_contexts.json"
            outcomes_file = self.data_dir / "trade_outcomes.json"

            historical_contexts = []
            historical_outcomes = []

            if contexts_file.exists() and outcomes_file.exists():
                with open(contexts_file, 'r') as f:
                    contexts_data = json.load(f)

                with open(outcomes_file, 'r') as f:
                    outcomes_data = json.load(f)

                # Восстанавливаем объекты из JSON
                for ctx_dict in contexts_data:
                    try:
                        # Преобразуем ISO строки обратно в datetime
                        if 'timestamp' in ctx_dict and isinstance(ctx_dict['timestamp'], str):
                            ctx_dict['timestamp'] = datetime.fromisoformat(ctx_dict['timestamp'].replace('Z', '+00:00'))
                        historical_contexts.append(MarketContext(**ctx_dict))
                    except Exception as e:
                        logger.debug(f"⚠️ Failed to restore context: {e}")

                for outcome_dict in outcomes_data:
                    try:
                        historical_outcomes.append(TradeOutcome(**outcome_dict))
                    except Exception as e:
                        logger.debug(f"⚠️ Failed to restore outcome: {e}")

                logger.info(f"🧠 [ML_LOAD] Loaded {len(historical_contexts)} historical contexts, {len(historical_outcomes)} outcomes")

            # 🆕 v4: Загружаем ML модели для каждого символа
            if ML_AVAILABLE:
                total_models_loaded = 0
                total_classifiers_loaded = 0
                incompatible_files = []

                # Сканируем директорию на наличие файлов моделей
                # Паттерн: {SYMBOL}_{model_name}_model.pkl или {SYMBOL}_{model_name}_classifier.pkl
                model_files = list(self.data_dir.glob("*_model.pkl"))
                classifier_files = list(self.data_dir.glob("*_classifier.pkl"))

                # Собираем уникальные символы из имен файлов
                symbols_found = set()
                for file in model_files + classifier_files:
                    # Извлекаем символ из имени файла
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        symbol = parts[0]  # Первая часть - это символ
                        symbols_found.add(symbol)

                if not symbols_found:
                    logger.info(f"📚 [ML_LOAD] No saved models found - starting from scratch")
                else:
                    logger.info(f"🔍 [ML_LOAD] Found models for {len(symbols_found)} symbols: {', '.join(sorted(symbols_found))}")

                    # Загружаем модели для каждого символа
                    for symbol in sorted(symbols_found):
                        symbol_models_loaded = 0
                        symbol_classifiers_loaded = 0

                        # Создаем пустые модели для символа
                        models, classifiers = self._get_or_create_models(symbol)

                        # Загружаем регрессоры для символа
                        for model_name in self.model_template.keys():
                            model_file = self.data_dir / f"{symbol}_{model_name}_model.pkl"
                            scaler_file = self.data_dir / f"{symbol}_{model_name}_scaler.pkl"
                            metadata_file = self.data_dir / f"{symbol}_{model_name}_metadata.json"

                            if model_file.exists() and scaler_file.exists():
                                try:
                                    # Проверяем версию модели
                                    model_version = None
                                    if metadata_file.exists():
                                        with open(metadata_file, 'r') as f:
                                            metadata = json.load(f)
                                            model_version = metadata.get('model_version', 1)

                                    # Проверяем совместимость версии
                                    if model_version != MODEL_VERSION:
                                        logger.warning(f"⚠️ [ML_LOAD] {symbol}/{model_name} version mismatch: v{model_version} != v{MODEL_VERSION}")
                                        incompatible_files.append(str(model_file))
                                        continue

                                    # Загружаем модель
                                    models[model_name].model = joblib.load(model_file)
                                    models[model_name].scaler = joblib.load(scaler_file)
                                    models[model_name].is_fitted = True

                                    # Восстанавливаем метаданные
                                    if metadata_file.exists():
                                        with open(metadata_file, 'r') as f:
                                            metadata = json.load(f)
                                            models[model_name].samples_seen = metadata.get('samples_seen', 0)

                                    # Валидация
                                    if not self._validate_model_sanity(model_name, models[model_name]):
                                        logger.warning(f"⚠️ [ML_LOAD] {symbol}/{model_name} failed sanity check")
                                        models[model_name].is_fitted = False
                                        incompatible_files.append(str(model_file))
                                        continue

                                    symbol_models_loaded += 1

                                except Exception as e:
                                    logger.warning(f"⚠️ [ML_LOAD] Failed to load {symbol}/{model_name}: {e}")
                                    incompatible_files.append(str(model_file))

                        # Загружаем классификаторы для символа
                        for clf_name in self.classifier_template.keys():
                            model_file = self.data_dir / f"{symbol}_{clf_name}_classifier.pkl"
                            scaler_file = self.data_dir / f"{symbol}_{clf_name}_scaler.pkl"
                            metadata_file = self.data_dir / f"{symbol}_{clf_name}_metadata.json"

                            if model_file.exists() and scaler_file.exists():
                                try:
                                    # Проверяем версию модели
                                    model_version = None
                                    if metadata_file.exists():
                                        with open(metadata_file, 'r') as f:
                                            metadata = json.load(f)
                                            model_version = metadata.get('model_version', 1)

                                    # Проверяем совместимость версии
                                    if model_version != MODEL_VERSION:
                                        logger.warning(f"⚠️ [ML_LOAD] {symbol}/{clf_name} classifier version mismatch")
                                        incompatible_files.append(str(model_file))
                                        continue

                                    # Загружаем классификатор
                                    classifiers[clf_name].model = joblib.load(model_file)
                                    classifiers[clf_name].scaler = joblib.load(scaler_file)
                                    classifiers[clf_name].is_fitted = True

                                    # Восстанавливаем метаданные
                                    if metadata_file.exists():
                                        with open(metadata_file, 'r') as f:
                                            metadata = json.load(f)
                                            classifiers[clf_name].samples_seen = metadata.get('samples_seen', 0)
                                            classifiers[clf_name].classes_ = np.array(metadata.get('classes', [0, 1, 2]))

                                    symbol_classifiers_loaded += 1

                                except Exception as e:
                                    logger.warning(f"⚠️ [ML_LOAD] Failed to load {symbol}/{clf_name}: {e}")
                                    incompatible_files.append(str(model_file))

                        total_models_loaded += symbol_models_loaded
                        total_classifiers_loaded += symbol_classifiers_loaded

                        if symbol_models_loaded > 0 or symbol_classifiers_loaded > 0:
                            logger.info(f"  ✅ [ML_LOAD] {symbol}: {symbol_models_loaded} regressors, {symbol_classifiers_loaded} classifiers")

                    # Удаляем несовместимые файлы
                    if incompatible_files:
                        logger.warning(f"🗑️ [ML_CLEANUP] Deleting {len(incompatible_files)} incompatible model files")
                        for file_path in incompatible_files:
                            try:
                                Path(file_path).unlink(missing_ok=True)
                                # Также удаляем связанные файлы
                                Path(file_path.replace('_model.pkl', '_scaler.pkl')).unlink(missing_ok=True)
                                Path(file_path.replace('_model.pkl', '_metadata.json')).unlink(missing_ok=True)
                                Path(file_path.replace('_classifier.pkl', '_scaler.pkl')).unlink(missing_ok=True)
                                Path(file_path.replace('_classifier.pkl', '_metadata.json')).unlink(missing_ok=True)
                            except Exception as e:
                                logger.debug(f"Failed to delete {file_path}: {e}")

                    logger.info(f"🧠 [ML_LOAD] Successfully loaded {total_models_loaded} regressors and {total_classifiers_loaded} classifiers across {len(symbols_found)} symbols")

            # Восстанавливаем данные в память
            if len(historical_contexts) > 0:
                self.market_contexts.extend(historical_contexts)
            if len(historical_outcomes) > 0:
                self.trade_outcomes.extend(historical_outcomes)

            # 🔥 WARM START: Переобучаем модели на исторических данных
            if len(historical_contexts) > 0 and len(historical_outcomes) > 0:
                self._warm_start_training(historical_contexts, historical_outcomes)

        except Exception as e:
            logger.error(f"❌ [ML_LOAD] Error loading historical data: {e}")
    
    def save_data(self):
        """Сохраняет данные ML системы"""
        try:
            # Сохраняем контексты рынка
            contexts_data = []
            for context in self.market_contexts:
                contexts_data.append(asdict(context))

            with open(self.data_dir / "market_contexts.json", 'w') as f:
                json.dump(contexts_data, f, indent=2, default=str)

            # Сохраняем результаты сделок
            outcomes_data = []
            for outcome in self.trade_outcomes:
                outcomes_data.append(asdict(outcome))

            with open(self.data_dir / "trade_outcomes.json", 'w') as f:
                json.dump(outcomes_data, f, indent=2, default=str)

            # 🆕 v4: Сохраняем модели для каждого символа отдельно
            if ML_AVAILABLE:
                total_models_saved = 0
                total_classifiers_saved = 0

                # Сохраняем модели всех символов
                for symbol in self.models_by_symbol.keys():
                    models = self.models_by_symbol[symbol]
                    classifiers = self.classifiers_by_symbol[symbol]

                    symbol_models_saved = 0
                    symbol_classifiers_saved = 0

                    # Сохраняем регрессоры для символа
                    for name, model in models.items():
                        if model.is_fitted:
                            # Формируем имена файлов с префиксом символа
                            model_file = f"{symbol}_{name}_model.pkl"
                            scaler_file = f"{symbol}_{name}_scaler.pkl"
                            metadata_file = f"{symbol}_{name}_metadata.json"

                            # Сохраняем модель и скейлер
                            joblib.dump(model.model, self.data_dir / model_file)
                            joblib.dump(model.scaler, self.data_dir / scaler_file)

                            # Сохраняем метаданные
                            metadata = {
                                'symbol': symbol,
                                'model_name': name,
                                'samples_seen': model.samples_seen,
                                'is_fitted': model.is_fitted,
                                'saved_at': datetime.now(timezone.utc).isoformat(),
                                'model_version': MODEL_VERSION
                            }
                            with open(self.data_dir / metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)

                            symbol_models_saved += 1

                    # Сохраняем классификаторы для символа
                    for name, classifier in classifiers.items():
                        if classifier.is_fitted:
                            # Формируем имена файлов с префиксом символа
                            model_file = f"{symbol}_{name}_classifier.pkl"
                            scaler_file = f"{symbol}_{name}_scaler.pkl"
                            metadata_file = f"{symbol}_{name}_metadata.json"

                            # Сохраняем классификатор и скейлер
                            joblib.dump(classifier.model, self.data_dir / model_file)
                            joblib.dump(classifier.scaler, self.data_dir / scaler_file)

                            # Сохраняем метаданные
                            metadata = {
                                'symbol': symbol,
                                'model_name': name,
                                'samples_seen': classifier.samples_seen,
                                'is_fitted': classifier.is_fitted,
                                'classes': classifier.classes_.tolist() if classifier.classes_ is not None else [0, 1, 2],
                                'saved_at': datetime.now(timezone.utc).isoformat(),
                                'model_version': MODEL_VERSION
                            }
                            with open(self.data_dir / metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)

                            symbol_classifiers_saved += 1

                    total_models_saved += symbol_models_saved
                    total_classifiers_saved += symbol_classifiers_saved

                    if symbol_models_saved > 0 or symbol_classifiers_saved > 0:
                        logger.info(f"  💾 [ML_SAVE] {symbol}: {symbol_models_saved} regressors, {symbol_classifiers_saved} classifiers")

                logger.info(f"💾 [ML_SAVE] Saved {total_models_saved} regressors and {total_classifiers_saved} classifiers across {len(self.models_by_symbol)} symbols")

            logger.info(f"💾 [ML_SAVE] Saved ML data: {len(self.market_contexts)} contexts, "
                       f"{len(self.trade_outcomes)} outcomes")

        except Exception as e:
            logger.error(f"❌ [ML_SAVE] Error: {e}")