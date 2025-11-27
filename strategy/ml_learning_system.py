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
# v2: Added target clipping (±20% PnL, 0-24h hold_time, 0-10% risk)
# v3: Added trend direction classifier and trend strength predictor
MODEL_VERSION = 3

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
        
        # ML модели (регрессоры)
        self.models = {
            'pnl_predictor': OnlineLearningModel('PnL'),
            'win_probability': OnlineLearningModel('WinProb'),
            'hold_time_predictor': OnlineLearningModel('HoldTime'),
            'risk_estimator': OnlineLearningModel('Risk'),
            'trend_strength': OnlineLearningModel('TrendStrength')  # 🆕 Сила тренда (0-1)
        }

        # 🆕 ML классификаторы
        self.classifiers = {
            'trend_direction': OnlineLearningClassifier('TrendDirection')  # 0=DOWN, 1=SIDEWAYS, 2=UP
        }
        
        # Ensemble модели (для более сложных предсказаний)
        self.ensemble_models = {}

        # Статистика производительности
        all_model_names = list(self.models.keys()) + list(self.classifiers.keys())
        self.model_performance = {name: [] for name in all_model_names}
        
        logger.info("🧠 [ADVANCED_ML] System initialized")
        self._load_historical_data()
    
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

            # Извлекаем признаки
            features = self.extract_features(market_context, signal_strength, recent_performance)
            feature_array = np.array([list(asdict(features).values())])
            
            # Получаем предсказания от всех моделей (регрессоров)
            predictions = {}

            for name, model in self.models.items():
                if model.is_fitted:
                    pred = model.predict(feature_array)[0]
                    predictions[name] = float(pred)
                else:
                    predictions[name] = 0.0

            # 🆕 Предсказания от классификаторов
            trend_predictions = {}

            for name, classifier in self.classifiers.items():
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
            
            # Метрики качества предсказания
            all_samples = sum(model.samples_seen for model in self.models.values()) + \
                         sum(clf.samples_seen for clf in self.classifiers.values())
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
                'feature_importance': self._get_feature_importance(),
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

            # Извлекаем признаки
            features = self.extract_features(market_context, signal_strength, recent_performance)
            feature_array = np.array([list(asdict(features).values())])
            
            # 🔧 ИСПРАВЛЕНИЕ: Целевые переменные В ПРОЦЕНТАХ (не абсолютные значения!)
            # Clip extreme values to prevent model from learning outliers
            pnl_pct = np.clip(trade_outcome.pnl_pct, -20, 20)  # Ограничим ±20%

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

            # Обучаем все модели (регрессоры)
            for name, target in targets.items():
                if name in self.models:
                    self.models[name].partial_fit(feature_array, np.array([target]))

            # 🆕 Обучаем классификаторы
            # Определяем направление тренда на основе PnL
            if pnl_pct > 1.0:  # Значительный рост
                trend_class = 2  # UP
            elif pnl_pct < -1.0:  # Значительное падение
                trend_class = 0  # DOWN
            else:  # Минимальное движение
                trend_class = 1  # SIDEWAYS

            # Обучаем классификатор тренда
            if 'trend_direction' in self.classifiers:
                self.classifiers['trend_direction'].partial_fit(
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
            
            logger.info(f"🧠 [ML_LEARNING] Learned from trade: {trade_outcome.pnl_pct:+.2f}% PnL")
            
        except Exception as e:
            logger.error(f"❌ [ML_LEARNING] Error: {e}")
    
    async def get_intelligent_recommendations(self, current_market: MarketContext,
                                            recent_performance: Dict) -> Dict[str, Any]:
        """Получить рекомендации от AI системы"""

        try:
            # 🛡️ Защита: проверяем наличие обученных моделей и валидность контекста
            if not self.models['pnl_predictor'].is_fitted:
                return {'confidence': 0.0, 'recommendations': []}

            if current_market is None or isinstance(current_market, dict):
                logger.warning(f"⚠️ [ML_RECOMMENDATIONS] Invalid market context type: {type(current_market)}")
                return {'confidence': 0.0, 'recommendations': []}

            # Анализируем текущие рыночные условия
            features = self.extract_features(current_market, 1.0, recent_performance)
            feature_array = np.array([list(asdict(features).values())])
            
            # Получаем предсказания
            expected_pnl = self.models['pnl_predictor'].predict(feature_array)[0]
            win_prob = self.models['win_probability'].predict(feature_array)[0]
            risk_score = self.models['risk_estimator'].predict(feature_array)[0]
            
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
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Получить важность признаков"""
        try:
            if not ML_AVAILABLE or not self.models['pnl_predictor'].is_fitted:
                return {}
            
            # Для SGD модели используем коэффициенты как важность
            coef = self.models['pnl_predictor'].model.coef_
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
        """Оценка производительности моделей"""
        try:
            if len(self.trade_outcomes) < 30:
                return
            
            # Берем последние N сделок для оценки
            recent_outcomes = list(self.trade_outcomes)[-50:]
            recent_features = list(self.feature_history)[-50:]
            
            if len(recent_features) != len(recent_outcomes):
                return
            
            # Создаем массивы для оценки
            X = np.array([list(asdict(f).values()) for f in recent_features])
            
            # Оцениваем каждую модель
            for name, model in self.models.items():
                if not model.is_fitted:
                    continue
                
                if name == 'pnl_predictor':
                    y_true = [outcome.pnl_pct for outcome in recent_outcomes]
                elif name == 'win_probability':
                    y_true = [1.0 if outcome.pnl > 0 else 0.0 for outcome in recent_outcomes]
                elif name == 'hold_time_predictor':
                    y_true = [outcome.hold_time_minutes for outcome in recent_outcomes]
                else:
                    continue
                
                y_pred = model.predict(X)
                mse = mean_squared_error(y_true, y_pred)
                
                self.model_performance[name].append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'mse': float(mse),
                    'samples': len(y_true)
                })
                
                logger.info(f"📊 [ML_PERFORMANCE] {name}: MSE = {mse:.4f}")
            
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
                # PnL должен быть в пределах ±50% (даже ±20% слишком много для одного trade)
                if abs(prediction) > 50:
                    logger.warning(f"⚠️ [VALIDATION] Model '{model_name}' predicts absurd PnL: {prediction:.2f}% (expected ±50%)")
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

            logger.info(f"🔥 [WARM_START] Pretraining on {len(valid_pairs)} historical trades...")

            # Переобучаем модели на исторических данных
            for ctx, outcome in valid_pairs:
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
                        'trend_strength': np.clip(abs(pnl_pct) / 10, 0, 1)  # 🆕 Сила тренда
                    }

                    # Обучаем регрессоры
                    for name, target in targets.items():
                        if name in self.models:
                            self.models[name].partial_fit(feature_array, np.array([target]))

                    # 🆕 Обучаем классификаторы
                    # Определяем направление тренда
                    if pnl_pct > 1.0:
                        trend_class = 2  # UP
                    elif pnl_pct < -1.0:
                        trend_class = 0  # DOWN
                    else:
                        trend_class = 1  # SIDEWAYS

                    if 'trend_direction' in self.classifiers:
                        self.classifiers['trend_direction'].partial_fit(
                            feature_array,
                            np.array([trend_class])
                        )

                except Exception as e:
                    logger.debug(f"⚠️ [WARM_START] Failed to train on one trade: {e}")

            # Логируем результат
            total_samples = sum(model.samples_seen for model in self.models.values())
            logger.info(f"✅ [WARM_START] Pretrained on {len(valid_pairs)} historical trades")
            logger.info(f"🧠 [WARM_START] Total samples across all models: {total_samples}")

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

            # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Загружаем сохраненные ML модели
            if ML_AVAILABLE:
                models_loaded = 0
                classifiers_loaded = 0
                incompatible_models = []

                # Загружаем регрессоры
                for name, model in self.models.items():
                    model_file = self.data_dir / f"{name}_model.pkl"
                    scaler_file = self.data_dir / f"{name}_scaler.pkl"
                    metadata_file = self.data_dir / f"{name}_metadata.json"

                    if model_file.exists() and scaler_file.exists():
                        try:
                            # 🔢 Проверяем версию модели
                            model_version = None
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    model_version = metadata.get('model_version', 1)  # default v1 для старых моделей

                            # 🛡️ ЗАЩИТА: Проверяем совместимость версии
                            if model_version != MODEL_VERSION:
                                logger.warning(f"⚠️ [ML_LOAD] Model '{name}' version mismatch: saved v{model_version}, current v{MODEL_VERSION}")
                                incompatible_models.append(name)
                                continue

                            # Загружаем модель
                            model.model = joblib.load(model_file)
                            model.scaler = joblib.load(scaler_file)
                            model.is_fitted = True

                            # Восстанавливаем samples_seen из метаданных
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    model.samples_seen = metadata.get('samples_seen', 0)

                            # 🛡️ ВАЛИДАЦИЯ: Проверяем что модель дает вменяемые предсказания
                            if not self._validate_model_sanity(name, model):
                                logger.warning(f"⚠️ [ML_LOAD] Model '{name}' failed sanity check - discarding")
                                incompatible_models.append(name)
                                model.is_fitted = False
                                continue

                            models_loaded += 1
                            logger.info(f"✅ [ML_LOAD] Loaded regressor '{name}': {model.samples_seen} samples seen")

                        except Exception as e:
                            logger.warning(f"⚠️ [ML_LOAD] Failed to load model '{name}': {e}")
                            incompatible_models.append(name)

                # 🆕 Загружаем классификаторы
                for name, classifier in self.classifiers.items():
                    model_file = self.data_dir / f"{name}_classifier.pkl"
                    scaler_file = self.data_dir / f"{name}_scaler.pkl"
                    metadata_file = self.data_dir / f"{name}_metadata.json"

                    if model_file.exists() and scaler_file.exists():
                        try:
                            # 🔢 Проверяем версию модели
                            model_version = None
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    model_version = metadata.get('model_version', 1)

                            # 🛡️ ЗАЩИТА: Проверяем совместимость версии
                            if model_version != MODEL_VERSION:
                                logger.warning(f"⚠️ [ML_LOAD] Classifier '{name}' version mismatch: saved v{model_version}, current v{MODEL_VERSION}")
                                incompatible_models.append(name)
                                continue

                            # Загружаем классификатор
                            classifier.model = joblib.load(model_file)
                            classifier.scaler = joblib.load(scaler_file)
                            classifier.is_fitted = True

                            # Восстанавливаем метаданные
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    classifier.samples_seen = metadata.get('samples_seen', 0)
                                    classifier.classes_ = np.array(metadata.get('classes', [0, 1, 2]))

                            classifiers_loaded += 1
                            logger.info(f"✅ [ML_LOAD] Loaded classifier '{name}': {classifier.samples_seen} samples seen")

                        except Exception as e:
                            logger.warning(f"⚠️ [ML_LOAD] Failed to load classifier '{name}': {e}")
                            incompatible_models.append(name)

                # 🗑️ Удаляем несовместимые модели
                if incompatible_models:
                    logger.warning(f"🗑️ [ML_CLEANUP] Deleting {len(incompatible_models)} incompatible models: {incompatible_models}")
                    for name in incompatible_models:
                        try:
                            (self.data_dir / f"{name}_model.pkl").unlink(missing_ok=True)
                            (self.data_dir / f"{name}_classifier.pkl").unlink(missing_ok=True)
                            (self.data_dir / f"{name}_scaler.pkl").unlink(missing_ok=True)
                            (self.data_dir / f"{name}_metadata.json").unlink(missing_ok=True)
                        except Exception as e:
                            logger.debug(f"Failed to delete old model files for '{name}': {e}")

                if models_loaded > 0 or classifiers_loaded > 0:
                    logger.info(f"🧠 [ML_LOAD] Successfully loaded {models_loaded} regressors and {classifiers_loaded} classifiers")
                else:
                    logger.info(f"📚 [ML_LOAD] No saved models found - starting from scratch")

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

            # Сохраняем модели и их метаданные
            if ML_AVAILABLE:
                models_saved = 0
                classifiers_saved = 0

                # Сохраняем регрессоры
                for name, model in self.models.items():
                    if model.is_fitted:
                        # Сохраняем модель и скейлер
                        joblib.dump(model.model, self.data_dir / f"{name}_model.pkl")
                        joblib.dump(model.scaler, self.data_dir / f"{name}_scaler.pkl")

                        # Сохраняем метаданные (samples_seen и т.д.)
                        metadata = {
                            'samples_seen': model.samples_seen,
                            'is_fitted': model.is_fitted,
                            'saved_at': datetime.now(timezone.utc).isoformat(),
                            'model_version': MODEL_VERSION  # 🔢 Версия модели для совместимости
                        }
                        with open(self.data_dir / f"{name}_metadata.json", 'w') as f:
                            json.dump(metadata, f, indent=2)

                        models_saved += 1

                # 🆕 Сохраняем классификаторы
                for name, classifier in self.classifiers.items():
                    if classifier.is_fitted:
                        # Сохраняем классификатор и скейлер
                        joblib.dump(classifier.model, self.data_dir / f"{name}_classifier.pkl")
                        joblib.dump(classifier.scaler, self.data_dir / f"{name}_scaler.pkl")

                        # Сохраняем метаданные
                        metadata = {
                            'samples_seen': classifier.samples_seen,
                            'is_fitted': classifier.is_fitted,
                            'classes': classifier.classes_.tolist() if classifier.classes_ is not None else [0, 1, 2],
                            'saved_at': datetime.now(timezone.utc).isoformat(),
                            'model_version': MODEL_VERSION
                        }
                        with open(self.data_dir / f"{name}_metadata.json", 'w') as f:
                            json.dump(metadata, f, indent=2)

                        classifiers_saved += 1

                logger.info(f"💾 [ML_SAVE] Saved {models_saved} regressors and {classifiers_saved} classifiers with metadata")

            logger.info(f"💾 [ML_SAVE] Saved ML data: {len(self.market_contexts)} contexts, "
                       f"{len(self.trade_outcomes)} outcomes")

        except Exception as e:
            logger.error(f"❌ [ML_SAVE] Error: {e}")