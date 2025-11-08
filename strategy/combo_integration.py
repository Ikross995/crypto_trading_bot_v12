"""
COMBO System Integration Wrapper.

Integrates trained COMBO models (Ensemble + RL Agent + Meta-Learner) into the trading bot.
When use_combo_signals=True, uses ML models for signal generation.

COMBO система включает:
- Ensemble из 5 GRU моделей
- RL Agent (Deep Q-Network)
- Meta-Learner (интеллектуальный выбор стратегии)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from core.config import Config

logger = logging.getLogger(__name__)


class COMBOSignalIntegration:
    """
    Интеграция COMBO моделей в торгового бота.

    Загружает обученные модели для каждого символа и генерирует сигналы.
    """

    def __init__(self, config: Config):
        """
        Инициализация COMBO системы.

        Args:
            config: Конфигурация бота
        """
        self.config = config
        self.models = {}  # {symbol: {'ensemble': ..., 'rl_agent': ..., 'meta': ...}}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info("🚀 Initializing COMBO Signal Integration...")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Min confidence: {config.bt_conf_min}")

    def load_models_for_symbol(self, symbol: str) -> bool:
        """
        Загрузить модели для конкретного символа.

        Args:
            symbol: Торговая пара (например, BTCUSDT)

        Returns:
            True если модели загружены успешно
        """
        if symbol in self.models:
            logger.debug(f"Models for {symbol} already loaded")
            return True

        try:
            # Пути к моделям
            ensemble_path = Path(f'models/combo_ensemble_{symbol}')
            rl_agent_path = Path(f'models/combo_rl_agent_{symbol}.pt')
            meta_path = Path(f'data/combo_meta_learner_{symbol}.json')

            # Проверка существования
            if not ensemble_path.exists():
                logger.warning(f"❌ Ensemble models not found for {symbol}: {ensemble_path}")
                logger.warning(f"   Run training first: python run_full_combo_system_multi.py --quick --symbols {symbol}")
                return False

            if not rl_agent_path.exists():
                logger.warning(f"❌ RL Agent model not found for {symbol}: {rl_agent_path}")
                return False

            # Загрузка Ensemble
            from examples.ensemble_trainer import EnsembleTrainer
            ensemble = EnsembleTrainer()
            ensemble.load_ensemble(str(ensemble_path))
            logger.info(f"   ✅ Loaded Ensemble for {symbol}: {len(ensemble.models)} models")

            # Загрузка RL Agent
            from examples.rl_trading_agent import RLTradingAgent

            # Создаем агента (размеры будут из чекпоинта)
            checkpoint = torch.load(rl_agent_path, map_location=self.device)

            # Правильно извлекаем state_dict из checkpoint
            # Checkpoint может содержать: 'policy_net', 'target_net', 'optimizer', etc.
            if 'policy_net' in checkpoint:
                state_dict = checkpoint['policy_net']
            elif 'policy_net_state' in checkpoint:
                state_dict = checkpoint['policy_net_state']
            else:
                # Fallback: весь checkpoint это state_dict
                state_dict = checkpoint

            # Определяем state_size из первого слоя
            first_layer_key = 'fc1.weight'
            if first_layer_key in state_dict:
                state_size = state_dict[first_layer_key].shape[1]
            else:
                state_size = 22  # default

            action_size = 4  # HOLD, LONG, SHORT, CLOSE

            rl_agent = RLTradingAgent(
                state_size=state_size,
                action_size=action_size,
                device=self.device
            )

            # Загружаем веса policy_net
            rl_agent.policy_net.load_state_dict(state_dict)
            rl_agent.policy_net.eval()

            logger.info(f"   ✅ Loaded RL Agent for {symbol}")

            # Загрузка Meta-Learner (опционально)
            meta_learner = None
            if meta_path.exists():
                from examples.meta_learner import MetaLearner
                meta_learner = MetaLearner()
                meta_learner.load_state(str(meta_path))
                logger.info(f"   ✅ Loaded Meta-Learner for {symbol}")
            else:
                logger.warning(f"   ⚠️  Meta-Learner state not found for {symbol} (optional)")

            # Сохраняем модели
            self.models[symbol] = {
                'ensemble': ensemble,
                'rl_agent': rl_agent,
                'meta_learner': meta_learner,
                'state_size': state_size
            }

            logger.info(f"🎉 COMBO models loaded successfully for {symbol}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load COMBO models for {symbol}: {e}", exc_info=True)
            return False

    def _prepare_state(self, df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """
        Подготовить state для RL Agent из последних данных.

        ВАЖНО: State должен соответствовать обучению (15 features):
        - 10 market features: close_norm, volume_norm, returns, volatility, rsi, macd, bb_upper, bb_lower, sma_20, sma_50
        - 3 position features: position, pnl%, position_size%
        - 2 balance features: total_return, num_trades

        Args:
            df: DataFrame с OHLCV (индикаторы будут добавлены если отсутствуют)
            symbol: Торговый символ

        Returns:
            State вектор для агента (15 features) или None
        """
        try:
            if len(df) < 100:
                logger.warning(f"Not enough data for state preparation: {len(df)} < 100")
                return None

            # Добавляем индикаторы, если их нет
            from examples.adaptive_trading_integration import add_technical_indicators

            # Проверяем, есть ли индикаторы
            required_indicators = ['rsi', 'macd', 'bb_upper', 'sma_20', 'sma_50']
            if not all(col in df.columns for col in required_indicators):
                df = add_technical_indicators(df)

            # Последняя свеча
            latest = df.iloc[-1]

            # Calculate normalized features (как при обучении)
            # Price normalization: (price - mean) / std
            close_rolling_mean = df['close'].rolling(100).mean().iloc[-1]
            close_rolling_std = df['close'].rolling(100).std().iloc[-1]
            close_norm = (latest['close'] - close_rolling_mean) / close_rolling_std if close_rolling_std > 0 else 0

            volume_rolling_mean = df['volume'].rolling(100).mean().iloc[-1]
            volume_rolling_std = df['volume'].rolling(100).std().iloc[-1]
            volume_norm = (latest['volume'] - volume_rolling_mean) / volume_rolling_std if volume_rolling_std > 0 else 0

            # Returns and volatility
            returns = df['close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * 100 if len(df) > 20 else 0

            # Market features (10)
            market_features = [
                close_norm,
                volume_norm,
                returns,
                volatility,
                latest.get('rsi', 50),
                latest.get('macd', 0),
                latest.get('bb_upper', latest['close']),
                latest.get('bb_lower', latest['close']),
                latest.get('sma_20', latest['close']),
                latest.get('sma_50', latest['close']),
            ]

            # Position features (3) - нет позиции при генерации сигнала
            position_features = [
                0.0,  # position (no position)
                0.0,  # pnl% (no position)
                0.0,  # position_size% (no position)
            ]

            # Balance features (2)
            balance_features = [
                0.0,  # total_return (начало)
                0.0,  # num_trades (начало)
            ]

            state = np.array(market_features + position_features + balance_features, dtype=np.float32)

            # Handle NaN (как при обучении)
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

            return state

        except Exception as e:
            logger.error(f"Failed to prepare state: {e}", exc_info=True)
            return None

    def _get_ensemble_prediction(self, df: pd.DataFrame, symbol: str) -> Optional[float]:
        """
        Получить предсказание от Ensemble моделей.

        Args:
            df: DataFrame с данными (OHLCV)
            symbol: Торговый символ

        Returns:
            Предсказание изменения цены в % или None
        """
        try:
            if symbol not in self.models:
                return None

            ensemble = self.models[symbol]['ensemble']

            # Берем достаточно свечей для sequence (60) + нормализации
            min_required = 100  # Для rolling нормализации
            if len(df) < min_required:
                logger.warning(f"Not enough data for ensemble prediction: {len(df)} < {min_required}")
                return None

            recent_df = df.iloc[-min_required:].copy()

            # Добавляем индикаторы, если их нет
            from examples.adaptive_trading_integration import add_technical_indicators

            # Проверяем, есть ли индикаторы
            required_indicators = ['rsi', 'macd', 'bb_upper']
            if not all(col in recent_df.columns for col in required_indicators):
                recent_df = add_technical_indicators(recent_df)

            # Используем только индикаторы, которые добавляет add_technical_indicators()
            feature_columns = [
                'open', 'high', 'low', 'volume',
                'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_middle', 'bb_lower',
                'sma_20', 'sma_50', 'ema_50',
                'volume_sma', 'atr', 'volume_ratio'
            ]

            # Для inference: используем простую подготовку без split
            # Нормализация и создание последовательности вручную
            from sklearn.preprocessing import RobustScaler

            # Подготовка features и target
            df_features = recent_df[feature_columns].copy()
            df_target = recent_df['close'].pct_change().shift(-1) * 100  # % change

            # Удаляем NaN
            df_features = df_features.dropna()
            df_target = df_target.dropna()

            # Выравниваем длины
            min_len = min(len(df_features), len(df_target))
            df_features = df_features.iloc[:min_len]
            df_target = df_target.iloc[:min_len]

            if len(df_features) < 60:
                logger.warning(f"Not enough clean data after indicators: {len(df_features)} < 60")
                return None

            # Нормализация
            feature_scaler = RobustScaler()
            target_scaler = RobustScaler()

            features_scaled = feature_scaler.fit_transform(df_features)
            target_scaled = target_scaler.fit_transform(df_target.values.reshape(-1, 1))

            # Создаем последнюю последовательность (60 timesteps)
            sequence_length = 60
            last_sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)

            # Конвертируем в numpy для ensemble.predict()
            # ensemble.predict() сам конвертирует в torch tensor
            last_sequence_np = last_sequence  # Уже numpy array

            # Предсказание от ансамбля (передаем numpy array)
            prediction = ensemble.predict(last_sequence_np)

            # Обратное преобразование
            prediction_original = target_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

            logger.debug(f"Ensemble prediction for {symbol}: {prediction_original:.4f}%")

            return float(prediction_original)

        except Exception as e:
            logger.error(f"Failed to get ensemble prediction: {e}", exc_info=True)
            return None

    def generate_signal_from_df(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Генерация торгового сигнала используя COMBO модели.

        Args:
            df: DataFrame с OHLCV + индикаторами
            symbol: Торговый символ

        Returns:
            Словарь с сигналом:
            {
                'direction': 'buy' | 'sell' | 'wait',
                'strength': float (0.0-1.0),
                'confidence': float,
                'regime': dict,
                'signals': list,
                'filters_passed': bool,
                'metadata': dict
            }
        """
        # Загружаем модели если не загружены
        if symbol not in self.models:
            if not self.load_models_for_symbol(symbol):
                return self._wait_signal(f"Models not available for {symbol}")

        if len(df) < 250:
            return self._wait_signal(f"Insufficient data: {len(df)} < 250")

        try:
            models = self.models[symbol]

            # 1. Получаем предсказание Ensemble
            ensemble_prediction = self._get_ensemble_prediction(df, symbol)

            # 2. Определяем режим рынка
            regime = self._detect_market_regime(df)

            # 3. Получаем решение RL Agent
            state = self._prepare_state(df, symbol)

            if state is None:
                return self._wait_signal("Failed to prepare state")

            rl_agent = models['rl_agent']
            action = rl_agent.act(state, training=False)

            # Маппинг действий
            action_map = {
                0: 'wait',   # HOLD
                1: 'buy',    # LONG
                2: 'sell',   # SHORT
                3: 'wait'    # CLOSE (интерпретируем как wait при генерации сигнала)
            }

            rl_direction = action_map[action]

            # 4. Комбинируем сигналы
            # Ensemble предсказывает направление
            ensemble_direction = 'wait'
            if ensemble_prediction is not None:
                if ensemble_prediction > 0.5:  # > +0.5% - покупка
                    ensemble_direction = 'buy'
                elif ensemble_prediction < -0.5:  # < -0.5% - продажа
                    ensemble_direction = 'sell'

            # Согласованность сигналов
            if rl_direction == ensemble_direction and rl_direction != 'wait':
                final_direction = rl_direction
                confidence = 0.8  # Высокая уверенность при согласованности
            elif rl_direction != 'wait':
                final_direction = rl_direction
                confidence = 0.5  # Средняя уверенность
            elif ensemble_direction != 'wait':
                final_direction = ensemble_direction
                confidence = 0.4  # Низкая уверенность (только Ensemble)
            else:
                final_direction = 'wait'
                confidence = 0.0

            # Корректировка по режиму рынка
            if regime['kind'] == 'volatile':
                confidence *= 0.8  # Снижаем уверенность в волатильном рынке
            elif regime['kind'] == 'trend':
                confidence *= 1.2  # Повышаем уверенность в трендовом рынке

            confidence = min(confidence, 1.0)  # Ограничиваем 1.0

            # Формируем результат
            result = {
                'direction': final_direction,
                'strength': confidence,
                'confidence': confidence,
                'regime': regime,
                'signals': [
                    {
                        'name': 'RL Agent',
                        'direction': rl_direction,
                        'strength': 1.0 if rl_direction != 'wait' else 0.0
                    },
                    {
                        'name': 'Ensemble',
                        'direction': ensemble_direction,
                        'strength': abs(ensemble_prediction) / 2.0 if ensemble_prediction else 0.0,
                        'prediction_pct': ensemble_prediction
                    }
                ],
                'filters_passed': True,  # COMBO модели уже обучены с учетом фильтров
                'metadata': {
                    'combo_enabled': True,
                    'rl_action': action,
                    'ensemble_prediction': ensemble_prediction,
                    'regime': regime,
                    'agreement': rl_direction == ensemble_direction
                }
            }

            # Логируем результат
            self._log_signal(symbol, result, df)

            return result

        except Exception as e:
            logger.error(f"COMBO signal generation failed for {symbol}: {e}", exc_info=True)
            return self._wait_signal(f"Error: {e}")

    def _detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Определить режим рынка.

        Args:
            df: DataFrame с данными

        Returns:
            Словарь с информацией о режиме
        """
        try:
            latest = df.iloc[-1]

            # ADX для определения тренда
            adx = latest.get('adx', 25)

            # Bollinger Bands Width для волатильности
            bb_upper = latest.get('bb_upper', 0)
            bb_lower = latest.get('bb_lower', 0)
            close = latest.get('close', 1)

            bbw = (bb_upper - bb_lower) / close if close > 0 else 0

            # Определение режима
            if adx > 25:
                kind = 'trend'
                confidence = min(adx / 50, 1.0)
            elif bbw > 0.05:
                kind = 'volatile'
                confidence = min(bbw * 10, 1.0)
            else:
                kind = 'flat'
                confidence = 0.5

            return {
                'kind': kind,
                'adx': adx,
                'bbw': bbw,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Failed to detect regime: {e}")
            return {'kind': 'unknown', 'adx': 0, 'bbw': 0, 'confidence': 0}

    def _wait_signal(self, reason: str) -> Dict[str, Any]:
        """Создать сигнал WAIT."""
        return {
            'direction': 'wait',
            'strength': 0.0,
            'confidence': 0.0,
            'regime': {'kind': 'unknown'},
            'signals': [],
            'filters_passed': False,
            'metadata': {'combo_enabled': True, 'reason': reason}
        }

    def _log_signal(self, symbol: str, result: Dict[str, Any], df: pd.DataFrame):
        """
        Красивое логирование сигнала.

        Args:
            symbol: Торговый символ
            result: Результат генерации сигнала
            df: DataFrame с данными
        """
        try:
            direction = result['direction']
            confidence = result['confidence']
            regime = result['regime']
            metadata = result['metadata']

            # Эмодзи
            direction_emoji = {'buy': '🟢', 'sell': '🔴', 'wait': '⏸️'}.get(direction, '⏸️')
            regime_emoji = {'trend': '📈', 'flat': '📊', 'volatile': '⚡'}.get(regime['kind'], '❓')

            # Текущая цена
            current_price = float(df.iloc[-1]['close'])

            # Логируем
            logger.info("\n" + "=" * 80)
            logger.info(f"{direction_emoji} COMBO SIGNAL: {symbol} → {direction.upper()}")
            logger.info("=" * 80)
            logger.info(f"📊 Price: ${current_price:,.2f}")
            logger.info(f"🎯 Confidence: {confidence:.3f}")
            logger.info(f"{regime_emoji} Regime: {regime['kind'].upper()} (ADX={regime.get('adx', 0):.1f})")

            # Детали сигналов
            logger.info("\n🤖 MODEL SIGNALS:")
            for sig in result['signals']:
                name = sig['name']
                sig_dir = sig['direction']
                sig_strength = sig.get('strength', 0)

                sig_emoji = {'buy': '🟢', 'sell': '🔴', 'wait': '⏸️'}.get(sig_dir, '⏸️')

                if name == 'Ensemble' and 'prediction_pct' in sig:
                    pred_pct = sig['prediction_pct']
                    logger.info(f"  ├─ {name:15s} {sig_emoji} {sig_dir:4s} (pred: {pred_pct:+.2f}%)")
                else:
                    logger.info(f"  ├─ {name:15s} {sig_emoji} {sig_dir:4s} (strength: {sig_strength:.2f})")

            # Согласованность
            agreement = metadata.get('agreement', False)
            agreement_emoji = '✅' if agreement else '⚠️'
            logger.info(f"\n{agreement_emoji} Models Agreement: {'YES' if agreement else 'NO'}")

            # Решение
            will_execute = (
                direction in ('buy', 'sell') and
                confidence >= self.config.bt_conf_min
            )

            logger.info(f"\n🎯 DECISION:")
            logger.info(f"  ├─ Direction: {direction.upper()}")
            logger.info(f"  ├─ Confidence: {confidence:.3f} / {self.config.bt_conf_min:.1f}")
            logger.info(f"  └─ Action: {'🚀 EXECUTE' if will_execute else '⏸️  WAIT'}")
            logger.info("=" * 80)

        except Exception as e:
            logger.debug(f"Failed to log signal: {e}")


def should_use_combo_signals(config: Config) -> bool:
    """
    Проверить, нужно ли использовать COMBO сигналы.

    Args:
        config: Конфигурация бота

    Returns:
        True если COMBO включен
    """
    return getattr(config, 'use_combo_signals', False)
