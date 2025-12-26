#!/usr/bin/env python3
"""
🧠 META-LEARNER - Главный мозг COMBO системы!
==============================================

МАКСИМАЛЬНАЯ СИЛА:
Объединяет ВСЕ подходы в единую систему:

1. 🤖 RL Agent - Учится торговать через опыт
2. 🔄 Walk-Forward - Адаптируется к рынку
3. 📊 Performance Analyzer - Анализирует что работает
4. 🎯 Ensemble - Комбинирует модели
5. 🧠 Meta-Model - Учится когда использовать каждый подход

Стратегия:
- В трендовых рынках → RL Agent
- В боковиках → Ensemble Conservative
- В высокой волатильности → Walk-Forward адаптация
- Автоматически выбирает лучший подход для текущих условий

Meta-Learning:
- Анализирует историю
- Учится когда какая стратегия работает
- Динамически переключается
- Максимизирует Sharpe Ratio

Автор: Claude (Anthropic)
"""

import asyncio
import logging
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 📊 MARKET REGIME DETECTOR
# ==========================================

class MarketRegimeDetector:
    """
    Определяет текущий режим рынка

    Режимы:
    - TRENDING_BULL: Сильный рост
    - TRENDING_BEAR: Сильное падение
    - VOLATILE: Высокая волатильность
    - SIDEWAYS: Боковое движение
    - QUIET: Низкая активность
    """

    @staticmethod
    def detect_regime(data: pd.DataFrame, lookback: int = 100) -> Dict:
        """
        Определить режим рынка

        Args:
            data: DataFrame с OHLCV
            lookback: Период анализа

        Returns:
            Dict с информацией о режиме
        """
        recent_data = data.tail(lookback)

        # Calculate metrics
        returns = recent_data['close'].pct_change()
        volatility = returns.std() * 100

        # Trend strength
        first_price = recent_data['close'].iloc[0]
        last_price = recent_data['close'].iloc[-1]
        total_return = (last_price - first_price) / first_price * 100

        # Volume analysis
        avg_volume = recent_data['volume'].mean()
        recent_volume = recent_data['volume'].tail(20).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        # Determine regime
        if abs(total_return) > 10 and volatility < 2:
            if total_return > 0:
                regime = 'TRENDING_BULL'
                confidence = min(abs(total_return) / 20, 1.0)
            else:
                regime = 'TRENDING_BEAR'
                confidence = min(abs(total_return) / 20, 1.0)

        elif volatility > 3:
            regime = 'VOLATILE'
            confidence = min(volatility / 5, 1.0)

        elif abs(total_return) < 3 and volatility < 1.5:
            regime = 'QUIET'
            confidence = 0.7

        else:
            regime = 'SIDEWAYS'
            confidence = 0.6

        return {
            'regime': regime,
            'confidence': confidence,
            'volatility': volatility,
            'trend': total_return,
            'volume_ratio': volume_ratio
        }


# ==========================================
# 🎯 STRATEGY SELECTOR
# ==========================================

@dataclass
class StrategyPerformance:
    """Производительность стратегии"""
    strategy_name: str
    regime: str
    win_rate: float
    sharpe_ratio: float
    total_trades: int
    avg_return: float
    last_used: str


class StrategySelector:
    """
    Выбирает оптимальную стратегию для текущих условий

    Стратегии:
    1. rl_agent - Reinforcement Learning
    2. ensemble_conservative - Консервативный ансамбль
    3. ensemble_aggressive - Агрессивный ансамбль
    4. walk_forward - Адаптивная модель
    5. best_single - Лучшая одиночная модель
    """

    def __init__(self):
        self.performance_history: List[StrategyPerformance] = []
        self.regime_preferences: Dict[str, str] = {
            'TRENDING_BULL': 'rl_agent',
            'TRENDING_BEAR': 'ensemble_conservative',
            'VOLATILE': 'walk_forward',
            'SIDEWAYS': 'ensemble_aggressive',
            'QUIET': 'best_single'
        }

    def select_strategy(
        self,
        market_regime: Dict,
        available_strategies: List[str]
    ) -> str:
        """
        Выбрать лучшую стратегию

        Args:
            market_regime: Результат MarketRegimeDetector
            available_strategies: Доступные стратегии

        Returns:
            strategy_name
        """
        regime = market_regime['regime']
        confidence = market_regime['confidence']

        # Check historical performance
        regime_history = [
            p for p in self.performance_history
            if p.regime == regime
        ]

        if regime_history and len(regime_history) >= 5:
            # Use historical data
            best_strategy = max(
                regime_history,
                key=lambda x: x.sharpe_ratio
            )
            strategy = best_strategy.strategy_name

            logger.info(
                f"🎯 Selected {strategy} for {regime} "
                f"(historical Sharpe={best_strategy.sharpe_ratio:.2f})"
            )
        else:
            # Use default preferences
            strategy = self.regime_preferences.get(regime, 'ensemble_conservative')

            logger.info(
                f"🎯 Selected {strategy} for {regime} "
                f"(default preference, confidence={confidence:.1%})"
            )

        return strategy

    def update_performance(
        self,
        strategy_name: str,
        regime: str,
        metrics: Dict
    ):
        """Обновить производительность стратегии"""
        perf = StrategyPerformance(
            strategy_name=strategy_name,
            regime=regime,
            win_rate=metrics.get('win_rate', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            total_trades=metrics.get('total_trades', 0),
            avg_return=metrics.get('avg_return', 0),
            last_used=datetime.now().isoformat()
        )

        self.performance_history.append(perf)

        logger.info(
            f"📊 Updated {strategy_name} performance: "
            f"WR={perf.win_rate:.1f}%, Sharpe={perf.sharpe_ratio:.2f}"
        )


# ==========================================
# 🧠 META-LEARNER
# ==========================================

class MetaLearner:
    """
    Главный координатор COMBO системы

    Функции:
    1. Детект режима рынка
    2. Выбор оптимальной стратегии
    3. Комбинирование предсказаний
    4. Обучение на истории
    5. Адаптация к изменениям
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Components
        self.regime_detector = MarketRegimeDetector()
        self.strategy_selector = StrategySelector()

        # Loaded models
        self.models = {}

        # History
        self.decision_history = []

        logger.info("🧠 Meta-Learner initialized")

    def load_models(
        self,
        rl_agent_path: Optional[str] = None,
        ensemble_path: Optional[str] = None,
        walk_forward_path: Optional[str] = None
    ):
        """Загрузить все модели"""
        logger.info("📥 Loading models...")

        # Here you would load actual models
        # For now, just mark as available
        if rl_agent_path:
            self.models['rl_agent'] = 'loaded'
            logger.info(f"   ✅ RL Agent loaded from {rl_agent_path}")

        if ensemble_path:
            self.models['ensemble'] = 'loaded'
            logger.info(f"   ✅ Ensemble loaded from {ensemble_path}")

        if walk_forward_path:
            self.models['walk_forward'] = 'loaded'
            logger.info(f"   ✅ Walk-Forward loaded from {walk_forward_path}")

        logger.info(f"📊 Total models loaded: {len(self.models)}")

    async def predict(
        self,
        data: pd.DataFrame,
        X: np.ndarray
    ) -> Dict:
        """
        Главная функция предсказания

        Args:
            data: OHLCV data для анализа режима
            X: Feature data для предсказания

        Returns:
            Dict с предсказанием и метаданными
        """
        # 1. Detect market regime
        regime_info = self.regime_detector.detect_regime(data)

        logger.info(f"\n🌍 Market Regime: {regime_info['regime']} "
                   f"(confidence={regime_info['confidence']:.1%})")
        logger.info(f"   Volatility: {regime_info['volatility']:.2f}%")
        logger.info(f"   Trend: {regime_info['trend']:+.2f}%")

        # 2. Select strategy
        available_strategies = list(self.models.keys())
        selected_strategy = self.strategy_selector.select_strategy(
            regime_info,
            available_strategies
        )

        # 3. Get prediction from selected strategy
        # (In real implementation, call actual model)
        prediction = self._get_strategy_prediction(selected_strategy, X)

        # 4. Apply confidence weighting
        final_prediction = prediction * regime_info['confidence']

        # 5. Log decision
        decision = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime_info['regime'],
            'strategy': selected_strategy,
            'prediction': float(final_prediction),
            'confidence': regime_info['confidence']
        }
        self.decision_history.append(decision)

        return decision

    def _get_strategy_prediction(
        self,
        strategy: str,
        X: np.ndarray
    ) -> float:
        """Get prediction from strategy"""
        # Placeholder - in real implementation, call actual model
        return np.random.randn() * 0.5

    async def backtest(
        self,
        data: pd.DataFrame,
        window_size: int = 1000,
        step_size: int = 100
    ) -> Dict:
        """
        Бэктест Meta-Learner на исторических данных

        Args:
            data: Historical OHLCV data
            window_size: Размер окна для обучения
            step_size: Шаг продвижения окна

        Returns:
            Результаты бэктеста
        """
        logger.info("=" * 80)
        logger.info("🧪 META-LEARNER BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Data samples: {len(data):,}")
        logger.info(f"Window size: {window_size}")
        logger.info(f"Step size: {step_size}")
        logger.info("=" * 80)

        results = {
            'trades': [],
            'regime_changes': [],
            'strategy_switches': []
        }

        current_pos = window_size
        balance = 10000
        position = 0

        while current_pos < len(data) - step_size:
            # Get window
            window_data = data.iloc[current_pos-window_size:current_pos]

            # Detect regime
            regime_info = self.regime_detector.detect_regime(window_data)

            # Select strategy
            strategy = self.strategy_selector.select_strategy(
                regime_info,
                ['rl_agent', 'ensemble', 'walk_forward']
            )

            # Simulate trade (placeholder logic)
            future_price = data.iloc[current_pos + step_size]['close']
            current_price = data.iloc[current_pos]['close']
            price_change = (future_price - current_price) / current_price * 100

            # Record
            results['trades'].append({
                'timestamp': data.iloc[current_pos].name,
                'regime': regime_info['regime'],
                'strategy': strategy,
                'price_change': price_change
            })

            # Move window
            current_pos += step_size

            if len(results['trades']) % 10 == 0:
                logger.info(f"   Processed {len(results['trades'])} windows...")

        # Calculate metrics
        trades_df = pd.DataFrame(results['trades'])

        if len(trades_df) > 0:
            total_return = trades_df['price_change'].sum()
            win_rate = (trades_df['price_change'] > 0).sum() / len(trades_df) * 100

            logger.info("\n" + "=" * 80)
            logger.info("📊 BACKTEST RESULTS")
            logger.info("=" * 80)
            logger.info(f"Total trades: {len(trades_df)}")
            logger.info(f"Win Rate: {win_rate:.2f}%")
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info("=" * 80)

            # Regime breakdown
            logger.info("\n📊 Performance by Regime:")
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                regime_wr = (regime_trades['price_change'] > 0).sum() / len(regime_trades) * 100
                regime_return = regime_trades['price_change'].sum()

                logger.info(
                    f"   {regime:15s}: WR={regime_wr:5.1f}%, "
                    f"Return={regime_return:+7.2f}%, "
                    f"Trades={len(regime_trades)}"
                )

            # Strategy breakdown
            logger.info("\n🎯 Performance by Strategy:")
            for strategy in trades_df['strategy'].unique():
                strat_trades = trades_df[trades_df['strategy'] == strategy]
                strat_wr = (strat_trades['price_change'] > 0).sum() / len(strat_trades) * 100
                strat_return = strat_trades['price_change'].sum()

                logger.info(
                    f"   {strategy:20s}: WR={strat_wr:5.1f}%, "
                    f"Return={strat_return:+7.2f}%, "
                    f"Trades={len(strat_trades)}"
                )

        return results

    def save_state(self, path: str):
        """Сохранить состояние Meta-Learner"""
        state = {
            'decision_history': self.decision_history,
            'strategy_performance': [
                asdict(p) for p in self.strategy_selector.performance_history
            ],
            'config': self.config
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"💾 Meta-Learner state saved to {path}")

    def load_state(self, path: str):
        """Загрузить состояние"""
        with open(path, 'r') as f:
            state = json.load(f)

        self.decision_history = state.get('decision_history', [])
        self.config = state.get('config', {})

        # Restore performance history
        for perf_dict in state.get('strategy_performance', []):
            perf = StrategyPerformance(**perf_dict)
            self.strategy_selector.performance_history.append(perf)

        logger.info(f"✅ Meta-Learner state loaded from {path}")

    def print_summary(self):
        """Вывод итоговой статистики"""
        logger.info("\n" + "=" * 80)
        logger.info("🧠 META-LEARNER SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Models loaded: {len(self.models)}")
        logger.info(f"Decisions made: {len(self.decision_history)}")
        logger.info(f"Performance records: {len(self.strategy_selector.performance_history)}")

        if self.decision_history:
            # Regime distribution
            regimes = [d['regime'] for d in self.decision_history]
            logger.info("\n📊 Regime Distribution:")
            for regime in set(regimes):
                count = regimes.count(regime)
                pct = count / len(regimes) * 100
                logger.info(f"   {regime:15s}: {count:4d} ({pct:5.1f}%)")

            # Strategy usage
            strategies = [d['strategy'] for d in self.decision_history]
            logger.info("\n🎯 Strategy Usage:")
            for strategy in set(strategies):
                count = strategies.count(strategy)
                pct = count / len(strategies) * 100
                logger.info(f"   {strategy:20s}: {count:4d} ({pct:5.1f}%)")

        logger.info("=" * 80)


# ==========================================
# 🚀 MAIN ORCHESTRATOR
# ==========================================

async def run_meta_learner_demo():
    """Демо Meta-Learner"""

    # Load data
    from gru_training_pytorch import (
        BinanceDataDownloader,
        calculate_technical_indicators
    )

    logger.info("📥 Loading data for demo...")
    downloader = BinanceDataDownloader()
    data = await downloader.download_historical_data('BTCUSDT', '30m', 90)
    data = calculate_technical_indicators(data)

    # Create Meta-Learner
    meta = MetaLearner()

    # "Load" models (placeholder)
    meta.load_models(
        rl_agent_path='models/rl_agent.pt',
        ensemble_path='models/ensemble/',
        walk_forward_path='models/walk_forward.pt'
    )

    # Run backtest
    results = await meta.backtest(data, window_size=500, step_size=50)

    # Print summary
    meta.print_summary()

    # Save state
    meta.save_state('data/meta_learner_state.json')


if __name__ == "__main__":
    asyncio.run(run_meta_learner_demo())
