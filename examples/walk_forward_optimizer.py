#!/usr/bin/env python3
"""
🔄 Walk-Forward Optimization System
====================================

ADAPTIVE LEARNING:
- Обучает на скользящем окне
- Тестирует на будущих данных
- Анализирует что работает
- Адаптируется к изменяющемуся рынку
- Избегает overfitting

Процесс:
1. Train на Jan-Jun → Test на Jul
2. Train на Feb-Jul → Test на Aug
3. Train на Mar-Aug → Test на Sep
... и так далее

Выгоды:
- Видим что РЕАЛЬНО работает
- Адаптация к рынку
- Оптимальные гиперпараметры
- Объективная оценка

Автор: Claude (Anthropic)
"""

import asyncio
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
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
# 📊 RESULTS DATA CLASS
# ==========================================

@dataclass
class WalkForwardResult:
    """Результат одного walk-forward теста"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Training metrics
    train_samples: int
    test_samples: int
    training_time: float
    epochs_trained: int

    # Test performance
    win_rate: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    num_trades: int
    avg_profit: float
    profit_factor: float

    # Model info
    model_params: int
    best_val_loss: float

    # Market conditions
    market_volatility: float
    market_trend: float  # +1 bull, -1 bear, 0 sideways

    def to_dict(self):
        return asdict(self)


# ==========================================
# 🔄 WALK-FORWARD ENGINE
# ==========================================

class WalkForwardOptimizer:
    """
    Walk-Forward оптимизация для торговых моделей

    Процесс:
    1. Разбивает данные на окна
    2. Обучает модель на train window
    3. Тестирует на test window (будущие данные!)
    4. Сдвигает окно и повторяет
    5. Анализирует результаты
    """

    def __init__(
        self,
        train_window_months: int = 6,
        test_window_months: int = 1,
        step_months: int = 1,
        min_samples: int = 1000
    ):
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.min_samples = min_samples

        self.results: List[WalkForwardResult] = []

    def split_data(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Разбить данные на walk-forward windows

        Returns:
            List of (train_df, test_df) tuples
        """
        windows = []

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        start_date = df.index.min()
        end_date = df.index.max()

        logger.info(f"📅 Data range: {start_date} → {end_date}")
        logger.info(f"   Total samples: {len(df):,}")

        current_date = start_date

        window_id = 0
        while True:
            # Calculate window dates
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_window_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_window_months)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Get data for this window
            train_mask = (df.index >= train_start) & (df.index < train_end)
            test_mask = (df.index >= test_start) & (df.index < test_end)

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            # Check minimum samples
            if len(train_df) < self.min_samples or len(test_df) < 100:
                logger.warning(
                    f"⚠️  Window {window_id}: Insufficient data "
                    f"(train={len(train_df)}, test={len(test_df)})"
                )
                current_date += pd.DateOffset(months=self.step_months)
                window_id += 1
                continue

            windows.append((train_df, test_df))

            logger.info(
                f"✅ Window {window_id}: "
                f"Train {train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')} "
                f"({len(train_df):,} samples), "
                f"Test {test_start.strftime('%Y-%m-%d')} → {test_end.strftime('%Y-%m-%d')} "
                f"({len(test_df):,} samples)"
            )

            # Move to next window
            current_date += pd.DateOffset(months=self.step_months)
            window_id += 1

        logger.info(f"📊 Created {len(windows)} walk-forward windows")

        return windows

    async def optimize(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        train_func,  # Function to train model
        test_func,   # Function to test model
        hyperparams: Dict = None
    ) -> List[WalkForwardResult]:
        """
        Запустить walk-forward optimization

        Args:
            data: DataFrame с данными
            symbols: Список символов
            train_func: Функция обучения модели
            test_func: Функция тестирования модели
            hyperparams: Гиперпараметры для моделей

        Returns:
            List of WalkForwardResult
        """
        if hyperparams is None:
            hyperparams = {}

        windows = self.split_data(data)

        logger.info("=" * 80)
        logger.info("🔄 WALK-FORWARD OPTIMIZATION STARTED")
        logger.info("=" * 80)
        logger.info(f"   Windows: {len(windows)}")
        logger.info(f"   Train window: {self.train_window_months} months")
        logger.info(f"   Test window: {self.test_window_months} month(s)")
        logger.info("=" * 80)

        for window_id, (train_df, test_df) in enumerate(windows):
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 WINDOW {window_id + 1}/{len(windows)}")
            logger.info(f"{'='*80}")

            start_time = time.time()

            # Train model
            logger.info(f"🎯 Training model...")
            model, train_metrics = await train_func(
                train_df=train_df,
                val_split=0.15,
                **hyperparams
            )

            # Test model
            logger.info(f"📊 Testing model on future data...")
            test_metrics = await test_func(
                model=model,
                test_df=test_df
            )

            # Calculate market conditions
            market_metrics = self._analyze_market(test_df)

            # Store results
            result = WalkForwardResult(
                window_id=window_id + 1,
                train_start=train_df.index.min().strftime('%Y-%m-%d'),
                train_end=train_df.index.max().strftime('%Y-%m-%d'),
                test_start=test_df.index.min().strftime('%Y-%m-%d'),
                test_end=test_df.index.max().strftime('%Y-%m-%d'),
                train_samples=len(train_df),
                test_samples=len(test_df),
                training_time=time.time() - start_time,
                epochs_trained=train_metrics.get('epochs', 0),
                win_rate=test_metrics.get('win_rate', 0),
                sharpe_ratio=test_metrics.get('sharpe_ratio', 0),
                total_return=test_metrics.get('total_return', 0),
                max_drawdown=test_metrics.get('max_drawdown', 0),
                num_trades=test_metrics.get('num_trades', 0),
                avg_profit=test_metrics.get('avg_profit', 0),
                profit_factor=test_metrics.get('profit_factor', 0),
                model_params=train_metrics.get('model_params', 0),
                best_val_loss=train_metrics.get('best_val_loss', 0),
                market_volatility=market_metrics['volatility'],
                market_trend=market_metrics['trend']
            )

            self.results.append(result)

            # Log window results
            logger.info(f"📈 Window {window_id + 1} Results:")
            logger.info(f"   Win Rate: {result.win_rate:.2f}%")
            logger.info(f"   Total Return: {result.total_return:+.2f}%")
            logger.info(f"   Sharpe Ratio: {result.sharpe_ratio:+.2f}")
            logger.info(f"   Max Drawdown: {result.max_drawdown:.2f}%")
            logger.info(f"   Trades: {result.num_trades}")
            logger.info(f"   Market: {market_metrics['trend_label']} "
                       f"(vol={market_metrics['volatility']:.2f}%)")

        # Final summary
        self._print_summary()

        return self.results

    def _analyze_market(self, df: pd.DataFrame) -> Dict:
        """Анализ рыночных условий"""
        returns = df['close'].pct_change() * 100
        volatility = returns.std()

        # Trend
        first_price = df['close'].iloc[0]
        last_price = df['close'].iloc[-1]
        total_return = (last_price - first_price) / first_price * 100

        if total_return > 5:
            trend = 1.0
            trend_label = "BULL 🐂"
        elif total_return < -5:
            trend = -1.0
            trend_label = "BEAR 🐻"
        else:
            trend = 0.0
            trend_label = "SIDEWAYS ↔️"

        return {
            'volatility': volatility,
            'trend': trend,
            'trend_label': trend_label,
            'total_return': total_return
        }

    def _print_summary(self):
        """Вывод итоговой статистики"""
        if not self.results:
            return

        logger.info("\n" + "=" * 80)
        logger.info("📊 WALK-FORWARD OPTIMIZATION SUMMARY")
        logger.info("=" * 80)

        # Overall stats
        avg_win_rate = np.mean([r.win_rate for r in self.results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in self.results])
        avg_return = np.mean([r.total_return for r in self.results])
        total_return = sum([r.total_return for r in self.results])

        logger.info(f"🎯 Overall Performance:")
        logger.info(f"   Avg Win Rate: {avg_win_rate:.2f}%")
        logger.info(f"   Avg Sharpe: {avg_sharpe:+.2f}")
        logger.info(f"   Avg Return per window: {avg_return:+.2f}%")
        logger.info(f"   Total Return (sum): {total_return:+.2f}%")

        # Best/Worst windows
        best_window = max(self.results, key=lambda r: r.total_return)
        worst_window = min(self.results, key=lambda r: r.total_return)

        logger.info(f"\n🏆 Best Window:")
        logger.info(f"   Window {best_window.window_id}: {best_window.total_return:+.2f}% "
                   f"(WR={best_window.win_rate:.1f}%, Sharpe={best_window.sharpe_ratio:+.2f})")

        logger.info(f"\n📉 Worst Window:")
        logger.info(f"   Window {worst_window.window_id}: {worst_window.total_return:+.2f}% "
                   f"(WR={worst_window.win_rate:.1f}%, Sharpe={worst_window.sharpe_ratio:+.2f})")

        # Market condition analysis
        bull_results = [r for r in self.results if r.market_trend > 0]
        bear_results = [r for r in self.results if r.market_trend < 0]
        sideways_results = [r for r in self.results if r.market_trend == 0]

        if bull_results:
            bull_return = np.mean([r.total_return for r in bull_results])
            logger.info(f"\n🐂 Bull Markets ({len(bull_results)} windows):")
            logger.info(f"   Avg Return: {bull_return:+.2f}%")

        if bear_results:
            bear_return = np.mean([r.total_return for r in bear_results])
            logger.info(f"\n🐻 Bear Markets ({len(bear_results)} windows):")
            logger.info(f"   Avg Return: {bear_return:+.2f}%")

        if sideways_results:
            sideways_return = np.mean([r.total_return for r in sideways_results])
            logger.info(f"\n↔️  Sideways Markets ({len(sideways_results)} windows):")
            logger.info(f"   Avg Return: {sideways_return:+.2f}%")

        # Robustness
        positive_windows = len([r for r in self.results if r.total_return > 0])
        robustness = positive_windows / len(self.results) * 100

        logger.info(f"\n💪 Robustness:")
        logger.info(f"   Positive windows: {positive_windows}/{len(self.results)} ({robustness:.1f}%)")

        logger.info("=" * 80)

    def save_results(self, path: str):
        """Сохранить результаты в JSON"""
        results_dict = [r.to_dict() for r in self.results]

        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"💾 Results saved to {path}")

    def get_best_hyperparams(self) -> Dict:
        """
        Получить лучшие гиперпараметры на основе результатов

        Returns:
            Dict с рекомендованными параметрами
        """
        if not self.results:
            return {}

        # Find patterns
        high_sharpe_results = [r for r in self.results if r.sharpe_ratio > 1.0]

        if high_sharpe_results:
            # Analyze what worked
            avg_epochs = np.mean([r.epochs_trained for r in high_sharpe_results])

            recommendations = {
                'optimal_epochs': int(avg_epochs),
                'min_samples': self.min_samples,
                'retrain_frequency': f"{self.step_months} month(s)",
            }

            logger.info(f"💡 Recommendations based on {len(high_sharpe_results)} successful windows:")
            for key, val in recommendations.items():
                logger.info(f"   {key}: {val}")

            return recommendations

        return {}


# ==========================================
# 🚀 MAIN (EXAMPLE USAGE)
# ==========================================

async def run_walk_forward_example():
    """
    Пример использования Walk-Forward Optimizer
    """
    # Import training functions
    from gru_training_improved import train_improved_gru

    # Load data
    from gru_training_pytorch import (
        BinanceDataDownloader,
        calculate_technical_indicators
    )

    symbols = ['BTCUSDT']
    downloader = BinanceDataDownloader()

    logger.info("📥 Loading data...")
    all_data = []
    for symbol in symbols:
        df = await downloader.download_historical_data(symbol, '30m', 365)
        if len(df) > 0:
            df = calculate_technical_indicators(df)
            all_data.append(df)

    data = pd.concat(all_data, ignore_index=False)

    # Define training function wrapper
    async def train_wrapper(train_df, val_split=0.15, **kwargs):
        """Wrapper для функции обучения"""
        # Here you would call your actual training function
        # For now, return dummy metrics
        return None, {
            'epochs': 20,
            'model_params': 100000,
            'best_val_loss': 0.5
        }

    # Define testing function wrapper
    async def test_wrapper(model, test_df):
        """Wrapper для функции тестирования"""
        # Here you would test your model
        # For now, return dummy metrics
        return {
            'win_rate': np.random.uniform(45, 60),
            'sharpe_ratio': np.random.uniform(-1, 2),
            'total_return': np.random.uniform(-10, 15),
            'max_drawdown': np.random.uniform(-20, -5),
            'num_trades': int(np.random.uniform(10, 50)),
            'avg_profit': np.random.uniform(-1, 2),
            'profit_factor': np.random.uniform(0.8, 1.5)
        }

    # Create optimizer
    optimizer = WalkForwardOptimizer(
        train_window_months=6,
        test_window_months=1,
        step_months=1
    )

    # Run optimization
    results = await optimizer.optimize(
        data=data,
        symbols=symbols,
        train_func=train_wrapper,
        test_func=test_wrapper,
        hyperparams={
            'epochs': 50,
            'batch_size': 256
        }
    )

    # Save results
    optimizer.save_results('data/walk_forward_results.json')

    # Get recommendations
    optimizer.get_best_hyperparams()


if __name__ == "__main__":
    asyncio.run(run_walk_forward_example())
