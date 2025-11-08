#!/usr/bin/env python3
"""
📊 Performance Analyzer - Умная система анализа торговли
========================================================

ГЛУБОКИЙ АНАЛИЗ:
- Анализирует каждую сделку
- Находит паттерны успеха/провала
- Выявляет оптимальные условия
- Рекомендует улучшения
- Учится на ошибках

Метрики:
- Win Rate по времени суток/дню недели
- Профитность по волатильности
- Лучшие индикаторы
- Оптимальный размер позиции
- Рыночные условия для входа

Автор: Claude (Anthropic)
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 📊 TRADE ANALYSIS
# ==========================================

@dataclass
class TradeAnalysis:
    """Анализ одной сделки"""
    trade_id: int
    entry_time: str
    exit_time: str
    direction: str  # LONG/SHORT
    entry_price: float
    exit_price: float
    profit_pct: float
    profit_usd: float
    hold_time_minutes: int

    # Market conditions
    hour_of_day: int
    day_of_week: int
    volatility: float
    trend: str  # UP/DOWN/SIDEWAYS

    # Indicators at entry
    rsi: float
    macd: float
    bb_position: float  # -1 to 1 (lower to upper)

    # Result
    success: bool  # True if profit > 0

    def to_dict(self):
        return asdict(self)


class PerformanceAnalyzer:
    """
    Анализатор торговой производительности

    Функции:
    - Анализ сделок
    - Поиск паттернов
    - Оптимальные условия
    - Рекомендации
    """

    def __init__(self):
        self.trades: List[TradeAnalysis] = []
        self.patterns: Dict = {}

    def add_trade(self, trade: TradeAnalysis):
        """Добавить сделку для анализа"""
        self.trades.append(trade)

    def analyze(self) -> Dict:
        """
        Полный анализ всех сделок

        Returns:
            Dict с результатами анализа
        """
        if not self.trades:
            logger.warning("⚠️  No trades to analyze!")
            return {}

        logger.info("=" * 80)
        logger.info("📊 PERFORMANCE ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Total trades analyzed: {len(self.trades)}")

        results = {}

        # 1. Overall statistics
        results['overall'] = self._analyze_overall()

        # 2. Time-based analysis
        results['time_analysis'] = self._analyze_by_time()

        # 3. Market conditions
        results['market_conditions'] = self._analyze_market_conditions()

        # 4. Indicator effectiveness
        results['indicators'] = self._analyze_indicators()

        # 5. Hold time analysis
        results['hold_time'] = self._analyze_hold_time()

        # 6. Best/Worst trades
        results['extremes'] = self._analyze_extremes()

        # 7. Recommendations
        results['recommendations'] = self._generate_recommendations(results)

        # Print summary
        self._print_analysis(results)

        return results

    def _analyze_overall(self) -> Dict:
        """Общая статистика"""
        profits = [t.profit_pct for t in self.trades]
        winning_trades = [t for t in self.trades if t.success]
        losing_trades = [t for t in self.trades if not t.success]

        win_rate = len(winning_trades) / len(self.trades) * 100

        avg_win = np.mean([t.profit_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.profit_pct for t in losing_trades]) if losing_trades else 0

        profit_factor = (
            sum([t.profit_pct for t in winning_trades]) /
            abs(sum([t.profit_pct for t in losing_trades]))
            if losing_trades and sum([t.profit_pct for t in losing_trades]) != 0 else np.inf
        )

        # Sharpe Ratio
        if len(profits) > 1 and np.std(profits) > 0:
            sharpe = np.sqrt(252 * 48) * np.mean(profits) / np.std(profits)
        else:
            sharpe = 0

        return {
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'total_return': sum(profits)
        }

    def _analyze_by_time(self) -> Dict:
        """Анализ по времени"""
        by_hour = defaultdict(list)
        by_day = defaultdict(list)

        for trade in self.trades:
            by_hour[trade.hour_of_day].append(trade)
            by_day[trade.day_of_week].append(trade)

        # Best hours
        hour_stats = {}
        for hour, trades in by_hour.items():
            if len(trades) >= 3:  # Minimum 3 trades
                win_rate = sum([1 for t in trades if t.success]) / len(trades) * 100
                avg_profit = np.mean([t.profit_pct for t in trades])
                hour_stats[hour] = {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'num_trades': len(trades)
                }

        # Best days
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_stats = {}
        for day, trades in by_day.items():
            if len(trades) >= 3:
                win_rate = sum([1 for t in trades if t.success]) / len(trades) * 100
                avg_profit = np.mean([t.profit_pct for t in trades])
                day_stats[day_names[day]] = {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'num_trades': len(trades)
                }

        # Find best times
        if hour_stats:
            best_hour = max(hour_stats.items(), key=lambda x: x[1]['win_rate'])
            worst_hour = min(hour_stats.items(), key=lambda x: x[1]['win_rate'])
        else:
            best_hour = None
            worst_hour = None

        if day_stats:
            best_day = max(day_stats.items(), key=lambda x: x[1]['win_rate'])
            worst_day = min(day_stats.items(), key=lambda x: x[1]['win_rate'])
        else:
            best_day = None
            worst_day = None

        return {
            'by_hour': hour_stats,
            'by_day': day_stats,
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'best_day': best_day,
            'worst_day': worst_day
        }

    def _analyze_market_conditions(self) -> Dict:
        """Анализ по рыночным условиям"""
        by_volatility = {'low': [], 'medium': [], 'high': []}
        by_trend = {'UP': [], 'DOWN': [], 'SIDEWAYS': []}

        for trade in self.trades:
            # Volatility buckets
            if trade.volatility < 0.3:
                by_volatility['low'].append(trade)
            elif trade.volatility < 0.6:
                by_volatility['medium'].append(trade)
            else:
                by_volatility['high'].append(trade)

            # Trend
            by_trend[trade.trend].append(trade)

        # Stats by volatility
        vol_stats = {}
        for vol_level, trades in by_volatility.items():
            if trades:
                win_rate = sum([1 for t in trades if t.success]) / len(trades) * 100
                avg_profit = np.mean([t.profit_pct for t in trades])
                vol_stats[vol_level] = {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'num_trades': len(trades)
                }

        # Stats by trend
        trend_stats = {}
        for trend, trades in by_trend.items():
            if trades:
                win_rate = sum([1 for t in trades if t.success]) / len(trades) * 100
                avg_profit = np.mean([t.profit_pct for t in trades])
                trend_stats[trend] = {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'num_trades': len(trades)
                }

        return {
            'by_volatility': vol_stats,
            'by_trend': trend_stats
        }

    def _analyze_indicators(self) -> Dict:
        """Анализ эффективности индикаторов"""
        # RSI analysis
        rsi_buckets = {'oversold': [], 'neutral': [], 'overbought': []}
        for trade in self.trades:
            if trade.rsi < 30:
                rsi_buckets['oversold'].append(trade)
            elif trade.rsi < 70:
                rsi_buckets['neutral'].append(trade)
            else:
                rsi_buckets['overbought'].append(trade)

        rsi_stats = {}
        for bucket, trades in rsi_buckets.items():
            if trades:
                win_rate = sum([1 for t in trades if t.success]) / len(trades) * 100
                rsi_stats[bucket] = {
                    'win_rate': win_rate,
                    'num_trades': len(trades)
                }

        # BB position analysis
        bb_buckets = {'lower': [], 'middle': [], 'upper': []}
        for trade in self.trades:
            if trade.bb_position < -0.3:
                bb_buckets['lower'].append(trade)
            elif trade.bb_position < 0.3:
                bb_buckets['middle'].append(trade)
            else:
                bb_buckets['upper'].append(trade)

        bb_stats = {}
        for bucket, trades in bb_buckets.items():
            if trades:
                win_rate = sum([1 for t in trades if t.success]) / len(trades) * 100
                bb_stats[bucket] = {
                    'win_rate': win_rate,
                    'num_trades': len(trades)
                }

        return {
            'rsi': rsi_stats,
            'bollinger_bands': bb_stats
        }

    def _analyze_hold_time(self) -> Dict:
        """Анализ времени удержания"""
        hold_times = [t.hold_time_minutes for t in self.trades]

        # Buckets
        buckets = {
            '0-30min': [],
            '30-60min': [],
            '1-2h': [],
            '2-4h': [],
            '4h+': []
        }

        for trade in self.trades:
            ht = trade.hold_time_minutes
            if ht < 30:
                buckets['0-30min'].append(trade)
            elif ht < 60:
                buckets['30-60min'].append(trade)
            elif ht < 120:
                buckets['1-2h'].append(trade)
            elif ht < 240:
                buckets['2-4h'].append(trade)
            else:
                buckets['4h+'].append(trade)

        bucket_stats = {}
        for bucket, trades in buckets.items():
            if trades:
                win_rate = sum([1 for t in trades if t.success]) / len(trades) * 100
                avg_profit = np.mean([t.profit_pct for t in trades])
                bucket_stats[bucket] = {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'num_trades': len(trades)
                }

        return {
            'avg_hold_time': np.mean(hold_times),
            'median_hold_time': np.median(hold_times),
            'by_bucket': bucket_stats
        }

    def _analyze_extremes(self) -> Dict:
        """Лучшие и худшие сделки"""
        best_trades = sorted(self.trades, key=lambda t: t.profit_pct, reverse=True)[:5]
        worst_trades = sorted(self.trades, key=lambda t: t.profit_pct)[:5]

        return {
            'best_trades': [t.to_dict() for t in best_trades],
            'worst_trades': [t.to_dict() for t in worst_trades]
        }

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []

        overall = results.get('overall', {})
        time_analysis = results.get('time_analysis', {})
        market = results.get('market_conditions', {})

        # Win rate recommendation
        if overall.get('win_rate', 0) < 55:
            recommendations.append(
                f"⚠️  Win Rate {overall['win_rate']:.1f}% is below target (55%). "
                "Consider: (1) Stricter entry conditions, (2) Better risk management, "
                "(3) Filter trades by best time/conditions."
            )
        else:
            recommendations.append(
                f"✅ Win Rate {overall['win_rate']:.1f}% is good! Keep current strategy."
            )

        # Time recommendation
        if time_analysis.get('best_hour'):
            hour, stats = time_analysis['best_hour']
            recommendations.append(
                f"⏰ Best trading hour: {hour}:00 "
                f"(WR={stats['win_rate']:.1f}%, Trades={stats['num_trades']}). "
                "Focus trading during this time."
            )

        if time_analysis.get('worst_hour'):
            hour, stats = time_analysis['worst_hour']
            recommendations.append(
                f"🚫 Avoid trading at {hour}:00 "
                f"(WR={stats['win_rate']:.1f}%). Consider skipping this period."
            )

        # Volatility recommendation
        vol_stats = market.get('by_volatility', {})
        if vol_stats:
            best_vol = max(vol_stats.items(), key=lambda x: x[1]['win_rate'])
            recommendations.append(
                f"📊 Best volatility: {best_vol[0]} "
                f"(WR={best_vol[1]['win_rate']:.1f}%). "
                "Trade more in these conditions."
            )

        # Profit factor
        if overall.get('profit_factor', 0) < 1.5:
            recommendations.append(
                f"💰 Profit Factor {overall.get('profit_factor', 0):.2f} is low. "
                "Target: >1.5. Improve by: (1) Cut losses faster, (2) Let winners run."
            )

        return recommendations

    def _print_analysis(self, results: Dict):
        """Вывод результатов анализа"""
        logger.info("\n" + "=" * 80)
        logger.info("📈 OVERALL PERFORMANCE")
        logger.info("=" * 80)

        overall = results['overall']
        logger.info(f"Win Rate: {overall['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {overall['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
        logger.info(f"Total Return: {overall['total_return']:+.2f}%")
        logger.info(f"Avg Win: {overall['avg_win']:+.2f}%")
        logger.info(f"Avg Loss: {overall['avg_loss']:+.2f}%")

        # Time analysis
        logger.info("\n" + "=" * 80)
        logger.info("⏰ TIME ANALYSIS")
        logger.info("=" * 80)

        time_analysis = results['time_analysis']
        if time_analysis.get('best_hour'):
            hour, stats = time_analysis['best_hour']
            logger.info(f"✅ Best Hour: {hour}:00 (WR={stats['win_rate']:.1f}%, {stats['num_trades']} trades)")
        if time_analysis.get('worst_hour'):
            hour, stats = time_analysis['worst_hour']
            logger.info(f"❌ Worst Hour: {hour}:00 (WR={stats['win_rate']:.1f}%)")

        # Market conditions
        logger.info("\n" + "=" * 80)
        logger.info("🌍 MARKET CONDITIONS")
        logger.info("=" * 80)

        market = results['market_conditions']
        for vol_level, stats in market['by_volatility'].items():
            logger.info(
                f"{vol_level.upper()} volatility: "
                f"WR={stats['win_rate']:.1f}%, Avg={stats['avg_profit']:+.2f}%"
            )

        # Recommendations
        logger.info("\n" + "=" * 80)
        logger.info("💡 RECOMMENDATIONS")
        logger.info("=" * 80)

        for i, rec in enumerate(results['recommendations'], 1):
            logger.info(f"{i}. {rec}")

        logger.info("=" * 80)

    def save_analysis(self, results: Dict, path: str):
        """Сохранить результаты анализа"""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Analysis saved to {path}")

    def plot_analysis(self, save_path: str = None):
        """Визуализация результатов"""
        if not self.trades:
            logger.warning("⚠️  No trades to plot!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trading Performance Analysis', fontsize=16)

        # 1. Win Rate by hour
        hours = [t.hour_of_day for t in self.trades]
        success = [t.success for t in self.trades]

        hour_wr = {}
        for h in range(24):
            hour_trades = [s for i, s in enumerate(success) if hours[i] == h]
            if hour_trades:
                hour_wr[h] = sum(hour_trades) / len(hour_trades) * 100

        if hour_wr:
            axes[0, 0].bar(hour_wr.keys(), hour_wr.values())
            axes[0, 0].axhline(y=50, color='r', linestyle='--', label='50%')
            axes[0, 0].set_title('Win Rate by Hour')
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Win Rate (%)')
            axes[0, 0].legend()

        # 2. Profit distribution
        profits = [t.profit_pct for t in self.trades]
        axes[0, 1].hist(profits, bins=30, edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].set_title('Profit Distribution')
        axes[0, 1].set_xlabel('Profit %')
        axes[0, 1].set_ylabel('Frequency')

        # 3. Cumulative returns
        cumulative = np.cumsum(profits)
        axes[1, 0].plot(cumulative)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative Return %')
        axes[1, 0].grid(True)

        # 4. Win Rate by volatility
        vol_bins = {'low': [], 'med': [], 'high': []}
        for t in self.trades:
            if t.volatility < 0.3:
                vol_bins['low'].append(t.success)
            elif t.volatility < 0.6:
                vol_bins['med'].append(t.success)
            else:
                vol_bins['high'].append(t.success)

        vol_wr = {}
        for vol, trades in vol_bins.items():
            if trades:
                vol_wr[vol] = sum(trades) / len(trades) * 100

        if vol_wr:
            axes[1, 1].bar(vol_wr.keys(), vol_wr.values())
            axes[1, 1].axhline(y=50, color='r', linestyle='--')
            axes[1, 1].set_title('Win Rate by Volatility')
            axes[1, 1].set_xlabel('Volatility Level')
            axes[1, 1].set_ylabel('Win Rate (%)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 Plot saved to {save_path}")
        else:
            plt.show()


# ==========================================
# 🚀 EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # Create analyzer
    analyzer = PerformanceAnalyzer()

    # Add some example trades
    np.random.seed(42)
    for i in range(100):
        trade = TradeAnalysis(
            trade_id=i,
            entry_time=f"2025-01-{i%30+1:02d} {i%24:02d}:00",
            exit_time=f"2025-01-{i%30+1:02d} {(i%24+2)%24:02d}:00",
            direction='LONG' if i % 2 == 0 else 'SHORT',
            entry_price=100.0,
            exit_price=100.0 + np.random.randn() * 2,
            profit_pct=np.random.randn() * 2,
            profit_usd=np.random.randn() * 20,
            hold_time_minutes=np.random.randint(30, 240),
            hour_of_day=i % 24,
            day_of_week=i % 7,
            volatility=np.random.uniform(0.2, 0.8),
            trend=np.random.choice(['UP', 'DOWN', 'SIDEWAYS']),
            rsi=np.random.uniform(20, 80),
            macd=np.random.randn(),
            bb_position=np.random.uniform(-1, 1),
            success=np.random.random() > 0.4
        )
        analyzer.add_trade(trade)

    # Analyze
    results = analyzer.analyze()

    # Save
    analyzer.save_analysis(results, 'data/performance_analysis.json')

    # Plot
    try:
        analyzer.plot_analysis('data/performance_plot.png')
    except:
        logger.warning("⚠️  Plotting requires matplotlib")
