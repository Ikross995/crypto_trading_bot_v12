"""
Тестовый скрипт для проверки распознавания трендов ML системой
"""

import numpy as np
from datetime import datetime, timezone
from strategy.ml_learning_system import (
    AdvancedMLLearningSystem,
    MarketContext,
    TradeOutcome
)
from core.config import Config


def create_test_market_context(trend_type='up'):
    """Создает тестовый контекст рынка с заданным трендом"""

    if trend_type == 'up':
        # Восходящий тренд
        return MarketContext(
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            rsi_14=65,
            rsi_7=70,
            macd=50,
            macd_signal=30,
            bb_position=0.8,
            sma_20=50000,
            ema_50=51000,  # EMA выше SMA = восходящий тренд
            atr_14=500,
            volume_ratio=1.5,
            volatility_percentile=60,
            trend_strength=0.7,
            market_regime='trending',
            fear_greed_index=65,
            btc_dominance=45,
            hour_of_day=14,
            day_of_week=2,
            session='european',
            support_distance=5,
            resistance_distance=10,
            bid_ask_spread=0.01,
            order_book_imbalance=0.6
        )
    elif trend_type == 'down':
        # Нисходящий тренд
        return MarketContext(
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            rsi_14=35,
            rsi_7=30,
            macd=-50,
            macd_signal=-30,
            bb_position=0.2,
            sma_20=50000,
            ema_50=49000,  # EMA ниже SMA = нисходящий тренд
            atr_14=500,
            volume_ratio=1.5,
            volatility_percentile=60,
            trend_strength=-0.7,
            market_regime='trending',
            fear_greed_index=35,
            btc_dominance=45,
            hour_of_day=14,
            day_of_week=2,
            session='european',
            support_distance=10,
            resistance_distance=5,
            bid_ask_spread=0.01,
            order_book_imbalance=0.4
        )
    else:  # sideways
        # Боковой тренд
        return MarketContext(
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            rsi_14=50,
            rsi_7=50,
            macd=0,
            macd_signal=0,
            bb_position=0.5,
            sma_20=50000,
            ema_50=50000,  # EMA = SMA = флэт
            atr_14=300,
            volume_ratio=0.8,
            volatility_percentile=30,
            trend_strength=0.1,
            market_regime='ranging',
            fear_greed_index=50,
            btc_dominance=45,
            hour_of_day=14,
            day_of_week=2,
            session='european',
            support_distance=5,
            resistance_distance=5,
            bid_ask_spread=0.01,
            order_book_imbalance=0.5
        )


def create_test_trade_outcome(trend_type='up'):
    """Создает тестовый результат сделки"""

    if trend_type == 'up':
        # Прибыльная сделка на восходящем тренде
        return TradeOutcome(
            trade_id='test_up_1',
            pnl=100,
            pnl_pct=2.5,
            hold_time_minutes=45,
            exit_reason='take_profit',
            sharpe_ratio=2.0,
            max_favorable_excursion=3.0,
            max_adverse_excursion=0.5,
            win_probability=0.7,
            stress_level=0.2,
            confidence_decay=0.1
        )
    elif trend_type == 'down':
        # Прибыльная сделка на нисходящем тренде (шорт)
        return TradeOutcome(
            trade_id='test_down_1',
            pnl=80,
            pnl_pct=-2.0,  # Отрицательное движение цены
            hold_time_minutes=60,
            exit_reason='take_profit',
            sharpe_ratio=1.8,
            max_favorable_excursion=2.5,
            max_adverse_excursion=0.4,
            win_probability=0.65,
            stress_level=0.3,
            confidence_decay=0.15
        )
    else:  # sideways
        # Небольшая прибыль на боковом тренде
        return TradeOutcome(
            trade_id='test_sideways_1',
            pnl=20,
            pnl_pct=0.5,
            hold_time_minutes=30,
            exit_reason='take_profit',
            sharpe_ratio=1.2,
            max_favorable_excursion=0.8,
            max_adverse_excursion=0.3,
            win_probability=0.55,
            stress_level=0.4,
            confidence_decay=0.2
        )


async def test_trend_recognition():
    """Тестирует распознавание трендов"""

    print("=" * 80)
    print("🧪 ТЕСТ РАСПОЗНАВАНИЯ ТРЕНДОВ")
    print("=" * 80)

    # Инициализация ML системы
    config = Config()
    ml_system = AdvancedMLLearningSystem(config)

    print("\n📚 [ОБУЧЕНИЕ] Обучаем модель на примерах трендов...")

    # Обучаем на разных типах трендов
    trends = ['up', 'down', 'sideways']

    for trend in trends:
        print(f"\n  Обучение на тренде: {trend.upper()}")
        for i in range(20):  # 20 примеров каждого типа
            ctx = create_test_market_context(trend)
            outcome = create_test_trade_outcome(trend)

            await ml_system.learn_from_trade(
                ctx,
                outcome,
                signal_strength=0.7,
                recent_performance={'recent_accuracy': 0.6}
            )

    print("\n✅ Обучение завершено!")

    # Тестируем предсказания
    print("\n" + "=" * 80)
    print("🎯 [ПРЕДСКАЗАНИЯ] Тестируем распознавание трендов")
    print("=" * 80)

    for trend in trends:
        print(f"\n📊 Тестирование тренда: {trend.upper()}")
        print("-" * 80)

        ctx = create_test_market_context(trend)
        prediction = await ml_system.predict_trade_outcome(
            ctx,
            signal_strength=0.7,
            recent_performance={'recent_accuracy': 0.6}
        )

        print(f"  Направление: {prediction['trend_direction']}")
        print(f"  Класс: {prediction['trend_direction_class']} (0=DOWN, 1=SIDEWAYS, 2=UP)")
        print(f"  Уверенность: {prediction['trend_confidence']:.2%}")
        print(f"  Сила тренда: {prediction['trend_strength']:.2f}")
        print(f"  Вероятности:")
        print(f"    - DOWN: {prediction['trend_probabilities']['down']:.2%}")
        print(f"    - SIDEWAYS: {prediction['trend_probabilities']['sideways']:.2%}")
        print(f"    - UP: {prediction['trend_probabilities']['up']:.2%}")
        print(f"  Ожидаемый PnL: {prediction['expected_pnl_pct']:+.2f}%")
        print(f"  Win Probability: {prediction['win_probability']:.2%}")

    print("\n" + "=" * 80)
    print("✅ ТЕСТ ЗАВЕРШЕН!")
    print("=" * 80)

    # Сохраняем обученные модели
    ml_system.save_data()
    print("\n💾 Модели сохранены")


if __name__ == '__main__':
    import asyncio
    asyncio.run(test_trend_recognition())
