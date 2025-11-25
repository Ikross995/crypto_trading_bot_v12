#!/usr/bin/env python3
"""
Тест ML Model Persistence
Демонстрирует, что модели сохраняются и загружаются между запусками
"""

import asyncio
import sys
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from strategy.ml_learning_system import (
    AdvancedMLLearningSystem,
    MarketContext,
    TradeOutcome,
    ML_AVAILABLE
)

class MockConfig:
    pass

async def simulate_training_cycle():
    """Симулирует цикл обучения ML моделей"""

    print("=" * 80)
    print("🧠 ML MODEL PERSISTENCE TEST")
    print("=" * 80)
    print(f"\n✅ ML Libraries Available: {ML_AVAILABLE}\n")

    # Создаем ML систему
    config = MockConfig()
    ml_system = AdvancedMLLearningSystem(config)

    print("\n📊 INITIAL STATE:")
    print("-" * 80)
    for name, model in ml_system.models.items():
        print(f"  {name:20s}: is_fitted={model.is_fitted}, samples_seen={model.samples_seen}")

    # Симулируем несколько сделок для обучения
    print("\n\n🎓 SIMULATING TRADES FOR TRAINING:")
    print("-" * 80)

    for i in range(15):  # 15 симулированных сделок
        # Создаем фейковый market context
        market_context = MarketContext(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            rsi_14=50.0 + np.random.randn() * 10,
            rsi_7=50.0 + np.random.randn() * 10,
            macd=np.random.randn(),
            macd_signal=np.random.randn(),
            bb_position=np.random.uniform(0, 1),
            sma_20=50000.0,
            ema_50=50100.0,
            atr_14=500.0,
            volume_ratio=1.0 + np.random.uniform(-0.5, 0.5),
            volatility_percentile=np.random.uniform(30, 70),
            trend_strength=np.random.uniform(0, 1),
            market_regime="trending",
            fear_greed_index=50,
            btc_dominance=45.0,
            hour_of_day=12,
            day_of_week=1,
            session="american",
            support_distance=2.0,
            resistance_distance=3.0,
            bid_ask_spread=0.01,
            order_book_imbalance=0.5
        )

        # Создаем фейковый trade outcome
        pnl_pct = np.random.randn() * 0.5  # -0.5% to +0.5%
        trade_outcome = TradeOutcome(
            trade_id=f"TEST_{i+1}",
            pnl=pnl_pct * 100,
            pnl_pct=pnl_pct,
            hold_time_minutes=30.0 + np.random.randn() * 10,
            exit_reason="tp" if pnl_pct > 0 else "sl",
            sharpe_ratio=1.5,
            max_favorable_excursion=abs(pnl_pct) * 1.5,
            max_adverse_excursion=abs(pnl_pct) * 0.5,
            win_probability=0.6,
            stress_level=0.3,
            confidence_decay=0.1
        )

        # Обучаем модель
        signal_strength = 1.5
        recent_performance = {
            'today_pnl_pct': 0.5,
            'recent_accuracy': 0.6,
            'recent_trades': i,
            'avg_hold_time': 30.0,
            'win_rate': 0.6,
            'profit_factor': 1.3
        }

        await ml_system.learn_from_trade(
            market_context=market_context,
            trade_outcome=trade_outcome,
            signal_strength=signal_strength,
            recent_performance=recent_performance
        )

        print(f"  Trade {i+1:2d}: PnL {pnl_pct:+.2f}% - Model trained")

    print("\n\n📊 STATE AFTER TRAINING:")
    print("-" * 80)
    total_samples = 0
    for name, model in ml_system.models.items():
        print(f"  {name:20s}: is_fitted={model.is_fitted}, samples_seen={model.samples_seen}")
        total_samples += model.samples_seen

    print(f"\n  Total samples across all models: {total_samples}")
    print(f"  Trade outcomes stored: {len(ml_system.trade_outcomes)}")
    print(f"  Market contexts stored: {len(ml_system.market_contexts)}")

    # Сохраняем модели
    print("\n\n💾 SAVING MODELS:")
    print("-" * 80)
    ml_system.save_data()

    # Проверяем сохраненные файлы
    data_dir = Path("ml_learning_data")
    saved_files = list(data_dir.glob("*"))
    print(f"  Files saved: {len(saved_files)}")
    for f in sorted(saved_files):
        size = f.stat().st_size
        print(f"    - {f.name:30s} ({size:,} bytes)")

    # Создаем новую систему и загружаем
    print("\n\n🔄 LOADING MODELS (NEW INSTANCE):")
    print("-" * 80)

    ml_system_new = AdvancedMLLearningSystem(config)

    print("\n📊 STATE AFTER LOADING:")
    print("-" * 80)
    loaded_samples = 0
    for name, model in ml_system_new.models.items():
        print(f"  {name:20s}: is_fitted={model.is_fitted}, samples_seen={model.samples_seen}")
        loaded_samples += model.samples_seen

    # Проверяем, что загрузилось правильно
    print("\n\n✅ VERIFICATION:")
    print("-" * 80)
    if loaded_samples == total_samples:
        print(f"  ✅ SUCCESS! Loaded {loaded_samples} samples (same as saved {total_samples})")
        print(f"  ✅ Models are persistent across restarts!")
    else:
        print(f"  ❌ MISMATCH! Loaded {loaded_samples} but expected {total_samples}")
        return False

    # Тестируем предсказание
    print("\n\n🎯 TESTING PREDICTION WITH LOADED MODEL:")
    print("-" * 80)

    test_context = MarketContext(
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        rsi_14=55.0, rsi_7=60.0, macd=0.5, macd_signal=0.3,
        bb_position=0.7, sma_20=50000.0, ema_50=50100.0, atr_14=500.0,
        volume_ratio=1.2, volatility_percentile=50.0, trend_strength=0.8,
        market_regime="trending", fear_greed_index=55, btc_dominance=45.0,
        hour_of_day=14, day_of_week=2, session="american",
        support_distance=2.0, resistance_distance=3.0,
        bid_ask_spread=0.01, order_book_imbalance=0.5
    )

    prediction = await ml_system_new.predict_trade_outcome(
        market_context=test_context,
        signal_strength=1.5,
        recent_performance=recent_performance
    )

    print(f"  Expected PnL: {prediction['expected_pnl_pct']:+.2f}%")
    print(f"  Win Probability: {prediction['win_probability']:.1%}")
    print(f"  Hold Time: {prediction['expected_hold_time']:.1f} min")
    print(f"  Risk Score: {prediction['risk_score']:.2f}")
    print(f"  Confidence: {prediction['prediction_confidence']:.2f}")

    print("\n" + "=" * 80)
    print("✅ ML MODEL PERSISTENCE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = asyncio.run(simulate_training_cycle())
    sys.exit(0 if success else 1)
