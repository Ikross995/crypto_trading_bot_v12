#!/usr/bin/env python3
"""
🔥 FINAL GRU Training Script - PERCENTAGE CHANGE + ENHANCED MODEL
==================================================================

ВСЕ ИСПРАВЛЕНИЯ:
✅ Предсказываем % изменение (не абсолютную цену)
✅ Усиленная архитектура (400K параметров)
✅ 3 GRU слоя + BatchNorm
✅ Свежие данные (180 дней)
✅ 30m таймфрейм (для trading)

USAGE:
    python train_gru_final.py --days 180 --epochs 30 --batch-size 1024

EXPECTED RESULTS:
- Direction Accuracy: >55% (хорошо)
- MAE: <1.5% (отлично)
- Обучение: ~1-2 часа на RTX 5070 Ti (batch_size=1024!)

RECOMMENDED BATCH SIZES:
- RTX 5070 Ti (16GB): 1024-2048 ⚡ МАКСИМАЛЬНАЯ СКОРОСТЬ
- RTX 4090 (24GB): 2048-4096
- RTX 3080 (10GB): 512-1024
- GTX 1080 (8GB): 256-512

"""

# Весь код из gru_training_pytorch_v2_percentage.py
# + замена GRUPriceModel на EnhancedGRUModel

# 🔥 Set flag to prevent base script's __main__ block from running
__SKIP_MAIN__ = True

# 🔥 ИСПРАВЛЕНИЕ: Windows encoding issue
exec(open('examples/gru_training_pytorch_v2_percentage.py', encoding='utf-8').read())

# Импорт усиленной модели
from models.gru_model_enhanced import EnhancedGRUModel

# Override создание модели
original_train = train_gru_percentage_model

async def train_gru_final(*args, **kwargs):
    """Wrapper с усиленной моделью"""
    
    # Патчим глобальную переменную
    import sys
    current_module = sys.modules[__name__]
    
    # Подменяем класс модели
    original_model = current_module.GRUPriceModel if hasattr(current_module, 'GRUPriceModel') else None
    current_module.GRUPriceModel = EnhancedGRUModel
    
    logger.info("🔥 Using ENHANCED GRU Model (400K params)")
    
    # Меняем путь сохранения
    if 'save_path' not in kwargs:
        kwargs['save_path'] = "models/checkpoints/gru_model_final.pt"
    
    # Запускаем обучение
    result = await original_train(*args, **kwargs)
    
    # Восстанавливаем
    if original_model:
        current_module.GRUPriceModel = original_model
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🔥 Train FINAL GRU model (% change + enhanced architecture)"
    )
    parser.add_argument('--days', type=int, default=180,
                        help='Days of historical data (default: 180)')
    parser.add_argument('--interval', type=str, default='30m',
                        help='Timeframe: 1m, 5m, 15m, 30m, 1h, 4h (default: 30m)')
    parser.add_argument('--sequence-length', type=int, default=60,
                        help='Sequence length for LSTM/GRU (default: 60)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024 for RTX 5070 Ti - MAX SPEED!)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Symbols (default: top 10)')

    args = parser.parse_args()

    import asyncio
    asyncio.run(train_gru_final(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    ))
