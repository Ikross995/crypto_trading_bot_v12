#!/usr/bin/env python3
"""
🧠 GRU Model Training v2: PERCENTAGE CHANGE PREDICTION (FIXED!)
================================================================

ИЗМЕНЕНИЕ: Предсказываем % ИЗМЕНЕНИЕ вместо абсолютной цены!

Преимущества:
- ✅ Работает для ВСЕХ монет (BTC $103k, DOGE $0.16)
- ✅ Один scaler для всех символов
- ✅ Модель учит паттерны движения, не абсолютные цены
- ✅ Нет отрицательных предсказаний
- ✅ Универсальность

Использование:
    python examples/gru_training_pytorch_v2_percentage.py --days 180 --epochs 30 --batch-size 1024

BATCH SIZE для GPU:
- RTX 5070 Ti (16GB): 1024-2048 ⚡
- RTX 4090 (24GB): 2048-4096 ⚡⚡
- RTX 3080 (10GB): 512-1024
- GTX 1080 (8GB): 256-512

Автор: Claude + User
Дата: 2025-11-06
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("Install scikit-learn: pip install scikit-learn")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 🔥 IMPORT EXISTING COMPONENTS
# ==========================================

# Импортируем существующий код из старого файла
try:
    # 🔥 Pass __SKIP_MAIN__ flag to prevent old script's argparse from running
    _old_globals = globals()
    _old_globals['__SKIP_MAIN__'] = True
    exec(open('examples/gru_training_pytorch.py', encoding='utf-8').read(), _old_globals)
    logger.info("✅ Imported existing training components")
except Exception as e:
    logger.error(f"❌ Failed to import base training script: {e}")
    logger.error("Make sure examples/gru_training_pytorch.py exists!")
    sys.exit(1)


# ==========================================
# 🔥 OVERRIDE: PERCENTAGE-BASED SEQUENCES
# ==========================================

def prepare_sequences_percentage(
    df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, Dict]:
    """
    🔥 НОВАЯ ВЕРСИЯ: Подготовить последовательности для обучения GRU.

    ИЗМЕНЕНИЕ: Предсказываем % ИЗМЕНЕНИЕ, а не абсолютную цену!

    Args:
        df: DataFrame с features и close
        feature_columns: Список колонок-фичей
        sequence_length: Длина последовательности

    Returns:
        X: (samples, sequence_length, features) - нормализованные
        y: (samples,) - нормализованные % изменения
        feature_scaler: Scaler для features
        target_scaler: Scaler для % изменений
        stats: Статистика для логирования
    """
    logger.info(f"📦 Preparing sequences with PERCENTAGE CHANGE target (length={sequence_length})...")

    # 🔥 КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Вычисляем % изменение
    logger.info("🔥 Computing percentage price changes...")
    df = df.copy()
    df['price_change_pct'] = ((df['close'].shift(-1) - df['close']) / df['close']) * 100

    # Удаляем последнюю строку (нет следующей цены для расчета)
    df = df.dropna(subset=['price_change_pct'])

    # 🔥 КРИТИЧЕСКОЕ: Клипаем экстремальные % изменения
    # BTC меняется на ±1%, DOGE на ±10% → нужна одинаковая шкала!
    MAX_PCT_CHANGE = 10.0  # Реалистичный максимум для 30m таймфрейма

    logger.info(f"📊 Price change statistics (BEFORE clipping):")
    logger.info(f"   Min: {df['price_change_pct'].min():.2f}%")
    logger.info(f"   Max: {df['price_change_pct'].max():.2f}%")
    logger.info(f"   Mean: {df['price_change_pct'].mean():.2f}%")
    logger.info(f"   Std: {df['price_change_pct'].std():.2f}%")

    # Клипаем к ±10%
    df['price_change_pct'] = df['price_change_pct'].clip(-MAX_PCT_CHANGE, MAX_PCT_CHANGE)

    # Статистика % изменений ПОСЛЕ клипания
    stats = {
        'min_pct': df['price_change_pct'].min(),
        'max_pct': df['price_change_pct'].max(),
        'mean_pct': df['price_change_pct'].mean(),
        'std_pct': df['price_change_pct'].std(),
        'median_pct': df['price_change_pct'].median()
    }

    logger.info(f"📊 Price change statistics (AFTER clipping to ±{MAX_PCT_CHANGE}%):")
    logger.info(f"   Min: {stats['min_pct']:.2f}%")
    logger.info(f"   Max: {stats['max_pct']:.2f}%")
    logger.info(f"   Mean: {stats['mean_pct']:.2f}%")
    logger.info(f"   Median: {stats['median_pct']:.2f}%")
    logger.info(f"   Std: {stats['std_pct']:.2f}%")
    logger.info(f"   🔥 ALL COINS NOW ON SAME SCALE!")

    # Нормализация features (0-1)
    logger.info("🔄 Normalizing features to 0-1 range...")
    feature_scaler = MinMaxScaler()
    features_normalized = feature_scaler.fit_transform(df[feature_columns])

    # 🔥 Нормализация % изменений (0-1)
    # NOTE: MinMaxScaler автоматически найдет min/max в данных
    target_scaler = MinMaxScaler()
    target_normalized = target_scaler.fit_transform(df[['price_change_pct']]).flatten()

    logger.info(f"   Feature range: {features_normalized.min():.4f} - {features_normalized.max():.4f}")
    logger.info(f"   Target (% change) range after normalization: {target_normalized.min():.4f} - {target_normalized.max():.4f}")
    logger.info(f"   Target scaler fitted on: {target_scaler.data_min_[0]:.2f}% to {target_scaler.data_max_[0]:.2f}%")

    # Создаём последовательности
    X, y = [], []

    for i in range(len(df) - sequence_length):
        X.append(features_normalized[i:i + sequence_length])
        y.append(target_normalized[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"✅ Sequences prepared:")
    logger.info(f"   X shape: {X.shape}")
    logger.info(f"   y shape: {y.shape}")
    logger.info(f"   🔥 Target: PERCENTAGE CHANGE (not absolute price!)")

    return X, y, feature_scaler, target_scaler, stats


# ==========================================
# 🔥 MAIN TRAINING FUNCTION (OVERRIDDEN)
# ==========================================

async def train_gru_percentage_model(
    symbols: List[str] = None,
    days: int = 180,  # 6 месяцев свежих данных
    interval: str = "30m",  # 30-минутный таймфрейм
    sequence_length: int = 60,
    epochs: int = 30,
    batch_size: int = 1024,  # 🔥 ОГРОМНЫЙ для RTX 5070 Ti - MAX SPEED!
    save_path: str = "models/checkpoints/gru_model_pytorch_v2_percentage.pt",
    use_cache: bool = False
):
    """
    🔥 Обучить GRU модель на % изменениях (ПРАВИЛЬНО!)

    Args:
        symbols: Список торговых пар
        days: Количество дней истории (180 = 6 месяцев СВЕЖИХ данных)
        interval: Таймфрейм (30m для trading)
        sequence_length: Длина последовательности
        epochs: Количество эпох обучения
        batch_size: Размер батча (128 для GPU)
        save_path: Путь для сохранения модели
    """
    # Default symbols
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'LINKUSDT', 'APTUSDT'
        ]

    logger.info("=" * 80)
    logger.info("🔥 GRU Model Training v2: PERCENTAGE CHANGE PREDICTION")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    logger.info(f"   Days: {days} (last {days} days of FRESH data)")
    logger.info(f"   Interval: {interval}")
    logger.info(f"   Sequence length: {sequence_length}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   🔥 Target: PERCENTAGE CHANGE (not price!)")
    logger.info("=" * 80)

    # Настройка GPU
    device = configure_gpu()

    # Загрузка данных
    downloader = BinanceDataDownloader()
    all_data = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"📥 Downloading {symbol} ({i}/{len(symbols)})...")
        df = await downloader.download_historical_data(symbol, interval, days)

        if len(df) > 0:
            df = calculate_technical_indicators(df)
            all_data.append(df)
            logger.info(f"   ✅ {symbol}: {len(df):,} candles")
        else:
            logger.warning(f"⚠️  Skipping {symbol} - no data")

    # Объединяем данные
    logger.info("🔗 Combining data from all symbols...")
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ Combined dataset: {len(combined_df):,} samples")

    # Список фичей (22 indicators)
    feature_columns = [
        # Price features (4)
        'open', 'high', 'low', 'volume',
        # Technical indicators (11)
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_mid', 'bb_lower',
        'sma_20', 'sma_50', 'ema_50',
        'volume_sma', 'atr',
        # Volume indicators (7)
        'volume_delta', 'obv', 'volume_ratio',
        'volume_spike', 'mfi', 'cvd', 'vwap_distance'
    ]

    logger.info(f"📊 Features: {len(feature_columns)} total (15 price + 7 volume)")

    # 🔥 Подготовка последовательностей (НОВАЯ ВЕРСИЯ с %)
    X, y, feature_scaler, target_scaler, stats = prepare_sequences_percentage(
        combined_df, feature_columns, sequence_length
    )

    # Train/Test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"📊 Train samples: {len(X_train):,}")
    logger.info(f"📊 Test samples: {len(X_test):,}")

    # Создаём DataLoaders
    train_dataset = PriceDataset(X_train, y_train)
    test_dataset = PriceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Создаём модель
    logger.info("🧠 Building GRU model...")
    model = GRUPriceModel(
        input_features=len(feature_columns),
        sequence_length=sequence_length
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model parameters: {total_params:,}")

    # Обучение
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=0.001
    )

    # Финальная оценка
    logger.info("📊 Final evaluation on test set...")
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(batch_y.numpy())

    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets)

    # 🔥 Денормализация обратно в % изменения
    logger.info("🔄 Denormalizing predictions back to percentage changes...")
    test_predictions_pct = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    test_targets_pct = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

    # Метрики на % изменениях
    mae_pct = np.mean(np.abs(test_predictions_pct - test_targets_pct))
    mse_pct = np.mean((test_predictions_pct - test_targets_pct) ** 2)
    rmse_pct = np.sqrt(mse_pct)

    # Точность направления (самая важная метрика!)
    direction_correct = np.sum(np.sign(test_predictions_pct) == np.sign(test_targets_pct))
    direction_accuracy = direction_correct / len(test_predictions_pct) * 100

    logger.info("=" * 80)
    logger.info("📊 Final metrics (PERCENTAGE CHANGES):")
    logger.info(f"   🔥 Direction Accuracy: {direction_accuracy:.2f}%  ← MOST IMPORTANT!")
    logger.info(f"   - MAE: {mae_pct:.2f}%")
    logger.info(f"   - RMSE: {rmse_pct:.2f}%")
    logger.info(f"   - MSE: {mse_pct:.2f}%²")
    logger.info("=" * 80)

    # Сохранение модели
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_features': len(feature_columns),
            'sequence_length': sequence_length,
            'feature_columns': feature_columns,
            'model_version': 'v2_percentage',  # 🔥 Метка версии
            'target_type': 'percentage_change'  # 🔥 Тип target
        },
        'scalers': {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        },
        'training_history': history,
        'final_metrics': {
            'mae_pct': mae_pct,
            'rmse_pct': rmse_pct,
            'mse_pct': mse_pct,
            'direction_accuracy': direction_accuracy
        },
        'percentage_stats': stats  # 🔥 Статистика % изменений
    }, save_path)

    logger.info(f"✅ Model saved to: {save_path}")
    logger.info(f"   Model size: {Path(save_path).stat().st_size / 1024 / 1024:.1f} MB")

    logger.info("=" * 80)
    logger.info("🎉 Training completed successfully!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 Next steps:")
    logger.info(f"   1. Update .env: GRU_ENABLE=true")
    logger.info(f"   2. Update .env: GRU_MODEL_PATH={save_path}")
    logger.info(f"   3. Update gru_predictor_pytorch.py to use % predictions")
    logger.info(f"   4. Run bot: python cli.py live --timeframe 30m --use-imba")


# ==========================================
# 🚀 MAIN
# ==========================================

# 🔥 Only run if NOT being imported by train_gru_final.py
if __name__ == "__main__" and not globals().get('__SKIP_MAIN__'):
    import argparse

    parser = argparse.ArgumentParser(description="Train GRU model on % changes (v2)")
    parser.add_argument('--days', type=int, default=180,
                        help='Days of historical data (default: 180 = 6 months fresh data)')
    parser.add_argument('--interval', type=str, default='30m',
                        help='Timeframe: 1m, 5m, 15m, 30m, 1h, 4h (default: 30m)')
    parser.add_argument('--sequence-length', type=int, default=60,
                        help='Sequence length for LSTM/GRU (default: 60)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024 for RTX 5070 Ti - MAX SPEED!)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Symbols to train on (default: top 10)')

    args = parser.parse_args()

    asyncio.run(train_gru_percentage_model(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    ))
