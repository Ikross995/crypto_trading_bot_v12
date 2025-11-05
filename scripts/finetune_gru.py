#!/usr/bin/env python3
"""
🔄 Fine-tune GRU Model on Fresh Data
====================================

Дотренировывает существующую GRU модель на свежих данных.

Использование:
    # Дотренировать на последних 30 днях (5 эпох)
    python scripts/finetune_gru.py --days 30 --epochs 5

    # Дотренировать на конкретной паре
    python scripts/finetune_gru.py --symbols BTCUSDT --days 60 --epochs 10

    # Дотренировать с другим learning rate
    python scripts/finetune_gru.py --days 30 --lr 0.0001

Преимущества:
- Адаптация к новым рыночным условиям
- Быстрое обучение (5-10 минут)
- Сохраняет уже выученные паттерны
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import numpy as np
import pandas as pd

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Импорты из основного скрипта
from examples.gru_training_pytorch import (
    BinanceDataDownloader,
    calculate_technical_indicators,
    GRUPriceModel,
    PriceDataset,
    prepare_sequences,
    configure_gpu
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_existing_model(model_path: str, device: torch.device):
    """
    Загрузить существующую обученную модель.

    Args:
        model_path: Путь к сохранённой модели
        device: Устройство (cuda/cpu)

    Returns:
        model: Загруженная модель
        config: Конфигурация модели
    """
    logger.info(f"📂 Loading existing model from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Загружаем checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Извлекаем конфигурацию
    config = checkpoint['model_config']

    # Создаём модель с той же архитектурой
    model = GRUPriceModel(
        input_features=config['input_features'],
        sequence_length=config['sequence_length']
    ).to(device)

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info("✅ Model loaded successfully")
    logger.info(f"   Input features: {config['input_features']}")
    logger.info(f"   Sequence length: {config['sequence_length']}")

    if 'final_metrics' in checkpoint:
        metrics = checkpoint['final_metrics']
        logger.info(f"   Previous MAPE: {metrics.get('mape', 0):.2f}%")

    return model, config


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple:
    """Обучение на одной эпохе"""
    model.train()
    train_losses = []
    train_maes = []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions.squeeze(), batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Метрики
        train_losses.append(loss.item())
        mae = torch.mean(torch.abs(predictions.squeeze() - batch_y)).item()
        train_maes.append(mae)

    return np.mean(train_losses), np.mean(train_maes)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Валидация"""
    model.eval()
    val_losses = []
    val_maes = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)

            val_losses.append(loss.item())
            mae = torch.mean(torch.abs(predictions.squeeze() - batch_y)).item()
            val_maes.append(mae)

    return np.mean(val_losses), np.mean(val_maes)


async def finetune_model(
    model_path: str = "models/checkpoints/gru_model_pytorch.pt",
    symbols: List[str] = None,
    days: int = 30,
    interval: str = "1m",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    save_path: Optional[str] = None
):
    """
    Дотренировать существующую модель на свежих данных.

    Args:
        model_path: Путь к существующей модели
        symbols: Список пар для дотренировки
        days: Количество дней свежих данных
        interval: Таймфрейм
        epochs: Количество эпох дотренировки
        batch_size: Размер батча
        learning_rate: Learning rate (меньше чем при первичном обучении!)
        save_path: Куда сохранить (если None - перезаписывает исходную)
    """
    logger.info("=" * 80)
    logger.info("🔄 Fine-tuning GRU Model on Fresh Data")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Existing model: {model_path}")
    logger.info(f"   Fresh data: {days} days")
    logger.info(f"   Symbols: {symbols if symbols else 'Same as original'}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Learning rate: {learning_rate} (low for fine-tuning)")
    logger.info("=" * 80)

    # Настройка GPU
    device = configure_gpu()

    # Загружаем существующую модель
    model, config = load_existing_model(model_path, device)

    # Используем те же символы что и при обучении
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'LINKUSDT', 'MATICUSDT'
        ]

    # Загрузка СВЕЖИХ данных
    logger.info(f"📥 Downloading fresh data ({days} days)...")
    downloader = BinanceDataDownloader()
    all_data = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"📥 Downloading {symbol} ({i}/{len(symbols)})...")
        df = await downloader.download_historical_data(symbol, interval, days)

        if len(df) > 0:
            df = calculate_technical_indicators(df)
            all_data.append(df)
        else:
            logger.warning(f"⚠️  Skipping {symbol} - no data")

    # Объединяем данные
    logger.info("🔗 Combining fresh data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ Fresh dataset: {len(combined_df):,} samples")

    # Используем те же фичи что и в оригинальной модели
    feature_columns = config['feature_columns']
    sequence_length = config['sequence_length']

    logger.info(f"📊 Using same features: {len(feature_columns)}")

    # Подготовка последовательностей
    X, y = prepare_sequences(combined_df, feature_columns, sequence_length)

    # Train/Val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"📊 Training samples: {len(X_train):,}")
    logger.info(f"📊 Validation samples: {len(X_val):,}")

    # Создаём DataLoaders
    train_dataset = PriceDataset(X_train, y_train)
    val_dataset = PriceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Оптимизатор и функция потерь
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("🎯 Starting fine-tuning...")
    logger.info(f"   Note: Using LOW learning rate ({learning_rate}) to preserve learned patterns")

    # Fine-tuning
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        # Training
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)

        # Логирование
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Train MAE: {train_mae:.2f} | "
            f"Val MAE: {val_mae:.2f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"   💾 New best! Val Loss: {best_val_loss:.6f}")

    total_time = time.time() - start_time
    logger.info(f"✅ Fine-tuning completed in {total_time/60:.1f} minutes")

    # Финальная оценка
    logger.info("📊 Final evaluation on validation set...")
    model.eval()
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            val_predictions.extend(predictions.cpu().numpy())
            val_targets.extend(batch_y.numpy())

    val_predictions = np.array(val_predictions).flatten()
    val_targets = np.array(val_targets)

    # Метрики
    mse = np.mean((val_predictions - val_targets) ** 2)
    mae = np.mean(np.abs(val_predictions - val_targets))
    mape = np.mean(np.abs((val_targets - val_predictions) / val_targets)) * 100

    logger.info("=" * 80)
    logger.info("📊 Final metrics after fine-tuning:")
    logger.info(f"   - MSE: {mse:.6f}")
    logger.info(f"   - MAE: {mae:.2f}")
    logger.info(f"   - MAPE: {mape:.2f}%")
    logger.info("=" * 80)

    # Сохранение
    if save_path is None:
        save_path = model_path

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем с обновлённой историей
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config,
        'finetuning_info': {
            'finetuned_on': datetime.now().isoformat(),
            'fresh_data_days': days,
            'finetune_epochs': epochs,
            'learning_rate': learning_rate
        },
        'final_metrics': {
            'mse': mse,
            'mae': mae,
            'mape': mape
        }
    }, save_path)

    logger.info(f"✅ Fine-tuned model saved to: {save_path}")
    logger.info("=" * 80)
    logger.info("🎉 Fine-tuning completed successfully!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 Model is now updated with fresh market data!")
    logger.info("   You can continue using it in your bot without any changes.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GRU model on fresh data")
    parser.add_argument('--model', type=str, default='models/checkpoints/gru_model_pytorch.pt',
                        help='Path to existing model')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of fresh data to train on (default: 30)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of fine-tuning epochs (default: 5)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Symbols to train on (default: same as original)')
    parser.add_argument('--interval', type=str, default='1m',
                        help='Timeframe (default: 1m)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001 - low for fine-tuning)')
    parser.add_argument('--save-as', type=str,
                        help='Save to different path (default: overwrite original)')

    args = parser.parse_args()

    asyncio.run(finetune_model(
        model_path=args.model,
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_as
    ))


if __name__ == "__main__":
    main()
