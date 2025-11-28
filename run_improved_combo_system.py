#!/usr/bin/env python3
"""
🚀 IMPROVED COMBO SYSTEM - Современная ML архитектура 2025
===========================================================

Запуск улучшенной COMBO системы с:
- Multi-Head Attention
- Bidirectional GRU
- Mixed Precision Training
- Advanced Learning Rate Scheduling
- Data Augmentation

Usage:
    python run_improved_combo_system.py --symbols BTCUSDT --quick
    python run_improved_combo_system.py --symbols BTCUSDT,ETHUSDT --epochs 30
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from examples.improved_ensemble_trainer import ImprovedEnsembleTrainer
from examples.gru_training_pytorch import BinanceDataDownloader, calculate_technical_indicators
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def train_improved_ensemble(
    symbol: str,
    days: int = 180,
    interval: str = '30m',
    epochs: int = 30,
    batch_size: int = 512,
    quick: bool = False
):
    """
    Обучить улучшенный ансамбль для одного символа
    """
    logger.info("\n" + "🔥" * 80)
    logger.info(f"🎯 TRAINING IMPROVED ENSEMBLE: {symbol}")
    logger.info("🔥" * 80)

    # 1. Load data
    logger.info(f"📥 Loading {symbol} data...")
    downloader = BinanceDataDownloader()
    df = await downloader.download_historical_data(symbol, interval, days)

    if len(df) == 0:
        logger.error(f"❌ No data for {symbol}")
        return

    # 2. Calculate indicators
    df = calculate_technical_indicators(df)
    logger.info(f"✅ Data prepared: {len(df):,} candles")

    # 3. Prepare sequences
    from examples.gru_training_pytorch import prepare_sequences

    feature_columns = [
        'open', 'high', 'low', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_mid', 'bb_lower',
        'sma_20', 'sma_50', 'ema_50',
        'adx', 'volume_sma', 'obv', 'mfi',
        'atr', 'stoch_k', 'stoch_d', 'cci'
    ]

    X_train, y_train, X_val, y_val, X_test, y_test, scalers = prepare_sequences(
        df, feature_columns, sequence_length=60, train_split=0.7, val_split=0.15
    )

    logger.info(f"📊 Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 4. Train improved ensemble
    trainer = ImprovedEnsembleTrainer()

    if quick:
        epochs = 10
        logger.info("⚡ Quick mode: 10 epochs")

    results = await trainer.train_ensemble(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        accumulation_steps=1
    )

    # 5. Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TESTING ON HOLD-OUT SET")
    logger.info("=" * 80)

    import torch
    X_test_t = torch.FloatTensor(X_test).to(trainer.device)
    y_test_t = torch.FloatTensor(y_test).to(trainer.device)

    for name, model in trainer.models.items():
        model.eval()
        with torch.no_grad():
            if trainer.scaler is not None:
                from torch.cuda.amp import autocast
                with autocast():
                    predictions = model(X_test_t).cpu().numpy().flatten()
            else:
                predictions = model(X_test_t).cpu().numpy().flatten()

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Directional accuracy
        pred_dir = np.sign(predictions)
        target_dir = np.sign(y_test)
        dir_acc = (pred_dir == target_dir).mean()

        logger.info(f"\n{name}:")
        logger.info(f"  MAE:  {mae:.4f}%")
        logger.info(f"  RMSE: {rmse:.4f}%")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  Dir:  {dir_acc:.2%}")

    # 6. Save models
    save_dir = f"models/improved_ensemble_{symbol}"
    trainer.save_ensemble(save_dir)
    logger.info(f"\n✅ Models saved to {save_dir}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Train Improved COMBO System")
    parser.add_argument('--symbols', type=str, default='BTCUSDT',
                       help='Symbols to train (comma-separated)')
    parser.add_argument('--days', type=int, default=180,
                       help='Days of historical data')
    parser.add_argument('--interval', type=str, default='30m',
                       help='Timeframe (1m, 5m, 15m, 30m, 1h)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size (512 optimal for RTX 5070 Ti)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (10 epochs)')

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]

    logger.info("=" * 80)
    logger.info("🚀 IMPROVED COMBO SYSTEM - 2025 Architecture")
    logger.info("=" * 80)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Epochs: {args.epochs if not args.quick else 10}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 80)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"✅ CUDA: {torch.version.cuda}")
    else:
        logger.warning("⚠️ GPU not available - training will be slow!")

    # Train each symbol
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"SYMBOL {i}/{len(symbols)}: {symbol}")
        logger.info(f"{'='*80}")

        await train_improved_ensemble(
            symbol=symbol,
            days=args.days,
            interval=args.interval,
            epochs=args.epochs,
            batch_size=args.batch_size,
            quick=args.quick
        )

    logger.info("\n" + "=" * 80)
    logger.info("🎉 ALL TRAINING COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
