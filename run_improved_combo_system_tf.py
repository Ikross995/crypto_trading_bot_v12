#!/usr/bin/env python3
"""
🚀 IMPROVED COMBO SYSTEM - TensorFlow Version
==============================================

Обучение улучшенного ансамбля на TensorFlow/Keras.
Оптимизировано для RTX 5070 Ti.

Usage:
    python run_improved_combo_system_tf.py --symbols BTCUSDT --quick
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from examples.improved_ensemble_trainer_tf import ImprovedEnsembleTrainerTF
from examples.gru_training_pytorch import BinanceDataDownloader, calculate_technical_indicators
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def train_improved_ensemble_tf(
    symbol: str,
    days: int = 180,
    interval: str = '30m',
    epochs: int = 30,
    batch_size: int = 512,
    quick: bool = False
):
    """Обучить улучшенный ансамбль на TensorFlow"""

    logger.info("\n" + "🔥" * 80)
    logger.info(f"🎯 TRAINING IMPROVED ENSEMBLE (TensorFlow): {symbol}")
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
    from examples.gru_training_improved import prepare_sequences_no_leakage

    feature_columns = [
        'open', 'high', 'low', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_mid', 'bb_lower',
        'sma_20', 'sma_50', 'ema_50', 'atr',
        'volume_sma', 'volume_delta', 'obv', 'volume_ratio',
        'volume_spike', 'mfi', 'cvd', 'vwap_distance'
    ]

    X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler = \
        prepare_sequences_no_leakage(
            df.copy(),
            feature_columns,
            sequence_length=60,
            train_ratio=0.7,
            val_ratio=0.15
        )

    logger.info(f"📊 Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 4. Train improved ensemble
    trainer = ImprovedEnsembleTrainerTF()

    if quick:
        epochs = 10
        logger.info("⚡ Quick mode: 10 epochs")

    histories = trainer.train_ensemble(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    # 5. Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TESTING ON HOLD-OUT SET")
    logger.info("=" * 80)

    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    for name, model in trainer.models.items():
        predictions = model.predict(X_test, verbose=0).flatten()

        # Calculate metrics
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
    save_dir = f"models/improved_ensemble_tf_{symbol}"
    trainer.save_ensemble(save_dir)
    logger.info(f"\n✅ Models saved to {save_dir}")

    return histories


async def main():
    parser = argparse.ArgumentParser(description="Train Improved COMBO (TensorFlow)")
    parser.add_argument('--symbols', type=str, default='BTCUSDT',
                       help='Symbols (comma-separated)')
    parser.add_argument('--days', type=int, default=180,
                       help='Days of data')
    parser.add_argument('--interval', type=str, default='30m',
                       help='Timeframe')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (10 epochs)')

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]

    logger.info("=" * 80)
    logger.info("🚀 IMPROVED COMBO SYSTEM - TensorFlow")
    logger.info("=" * 80)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Epochs: {args.epochs if not args.quick else 10}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 80)

    # Check TensorFlow GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"✅ TensorFlow GPU: {len(gpus)} device(s)")
        for gpu in gpus:
            logger.info(f"   - {gpu.name}")
    else:
        logger.warning("⚠️ No GPU - using CPU")

    # Train each symbol
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"SYMBOL {i}/{len(symbols)}: {symbol}")
        logger.info(f"{'='*80}")

        await train_improved_ensemble_tf(
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
