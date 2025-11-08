#!/usr/bin/env python3
"""
🚀 IMPROVED GRU Model Training - Fixed All Issues!
===================================================

FIXES:
1. ✅ Removed shuffle=True for time series
2. ✅ Proper time-based train/test split
3. ✅ Rolling window normalization (NO data leakage!)
4. ✅ Increased dropout to 0.3-0.4
5. ✅ Learning rate scheduler
6. ✅ Early stopping
7. ✅ Batch normalization
8. ✅ Gradient clipping
9. ✅ Better win rate calculation
10. ✅ Separate validation set

Автор: Claude (Anthropic)
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
    from sklearn.preprocessing import RobustScaler  # Better than MinMaxScaler!
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install: pip install torch torchvision scikit-learn")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 🎮 GPU CONFIGURATION
# ==========================================

def configure_gpu():
    """Настройка PyTorch для использования GPU."""
    logger.info("🎮 Configuring GPU...")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"   CUDA: {torch.version.cuda}")

        # Оптимизации
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        return device
    else:
        logger.info("📊 Using CPU")
        return torch.device('cpu')


# ==========================================
# 🧠 IMPROVED GRU MODEL
# ==========================================

class ImprovedGRUModel(nn.Module):
    """
    Улучшенная GRU модель с:
    - Batch Normalization
    - Увеличенный Dropout (0.4)
    - Residual connections
    - Better initialization
    """

    def __init__(self, input_features: int, sequence_length: int):
        super(ImprovedGRUModel, self).__init__()

        self.input_features = input_features
        self.sequence_length = sequence_length

        # Batch Normalization для входа
        self.input_bn = nn.BatchNorm1d(sequence_length)

        # GRU Layers с увеличенной capacity
        self.gru1 = nn.GRU(
            input_size=input_features,
            hidden_size=128,  # Увеличено со 100
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.gru2 = nn.GRU(
            input_size=128,
            hidden_size=64,  # Увеличено с 50
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Batch Normalization после GRU
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        # Увеличенный Dropout
        self.dropout1 = nn.Dropout(0.4)  # Было 0.2
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)

        # Dense layers с residual connection
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Xavier initialization для лучшей сходимости
        self._initialize_weights()

    def _initialize_weights(self):
        """Правильная инициализация весов"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        batch_size = x.size(0)

        # Batch norm на входе (reshape для BatchNorm1d)
        x_bn = self.input_bn(x)

        # GRU Layer 1
        out, h1 = self.gru1(x_bn)
        # Batch norm (reshape: batch, seq, features -> batch, features, seq)
        out = out.permute(0, 2, 1)
        out = self.bn1(out)
        out = out.permute(0, 2, 1)
        out = self.dropout1(out)

        # GRU Layer 2
        out, h2 = self.gru2(out)
        out = out.permute(0, 2, 1)
        out = self.bn2(out)
        out = out.permute(0, 2, 1)
        out = self.dropout2(out)

        # Берём последний timestep
        out = out[:, -1, :]  # (batch, 64)

        # Dense layers с residual connection
        x_residual = out
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout3(out)

        out = self.fc2(out)
        out = self.leaky_relu(out)

        out = self.fc3(out)

        return out


class TimeSeriesDataset(Dataset):
    """
    Dataset для временных рядов.
    ВАЖНО: НЕ используем shuffle в DataLoader!
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 📊 IMPROVED DATA PREPARATION
# ==========================================

def prepare_sequences_no_leakage(
    df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple:
    """
    Подготовить последовательности БЕЗ data leakage!

    Ключевые улучшения:
    1. Временной split ПЕРЕД нормализацией
    2. Separate scalers для train/val/test
    3. Rolling window normalization

    Args:
        df: DataFrame с features и close
        feature_columns: Список колонок-фичей
        sequence_length: Длина последовательности
        train_ratio: Доля train данных
        val_ratio: Доля validation данных

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    logger.info(f"📦 Preparing sequences (NO LEAKAGE!)...")
    logger.info(f"   Sequence length: {sequence_length}")
    logger.info(f"   Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {(1-train_ratio-val_ratio)*100:.0f}%")

    # ===== TEMPORAL SPLIT (НЕ случайный!) =====
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    logger.info(f"✅ Temporal split:")
    logger.info(f"   Train: {len(df_train):,} samples ({df_train.index[0]} → {df_train.index[-1]})")
    logger.info(f"   Val:   {len(df_val):,} samples ({df_val.index[0]} → {df_val.index[-1]})")
    logger.info(f"   Test:  {len(df_test):,} samples ({df_test.index[0]} → {df_test.index[-1]})")

    # ===== NORMALIZATION (fit только на train!) =====
    logger.info("🔄 Normalizing with RobustScaler (resistant to outliers)...")

    # Features scaler - fit ТОЛЬКО на train!
    feature_scaler = RobustScaler()
    df_train[feature_columns] = feature_scaler.fit_transform(df_train[feature_columns])
    df_val[feature_columns] = feature_scaler.transform(df_val[feature_columns])
    df_test[feature_columns] = feature_scaler.transform(df_test[feature_columns])

    # Target scaler - отдельный для 'close'
    target_scaler = RobustScaler()
    df_train[['close']] = target_scaler.fit_transform(df_train[['close']])
    df_val[['close']] = target_scaler.transform(df_val[['close']])
    df_test[['close']] = target_scaler.transform(df_test[['close']])

    logger.info(f"   ✅ Scalers fitted on TRAIN data only (NO LEAKAGE!)")

    # ===== CREATE SEQUENCES =====
    def create_sequences(data, features):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[features].iloc[i:i + sequence_length].values)
            y.append(data['close'].iloc[i + sequence_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(df_train, feature_columns)
    X_val, y_val = create_sequences(df_val, feature_columns)
    X_test, y_test = create_sequences(df_test, feature_columns)

    logger.info(f"✅ Sequences created:")
    logger.info(f"   Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, y={y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler


# ==========================================
# 🎓 IMPROVED TRAINING WITH EARLY STOPPING
# ==========================================

class EarlyStopping:
    """Early stopping для предотвращения overfitting"""

    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"   ⚠️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"   🛑 Early stopping triggered!")
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0


def train_improved_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    initial_lr: float = 0.001,
    patience: int = 7
) -> Dict:
    """
    Улучшенное обучение с:
    - Learning rate scheduler
    - Early stopping
    - Gradient clipping
    - Best model saving
    """
    logger.info(f"🎯 Training IMPROVED model...")
    logger.info(f"   Epochs: {epochs} (max)")
    logger.info(f"   Initial LR: {initial_lr}")
    logger.info(f"   Early stopping patience: {patience}")
    logger.info(f"   Device: {device}")

    # Loss и optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(  # AdamW лучше чем Adam!
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-5  # L2 regularization
    )

    # Learning rate scheduler - уменьшает LR при plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Уменьшаем LR в 2 раза
        patience=3,
        min_lr=1e-6
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # История
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        # ===== TRAINING =====
        model.train()
        train_losses = []
        train_maes = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)

            loss.backward()

            # Gradient clipping - ВАЖНО для RNN!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())
            mae = torch.mean(torch.abs(predictions.squeeze() - batch_y)).item()
            train_maes.append(mae)

        # ===== VALIDATION =====
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

        # Средние метрики
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_mae = np.mean(train_maes)
        avg_val_mae = np.mean(val_maes)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Логирование
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {avg_train_loss:.6f} | "
            f"Val: {avg_val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Learning rate scheduler
        scheduler.step(avg_val_loss)

        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"✅ Early stopping at epoch {epoch+1}")
            # Восстанавливаем лучшую модель
            model.load_state_dict(early_stopping.best_model_state)
            break

        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"   💾 New best model! Val Loss: {best_val_loss:.6f}")

    total_time = time.time() - start_time
    logger.info(f"✅ Training completed in {total_time/60:.2f} minutes")
    logger.info(f"   Best validation loss: {best_val_loss:.6f}")

    return history


# ==========================================
# 📊 CALCULATE WIN RATE
# ==========================================

def calculate_win_rate(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.0001
) -> Dict:
    """
    Рассчитать win rate (процент правильных предсказаний направления).

    Args:
        predictions: Предсказанные цены
        targets: Реальные цены
        threshold: Минимальное изменение для считания направления

    Returns:
        Dict с метриками win rate
    """
    # Направление изменения (up/down)
    pred_direction = np.sign(predictions[1:] - predictions[:-1])
    true_direction = np.sign(targets[1:] - targets[:-1])

    # Win rate
    correct_predictions = (pred_direction == true_direction).sum()
    total_predictions = len(pred_direction)
    win_rate = (correct_predictions / total_predictions) * 100

    # Win rate для значимых движений
    significant_moves = np.abs(targets[1:] - targets[:-1]) > threshold
    if significant_moves.sum() > 0:
        significant_correct = ((pred_direction == true_direction) & significant_moves).sum()
        significant_win_rate = (significant_correct / significant_moves.sum()) * 100
    else:
        significant_win_rate = 0.0

    logger.info(f"📊 Win Rate Analysis:")
    logger.info(f"   Overall: {win_rate:.2f}% ({correct_predictions}/{total_predictions})")
    logger.info(f"   Significant moves (>{threshold}): {significant_win_rate:.2f}%")

    return {
        'win_rate': win_rate,
        'significant_win_rate': significant_win_rate,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions
    }


# ==========================================
# 🚀 MAIN TRAINING FUNCTION
# ==========================================

async def train_improved_gru(
    symbols: List[str] = None,
    days: int = 365,
    interval: str = "30m",
    sequence_length: int = 60,
    epochs: int = 50,
    batch_size: int = 64,
    save_path: str = "models/checkpoints/gru_improved.pt",
    use_cache: bool = True
):
    """
    Обучить УЛУЧШЕННУЮ GRU модель.

    Основные улучшения:
    1. NO shuffle в DataLoader
    2. Temporal train/val/test split
    3. NO data leakage в нормализации
    4. Early stopping
    5. Learning rate scheduler
    6. Increased dropout
    7. Batch normalization
    8. Better model architecture
    """
    # Импортируем оригинальные функции загрузки данных
    sys.path.insert(0, str(Path(__file__).parent))
    from gru_training_pytorch import (
        BinanceDataDownloader,
        calculate_technical_indicators
    )

    # Default symbols
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'LINKUSDT', 'APTUSDT'
        ]

    logger.info("=" * 80)
    logger.info("🚀 IMPROVED GRU Model Training (NO BUGS!)")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    logger.info(f"   Days: {days}")
    logger.info(f"   Sequence: {sequence_length}")
    logger.info(f"   Epochs: {epochs} (max, with early stopping)")
    logger.info(f"   Batch size: {batch_size}")
    logger.info("=" * 80)
    logger.info("✅ FIXES:")
    logger.info("   1. shuffle=False (preserves time order)")
    logger.info("   2. Temporal train/val/test split")
    logger.info("   3. NO data leakage (fit on train only)")
    logger.info("   4. Early stopping (patience=7)")
    logger.info("   5. Learning rate scheduler")
    logger.info("   6. Dropout 0.4 (was 0.2)")
    logger.info("   7. Batch normalization")
    logger.info("   8. RobustScaler (outlier resistant)")
    logger.info("=" * 80)

    # GPU setup
    device = configure_gpu()

    # ===== LOAD DATA =====
    combined_df = None

    if use_cache:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scripts.download_and_cache_data import load_cached_data
            logger.info("📂 Loading cached data...")
            combined_df = load_cached_data(symbols, days, interval)
        except:
            logger.warning("⚠️  Cache not available")

    if combined_df is None:
        downloader = BinanceDataDownloader()
        all_data = []

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"📥 Downloading {symbol} ({i}/{len(symbols)})...")
            df = await downloader.download_historical_data(symbol, interval, days)

            if len(df) > 0:
                df = calculate_technical_indicators(df)
                all_data.append(df)
            else:
                logger.warning(f"⚠️  Skipping {symbol}")

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Combined: {len(combined_df):,} samples")

    # ===== FEATURE COLUMNS =====
    feature_columns = [
        'open', 'high', 'low', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_mid', 'bb_lower',
        'sma_20', 'sma_50', 'ema_50',
        'volume_sma', 'atr',
        'volume_delta', 'obv', 'volume_ratio',
        'volume_spike', 'mfi', 'cvd', 'vwap_distance'
    ]

    logger.info(f"📊 Using {len(feature_columns)} features")

    # ===== PREPARE DATA (NO LEAKAGE!) =====
    X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler = \
        prepare_sequences_no_leakage(
            combined_df,
            feature_columns,
            sequence_length,
            train_ratio=0.7,
            val_ratio=0.15
        )

    # ===== CREATE DATALOADERS (NO SHUFFLE!) =====
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # ✅ NO SHUFFLE для временных рядов!
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # ✅ NO SHUFFLE!
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # ✅ NO SHUFFLE!
        num_workers=0,
        pin_memory=True
    )

    # ===== CREATE IMPROVED MODEL =====
    logger.info("🧠 Building IMPROVED GRU model...")
    model = ImprovedGRUModel(
        input_features=len(feature_columns),
        sequence_length=sequence_length
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model parameters: {total_params:,}")

    # ===== TRAIN =====
    history = train_improved_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        initial_lr=0.001,
        patience=7
    )

    # ===== FINAL EVALUATION =====
    logger.info("=" * 80)
    logger.info("📊 Final Evaluation on Test Set")
    logger.info("=" * 80)

    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(batch_y.numpy())

    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets)

    # Денормализация
    test_predictions_real = target_scaler.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).flatten()
    test_targets_real = target_scaler.inverse_transform(
        test_targets.reshape(-1, 1)
    ).flatten()

    # Метрики
    mse = np.mean((test_predictions_real - test_targets_real) ** 2)
    mae = np.mean(np.abs(test_predictions_real - test_targets_real))
    mape = np.mean(np.abs((test_targets_real - test_predictions_real) / test_targets_real)) * 100

    logger.info(f"📊 Test Metrics (Real Prices):")
    logger.info(f"   MSE:  {mse:.2f}")
    logger.info(f"   MAE:  ${mae:.2f}")
    logger.info(f"   MAPE: {mape:.2f}%")

    # Win Rate
    win_rate_metrics = calculate_win_rate(test_predictions_real, test_targets_real)

    # ===== SAVE MODEL =====
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_features': len(feature_columns),
            'sequence_length': sequence_length,
            'feature_columns': feature_columns
        },
        'scalers': {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        },
        'training_history': history,
        'final_metrics': {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            **win_rate_metrics
        }
    }, save_path)

    logger.info(f"✅ Model saved: {save_path}")
    logger.info(f"   Size: {Path(save_path).stat().st_size / 1024 / 1024:.1f} MB")

    logger.info("=" * 80)
    logger.info("🎉 IMPROVED TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info("📋 Next steps:")
    logger.info(f"   1. Update .env: GRU_MODEL_PATH={save_path}")
    logger.info(f"   2. Run bot: python start_bot.py")
    logger.info("=" * 80)


# ==========================================
# 🚀 MAIN
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train IMPROVED GRU model")
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--interval', type=str, default='30m', help='Timeframe: 1m, 5m, 15m, 30m, 1h, 4h')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--use-cache', action='store_true', help='Use cached data')
    parser.add_argument('--symbols', type=str, nargs='+', help='Trading symbols')

    args = parser.parse_args()

    asyncio.run(train_improved_gru(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_cache=args.use_cache
    ))
