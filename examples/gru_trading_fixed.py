#!/usr/bin/env python3
"""
🚀 FIXED GRU Trading Model - Правильный подход к торговле
=========================================================

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. ✅ Предсказываем ПРОЦЕНТНЫЕ ИЗМЕНЕНИЯ, а не абсолютные цены
2. ✅ Добавлен Attention механизм для лучшего анализа паттернов  
3. ✅ Layer Normalization вместо Batch Normalization
4. ✅ Уменьшен dropout до разумных значений (0.2-0.3)
5. ✅ Добавлены временные фичи (час, день недели, месяц)
6. ✅ Правильная loss функция для трейдинга
7. ✅ Добавлены метрики Sharpe Ratio и Profit Factor
8. ✅ Multi-head attention для анализа разных временных масштабов
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
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    from sklearn.preprocessing import RobustScaler, StandardScaler
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
# 🧠 MULTI-HEAD ATTENTION MODULE
# ==========================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention для анализа временных паттернов"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        
        # Вычисляем Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape для multi-head
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention scores
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Final linear layer
        output = self.fc_out(context)
        output = self.dropout(output)
        
        # Residual connection и layer norm
        return self.layer_norm(output + x)


# ==========================================
# 🧠 FIXED GRU MODEL WITH ATTENTION
# ==========================================

class TradingGRUModel(nn.Module):
    """
    Улучшенная GRU модель для трейдинга:
    - Multi-head Attention для анализа паттернов
    - Layer Normalization вместо Batch Norm
    - Предсказание процентных изменений
    - Оптимальный dropout
    """
    
    def __init__(self, input_features: int, sequence_length: int, 
                 hidden_size: int = 128, num_heads: int = 4):
        super().__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(input_features, hidden_size)
        self.layer_norm_input = nn.LayerNorm(hidden_size)
        
        # GRU layers с правильными размерами
        self.gru1 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True  # Bidirectional для лучшего контекста
        )
        
        # После bidirectional GRU размерность удваивается
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_size * 2, 
            num_heads=num_heads,
            dropout=0.2
        )
        
        # Projection обратно к hidden_size
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # GRU layer 2
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        self.layer_norm3 = nn.LayerNorm(hidden_size // 2)
        
        # Dropout слои с разумными значениями
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size // 2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)  # 3 выхода: вверх, вниз, боковик
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        # Инициализация весов
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Правильная инициализация весов"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)  # Лучше для RNN
                elif 'fc' in name or 'projection' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = self.layer_norm_input(x)
        x = self.dropout1(x)
        
        # GRU Layer 1 (bidirectional)
        out, _ = self.gru1(x)
        out = self.layer_norm1(out)
        out = self.dropout1(out)
        
        # Multi-head attention
        out = self.attention(out)
        
        # Projection back
        out = self.projection(out)
        out = self.layer_norm2(out)
        out = self.dropout2(out)
        
        # GRU Layer 2
        out, _ = self.gru2(out)
        out = self.layer_norm3(out)
        out = self.dropout2(out)
        
        # Берём последний timestep
        out = out[:, -1, :]
        
        # Output layers
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.leaky_relu(out)
        
        out = self.fc3(out)
        
        return out  # Возвращаем logits для 3 классов


# ==========================================
# 📊 CUSTOM TRADING LOSS
# ==========================================

class TradingLoss(nn.Module):
    """
    Кастомная loss функция для трейдинга:
    - Учитывает направление движения
    - Штрафует за неправильное направление больше
    - Учитывает величину движения
    """
    
    def __init__(self, direction_weight: float = 2.0):
        super().__init__()
        self.direction_weight = direction_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, price_changes):
        """
        predictions: логиты для 3 классов (вверх, вниз, боковик)
        targets: класс (0=вниз, 1=боковик, 2=вверх)
        price_changes: реальное процентное изменение цены
        """
        # Classification loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # Direction penalty - штрафуем сильнее за неправильное направление
        pred_probs = F.softmax(predictions, dim=1)
        pred_direction = torch.argmax(pred_probs, dim=1).float() - 1  # -1, 0, 1
        true_direction = targets.float() - 1  # -1, 0, 1
        
        direction_error = torch.abs(pred_direction - true_direction)
        direction_loss = torch.mean(direction_error * torch.abs(price_changes))
        
        # Общая loss
        total_loss = ce_loss + self.direction_weight * direction_loss
        
        return total_loss


# ==========================================
# 📊 DATA PREPARATION WITH PROPER FEATURES
# ==========================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить временные фичи"""
    df = df.copy()
    
    # Временные фичи
    df['hour'] = pd.to_datetime(df.index).hour
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek
    df['day_of_month'] = pd.to_datetime(df.index).day
    df['month'] = pd.to_datetime(df.index).month
    
    # Циклическое кодирование времени
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Удаляем оригинальные колонки
    df.drop(['hour', 'day_of_week', 'day_of_month', 'month'], axis=1, inplace=True)
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить фичи основанные на ценах"""
    df = df.copy()
    
    # Процентные изменения
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[f'{col}_pct'] = df[col].pct_change()
        df[f'{col}_pct_ma5'] = df[f'{col}_pct'].rolling(5).mean()
        df[f'{col}_pct_std5'] = df[f'{col}_pct'].rolling(5).std()
    
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_ma5'] = df['log_return'].rolling(5).mean()
    df['log_return_std5'] = df['log_return'].rolling(5).std()
    
    # Price momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # High-Low spread
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    df['oc_spread'] = (df['close'] - df['open']) / df['open']
    
    # Volatility
    df['volatility_5'] = df['log_return'].rolling(5).std() * np.sqrt(252)
    df['volatility_20'] = df['log_return'].rolling(20).std() * np.sqrt(252)
    
    # Заполняем NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df


def prepare_trading_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    price_change_threshold: float = 0.001  # 0.1% для определения боковика
) -> Tuple:
    """
    Подготовить данные для обучения модели трейдинга.
    
    КЛЮЧЕВОЕ ОТЛИЧИЕ: Предсказываем КЛАСС движения цены, а не саму цену!
    """
    logger.info(f"📦 Preparing trading sequences...")
    logger.info(f"   Sequence length: {sequence_length}")
    logger.info(f"   Features: {len(feature_columns)}")
    
    # Добавляем временные фичи
    df = add_temporal_features(df)
    df = add_price_features(df)
    
    # Обновляем список фичей
    temporal_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    price_features = [col for col in df.columns if '_pct' in col or 'momentum' in col or 
                      'spread' in col or 'volatility' in col or 'log_return' in col]
    
    all_features = feature_columns + temporal_features + price_features
    all_features = [f for f in all_features if f in df.columns]
    
    logger.info(f"   Total features with temporal: {len(all_features)}")
    
    # Вычисляем целевую переменную - ПРОЦЕНТНОЕ ИЗМЕНЕНИЕ
    df['target_pct_change'] = df['close'].shift(-1).pct_change()
    
    # Классификация: 0 = падение, 1 = боковик, 2 = рост
    df['target_class'] = 1  # По умолчанию боковик
    df.loc[df['target_pct_change'] < -price_change_threshold, 'target_class'] = 0
    df.loc[df['target_pct_change'] > price_change_threshold, 'target_class'] = 2
    
    # Удаляем NaN
    df.dropna(inplace=True)
    
    # Временной сплит
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    logger.info(f"✅ Temporal split:")
    logger.info(f"   Train: {len(df_train):,} samples")
    logger.info(f"   Val:   {len(df_val):,} samples")
    logger.info(f"   Test:  {len(df_test):,} samples")
    
    # Распределение классов
    for name, data in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        class_dist = data['target_class'].value_counts().sort_index()
        logger.info(f"   {name} classes: Down={class_dist.get(0, 0)}, "
                   f"Sideways={class_dist.get(1, 0)}, Up={class_dist.get(2, 0)}")
    
    # Нормализация фичей (fit только на train!)
    scaler = RobustScaler()
    df_train[all_features] = scaler.fit_transform(df_train[all_features])
    df_val[all_features] = scaler.transform(df_val[all_features])
    df_test[all_features] = scaler.transform(df_test[all_features])
    
    # Создание последовательностей
    def create_sequences(data, features):
        X, y_class, y_pct = [], [], []
        for i in range(len(data) - sequence_length):
            X.append(data[features].iloc[i:i + sequence_length].values)
            y_class.append(data['target_class'].iloc[i + sequence_length])
            y_pct.append(data['target_pct_change'].iloc[i + sequence_length])
        return np.array(X), np.array(y_class), np.array(y_pct)
    
    X_train, y_train_class, y_train_pct = create_sequences(df_train, all_features)
    X_val, y_val_class, y_val_pct = create_sequences(df_val, all_features)
    X_test, y_test_class, y_test_pct = create_sequences(df_test, all_features)
    
    logger.info(f"✅ Sequences created:")
    logger.info(f"   X_train: {X_train.shape}")
    logger.info(f"   X_val: {X_val.shape}")
    logger.info(f"   X_test: {X_test.shape}")
    
    return (X_train, X_val, X_test, 
            y_train_class, y_val_class, y_test_class,
            y_train_pct, y_val_pct, y_test_pct,
            scaler, all_features)


# ==========================================
# 📊 TRADING METRICS
# ==========================================

def calculate_trading_metrics(predictions, targets, price_changes):
    """
    Рассчитать метрики для трейдинга:
    - Accuracy
    - Precision/Recall для каждого класса
    - Sharpe Ratio
    - Profit Factor
    """
    # Конвертируем в numpy если нужно
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(price_changes):
        price_changes = price_changes.cpu().numpy()
    
    # Accuracy
    accuracy = np.mean(predictions == targets) * 100
    
    # Per-class metrics
    metrics = {}
    for class_idx, class_name in enumerate(['Down', 'Sideways', 'Up']):
        mask = targets == class_idx
        if mask.sum() > 0:
            precision = np.mean(predictions[predictions == class_idx] == class_idx) * 100
            recall = np.mean(predictions[mask] == class_idx) * 100
            metrics[class_name] = {'precision': precision, 'recall': recall}
    
    # Симуляция торговли
    trading_signals = predictions - 1  # -1, 0, 1
    returns = trading_signals[:-1] * price_changes[1:]  # Сдвиг на 1 для правильного выравнивания
    
    # Sharpe Ratio (annualized)
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = np.sqrt(252 * 48) * returns.mean() / returns.std()  # 48 = торговых периодов в день для 30m
    else:
        sharpe_ratio = 0
    
    # Profit Factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf if gains > 0 else 0
    
    # Win Rate
    winning_trades = (returns > 0).sum()
    total_trades = (trading_signals[:-1] != 0).sum()
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    return {
        'accuracy': accuracy,
        'class_metrics': metrics,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'total_return': returns.sum() * 100  # в процентах
    }


# ==========================================
# 🎓 TRAINING WITH TRADING METRICS
# ==========================================

def train_trading_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    initial_lr: float = 0.001,
    patience: int = 10
) -> Dict:
    """
    Обучение модели с фокусом на метрики трейдинга
    """
    logger.info(f"🎯 Training Trading Model...")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Initial LR: {initial_lr}")
    logger.info(f"   Device: {device}")
    
    # Loss и optimizer
    criterion = TradingLoss(direction_weight=2.0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Early stopping
    best_sharpe = -np.inf
    patience_counter = 0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_sharpe': [],
        'val_profit_factor': [],
        'learning_rates': []
    }
    
    for epoch in range(epochs):
        # ===== TRAINING =====
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        
        for batch_X, batch_y_class, batch_y_pct in train_loader:
            batch_X = batch_X.to(device)
            batch_y_class = batch_y_class.to(device)
            batch_y_pct = batch_y_pct.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y_class.long(), batch_y_pct)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(batch_y_class.cpu().numpy())
        
        # ===== VALIDATION =====
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        val_price_changes = []
        
        with torch.no_grad():
            for batch_X, batch_y_class, batch_y_pct in val_loader:
                batch_X = batch_X.to(device)
                batch_y_class = batch_y_class.to(device)
                batch_y_pct = batch_y_pct.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y_class.long(), batch_y_pct)
                
                val_losses.append(loss.item())
                val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(batch_y_class.cpu().numpy())
                val_price_changes.extend(batch_y_pct.cpu().numpy())
        
        # Метрики
        train_metrics = calculate_trading_metrics(
            np.array(train_predictions),
            np.array(train_targets),
            np.zeros_like(train_predictions)  # Для train не считаем Sharpe
        )
        
        val_metrics = calculate_trading_metrics(
            np.array(val_predictions),
            np.array(val_targets),
            np.array(val_price_changes)
        )
        
        # Сохранение истории
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_sharpe'].append(val_metrics['sharpe_ratio'])
        history['val_profit_factor'].append(val_metrics['profit_factor'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Логирование
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"Acc: {train_metrics['accuracy']:.1f}%/{val_metrics['accuracy']:.1f}% | "
            f"Sharpe: {val_metrics['sharpe_ratio']:.2f} | "
            f"PF: {val_metrics['profit_factor']:.2f}"
        )
        
        # Learning rate scheduler
        scheduler.step()
        
        # Early stopping по Sharpe Ratio
        if val_metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = val_metrics['sharpe_ratio']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"   💾 New best model! Sharpe: {best_sharpe:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"   🛑 Early stopping triggered!")
                model.load_state_dict(best_model_state)
                break
    
    logger.info(f"✅ Training completed! Best Sharpe: {best_sharpe:.3f}")
    
    return history


# ==========================================
# 📊 DATASET CLASS
# ==========================================

class TradingDataset(Dataset):
    """Dataset для трейдинга с классификацией"""
    
    def __init__(self, X, y_class, y_pct):
        self.X = torch.FloatTensor(X)
        self.y_class = torch.LongTensor(y_class)
        self.y_pct = torch.FloatTensor(y_pct)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_pct[idx]


# ==========================================
# 🚀 MAIN TRAINING FUNCTION
# ==========================================

async def train_fixed_gru(
    symbols: List[str] = None,
    days: int = 365,
    interval: str = "30m",
    sequence_length: int = 60,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: str = "models/checkpoints/gru_trading_fixed.pt",
    use_cache: bool = True
):
    """
    Обучить исправленную GRU модель для трейдинга
    """
    # Импортируем функции загрузки данных
    sys.path.insert(0, str(Path(__file__).parent))
    from gru_training_pytorch import (
        BinanceDataDownloader,
        calculate_technical_indicators
    )
    
    # Default symbols
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT'
        ]
    
    logger.info("=" * 80)
    logger.info("🚀 FIXED GRU Trading Model Training")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    logger.info(f"   Days: {days}")
    logger.info(f"   Sequence: {sequence_length}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
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
                # Добавляем symbol как категорию
                df['symbol'] = symbol
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
    
    # ===== PREPARE DATA =====
    (X_train, X_val, X_test,
     y_train_class, y_val_class, y_test_class,
     y_train_pct, y_val_pct, y_test_pct,
     scaler, all_features) = prepare_trading_sequences(
        combined_df,
        feature_columns,
        sequence_length,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # ===== CREATE DATALOADERS =====
    train_dataset = TradingDataset(X_train, y_train_class, y_train_pct)
    val_dataset = TradingDataset(X_val, y_val_class, y_val_pct)
    test_dataset = TradingDataset(X_test, y_test_class, y_test_pct)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Для классификации можно использовать shuffle
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # ===== CREATE MODEL =====
    logger.info("🧠 Building FIXED Trading GRU model...")
    model = TradingGRUModel(
        input_features=len(all_features),
        sequence_length=sequence_length,
        hidden_size=128,
        num_heads=4
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model parameters: {total_params:,}")
    
    # ===== TRAIN =====
    history = train_trading_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        initial_lr=0.001,
        patience=15
    )
    
    # ===== FINAL EVALUATION =====
    logger.info("=" * 80)
    logger.info("📊 Final Evaluation on Test Set")
    logger.info("=" * 80)
    
    model.eval()
    test_predictions = []
    test_targets = []
    test_price_changes = []
    
    with torch.no_grad():
        for batch_X, batch_y_class, batch_y_pct in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(batch_y_class.numpy())
            test_price_changes.extend(batch_y_pct.numpy())
    
    # Финальные метрики
    test_metrics = calculate_trading_metrics(
        np.array(test_predictions),
        np.array(test_targets),
        np.array(test_price_changes)
    )
    
    logger.info(f"📊 Test Set Metrics:")
    logger.info(f"   Accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(f"   Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
    logger.info(f"   Profit Factor: {test_metrics['profit_factor']:.2f}")
    logger.info(f"   Win Rate: {test_metrics['win_rate']:.2f}%")
    logger.info(f"   Total Return: {test_metrics['total_return']:.2f}%")
    
    # Class-specific metrics
    for class_name, metrics in test_metrics['class_metrics'].items():
        logger.info(f"   {class_name}: Precision={metrics['precision']:.1f}%, "
                   f"Recall={metrics['recall']:.1f}%")
    
    # ===== SAVE MODEL =====
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_features': len(all_features),
            'sequence_length': sequence_length,
            'feature_columns': all_features,
            'hidden_size': 128,
            'num_heads': 4
        },
        'scaler': scaler,
        'training_history': history,
        'final_metrics': test_metrics,
        'model_type': 'classification',  # Важно для правильной загрузки
        'num_classes': 3
    }, save_path)
    
    logger.info(f"✅ Model saved: {save_path}")
    logger.info(f"   Size: {Path(save_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    logger.info("=" * 80)
    logger.info("🎉 FIXED TRAINING COMPLETED!")
    logger.info("=" * 80)
    
    # Рекомендации на основе метрик
    if test_metrics['sharpe_ratio'] < 0.5:
        logger.warning("⚠️  Low Sharpe Ratio. Recommendations:")
        logger.warning("   - Increase training data (more symbols/days)")
        logger.warning("   - Tune hyperparameters")
        logger.warning("   - Add more features (order flow, sentiment)")
    elif test_metrics['sharpe_ratio'] < 1.0:
        logger.info("📈 Decent Sharpe Ratio. Can be improved:")
        logger.info("   - Fine-tune on specific market conditions")
        logger.info("   - Add ensemble models")
    else:
        logger.info("🚀 Excellent Sharpe Ratio! Ready for trading!")
    
    logger.info("=" * 80)


# ==========================================
# 🚀 MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FIXED GRU Trading Model")
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--interval', type=str, default='30m', 
                       help='Timeframe: 1m, 5m, 15m, 30m, 1h, 4h')
    parser.add_argument('--sequence-length', type=int, default=60, 
                       help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--use-cache', action='store_true', help='Use cached data')
    parser.add_argument('--symbols', type=str, nargs='+', help='Trading symbols')
    
    args = parser.parse_args()
    
    asyncio.run(train_fixed_gru(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_cache=args.use_cache
    ))
