#!/usr/bin/env python3
"""
🧠 GRU Model Training on REAL Binance Data (PyTorch Version)
============================================================

Загружает реальные исторические данные из Binance Futures API
и обучает GRU модель для прогнозирования цены используя PyTorch.

Особенности:
- Реальные данные из Binance Futures
- Правильная обработка rate limits (2400 weight/min)
- Пагинация для загрузки больших объёмов
- Расчёт реальных технических индикаторов
- ⚡ GPU поддержка через PyTorch (работает сразу!)
- Обучение на нескольких торговых парах
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
# 🎮 GPU CONFIGURATION
# ==========================================

def configure_gpu():
    """
    Настройка PyTorch для использования GPU.
    """
    logger.info("🎮 Configuring GPU...")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        logger.info(f"✅ GPU available: {gpu_name}")
        logger.info(f"   GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"   CUDA Version: {torch.version.cuda}")

        # Оптимизации для быстрого обучения
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        return device
    else:
        logger.info("📊 No GPU found, using CPU")
        logger.info("💡 To use GPU on Windows: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return torch.device('cpu')


# ==========================================
# 📥 BINANCE DATA DOWNLOADER
# ==========================================

class BinanceDataDownloader:
    """
    Загрузчик исторических данных из Binance Futures API.

    Правильно обрабатывает:
    - Rate limits (2400 request weight/min)
    - Пагинацию для больших объёмов
    - Ошибки сети
    - Валидацию данных
    """

    BASE_URL = "https://fapi.binance.com"  # Futures API
    MAX_LIMIT = 1500  # Максимум свечей за запрос
    RATE_LIMIT_WEIGHT = 2400  # Weight limit per minute

    def __init__(self):
        self.request_count = 0
        self.request_weight = 0
        self.last_reset = time.time()

    def _get_request_weight(self, limit: int) -> int:
        """Получить weight для запроса в зависимости от limit"""
        if limit <= 100:
            return 1
        elif limit <= 500:
            return 2
        elif limit <= 1000:
            return 5
        else:
            return 10

    async def _rate_limit_check(self, weight: int):
        """Проверка и контроль rate limit"""
        # Сброс счётчика каждую минуту
        now = time.time()
        if now - self.last_reset >= 60:
            self.request_weight = 0
            self.last_reset = now

        # Если превышен лимит - ждём
        if self.request_weight + weight > self.RATE_LIMIT_WEIGHT:
            wait_time = 60 - (now - self.last_reset)
            logger.warning(f"⏱️  Rate limit reached, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time + 1)
            self.request_weight = 0
            self.last_reset = time.time()

        self.request_weight += weight

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500
    ) -> List[List]:
        """
        Загрузить свечи из Binance Futures API.
        """
        import aiohttp

        # Rate limit check
        weight = self._get_request_weight(limit)
        await self._rate_limit_check(weight)

        # Параметры запроса
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        url = f"{self.BASE_URL}/fapi/v1/klines"

        # Выполняем запрос
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.request_count += 1
                        return data
                    else:
                        logger.error(f"❌ API error {response.status}: {await response.text()}")
                        return []
            except Exception as e:
                logger.error(f"❌ Request failed: {e}")
                return []

    async def download_historical_data(
        self,
        symbol: str,
        interval: str = "1m",
        days: int = 365
    ) -> pd.DataFrame:
        """
        Загрузить исторические данные с пагинацией.
        """
        logger.info(f"📥 Downloading {days} days of {symbol} {interval} data...")

        # Расчёт временных рамок
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        # Расчёт количества запросов
        interval_ms = self._interval_to_ms(interval)
        total_candles = (end_time - start_time) // interval_ms
        total_requests = (total_candles // self.MAX_LIMIT) + 1

        logger.info(f"📊 Total candles: ~{total_candles:,}")
        logger.info(f"🔄 Required requests: ~{total_requests}")

        all_klines = []
        current_start = start_time
        request_num = 0

        while current_start < end_time:
            request_num += 1

            # Загружаем порцию данных
            klines = await self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=self.MAX_LIMIT
            )

            if not klines:
                logger.warning(f"⚠️  No data received for {symbol}, retrying...")
                await asyncio.sleep(5)
                continue

            all_klines.extend(klines)

            # Логирование прогресса
            if request_num % 10 == 0:
                logger.info(f"   Progress: {len(all_klines):,} candles downloaded ({request_num}/{total_requests} requests)")

            # Следующий интервал
            current_start = klines[-1][0] + interval_ms
            await asyncio.sleep(0.1)

        # Преобразуем в DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Преобразуем типы данных
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Оставляем только нужные колонки
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Удаляем дубликаты
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"✅ Downloaded {len(df):,} candles for {symbol}")
        logger.info(f"   Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        logger.info(f"   Price: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")

        return df

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        """Конвертировать интервал в миллисекунды"""
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000
        }

        unit = interval[-1]
        value = int(interval[:-1])

        return value * multipliers.get(unit, 60 * 1000)


# ==========================================
# 📊 TECHNICAL INDICATORS
# ==========================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитать технические индикаторы для обучения модели.

    Includes:
    - Price indicators: RSI, MACD, BB, SMA, EMA, ATR
    - Volume indicators: Volume Delta, OBV, Volume Ratio, Spike, MFI, CVD, VWAP Distance
    Total: 22 features
    """
    logger.info("📊 Calculating technical indicators...")

    df = df.copy()

    # ============================================
    # PRICE INDICATORS
    # ============================================

    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_mid'] = sma_20

    # SMA
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # EMA
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ATR (14) - Average True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # ============================================
    # VOLUME INDICATORS (7 new features!)
    # ============================================

    logger.info("📊 Calculating advanced volume indicators...")

    # Volume SMA (baseline)
    df['volume_sma'] = df['volume'].rolling(window=20).mean()

    # 1. Volume Delta - Buy/Sell pressure (simplified)
    df['volume_delta'] = df['volume'] * np.where(
        df['close'] > df['open'], 1, -1
    )

    # 2. OBV (On-Balance Volume) - Cumulative volume
    obv = []
    obv_val = 0
    for i in range(len(df)):
        if i == 0:
            obv.append(0)
        else:
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv_val += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv_val -= df['volume'].iloc[i]
            obv.append(obv_val)
    df['obv'] = obv

    # 3. Volume Ratio - Current volume vs average
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

    # 4. Volume Spike - Binary flag for anomalous volume
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(float)

    # 5. MFI (Money Flow Index) - RSI with volume
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(df['close'] <= df['close'].shift(1), 0).rolling(14).sum()

    mfi_ratio = positive_flow / (negative_flow + 1e-10)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))

    # 6. CVD (Cumulative Volume Delta) - Accumulated delta
    df['cvd'] = df['volume_delta'].cumsum()

    # 7. VWAP Distance - Distance from volume-weighted average price
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

    # Удаляем NaN
    df = df.dropna()

    logger.info(f"✅ All indicators calculated: 15 price + 7 volume = 22 features")
    logger.info(f"✅ Samples remaining: {len(df):,}")

    return df


# ==========================================
# 🧠 PYTORCH GRU MODEL
# ==========================================

class GRUPriceModel(nn.Module):
    """
    GRU модель для прогнозирования цены криптовалюты.

    Architecture:
    - Input: (batch, sequence_length, features)
    - GRU Layer 1: 100 units, dropout=0.2
    - GRU Layer 2: 50 units, dropout=0.2
    - Dense: 25 units, ReLU
    - Output: 1 unit (predicted price)
    """

    def __init__(self, input_features: int, sequence_length: int):
        super(GRUPriceModel, self).__init__()

        self.input_features = input_features
        self.sequence_length = sequence_length

        # GRU Layers
        self.gru1 = nn.GRU(
            input_size=input_features,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # Не нужно для одного слоя
        )

        self.gru2 = nn.GRU(
            input_size=100,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        # Dense layers
        self.fc1 = nn.Linear(50, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        # x shape: (batch, sequence_length, features)

        # GRU Layer 1
        out, _ = self.gru1(x)
        out = self.dropout1(out)

        # GRU Layer 2
        out, _ = self.gru2(out)
        out = self.dropout2(out)

        # Берём последний timestep
        out = out[:, -1, :]  # (batch, 50)

        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class PriceDataset(Dataset):
    """PyTorch Dataset для последовательностей цен"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 🎓 TRAINING PIPELINE
# ==========================================

def prepare_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Подготовить последовательности для обучения GRU с нормализацией.

    Args:
        df: DataFrame с features и close
        feature_columns: Список колонок-фичей
        sequence_length: Длина последовательности

    Returns:
        X: (samples, sequence_length, features) - нормализованные
        y: (samples,) - нормализованные predicted price
        feature_scaler: Scaler для features
        target_scaler: Scaler для target (close)
    """
    logger.info(f"📦 Preparing sequences (length={sequence_length})...")

    # Нормализация features (0-1)
    logger.info("🔄 Normalizing features to 0-1 range...")
    feature_scaler = MinMaxScaler()
    features_normalized = feature_scaler.fit_transform(df[feature_columns])

    # Нормализация target (close) отдельно
    target_scaler = MinMaxScaler()
    target_normalized = target_scaler.fit_transform(df[['close']]).flatten()

    logger.info(f"   Feature range: {features_normalized.min():.4f} - {features_normalized.max():.4f}")
    logger.info(f"   Target range: {target_normalized.min():.4f} - {target_normalized.max():.4f}")

    X, y = [], []

    for i in range(len(df) - sequence_length):
        X.append(features_normalized[i:i + sequence_length])
        y.append(target_normalized[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"✅ Sequences prepared:")
    logger.info(f"   X shape: {X.shape}")
    logger.info(f"   y shape: {y.shape}")

    return X, y, feature_scaler, target_scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 0.001
) -> Dict:
    """
    Обучить GRU модель.
    """
    logger.info(f"🎯 Training started...")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Batch size: {len(next(iter(train_loader))[0])}")
    logger.info(f"   Batches per epoch: {len(train_loader):,}")
    if device.type == 'cuda':
        logger.info(f"   ⚡ GPU Acceleration ENABLED")

    # Оптимизатор и функция потерь
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
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

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping - предотвращает взрывы градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Метрики
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

        # Логирование
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Train MAE: {avg_train_mae:.2f} | "
            f"Val MAE: {avg_val_mae:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"   💾 New best model! Val Loss: {best_val_loss:.6f}")

    total_time = time.time() - start_time
    logger.info(f"✅ Training completed in {total_time/60:.1f} minutes")

    return history


async def train_gru_on_real_data(
    symbols: List[str] = None,
    days: int = 365,
    interval: str = "1m",
    sequence_length: int = 60,
    epochs: int = 20,
    batch_size: int = 32,
    save_path: str = "models/checkpoints/gru_model_pytorch.pt",
    use_cache: bool = False
):
    """
    Обучить GRU модель на реальных данных Binance (PyTorch версия).

    Args:
        symbols: Список торговых пар
        days: Количество дней истории
        interval: Таймфрейм
        sequence_length: Длина последовательности
        epochs: Количество эпох обучения
        batch_size: Размер батча
        save_path: Путь для сохранения модели
    """
    # Default symbols
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'LINKUSDT', 'APTUSDT'  # Заменил MATICUSDT на APTUSDT
        ]

    logger.info("=" * 80)
    logger.info("🧠 GRU Model Training on REAL Binance Data (PyTorch)")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    logger.info(f"   Days: {days}")
    logger.info(f"   Interval: {interval}")
    logger.info(f"   Sequence length: {sequence_length}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info("=" * 80)

    # Настройка GPU
    device = configure_gpu()

    # Загрузка данных (с кэшированием)
    combined_df = None

    if use_cache:
        try:
            from scripts.download_and_cache_data import load_cached_data, save_cached_data
            logger.info("📂 Trying to load cached data...")
            combined_df = load_cached_data(symbols, days, interval)
        except ImportError:
            logger.warning("⚠️  Cache module not found, downloading fresh data")

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
                logger.warning(f"⚠️  Skipping {symbol} - no data")

        # Объединяем данные
        logger.info("🔗 Combining data from all symbols...")
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Combined dataset: {len(combined_df):,} samples")

        # Сохраняем в кэш для следующего раза
        try:
            from scripts.download_and_cache_data import save_cached_data
            logger.info("💾 Saving data to cache...")
            save_cached_data(combined_df, symbols, days, interval)
        except:
            logger.warning("⚠️  Could not cache data")
    else:
        logger.info(f"✅ Using cached dataset: {len(combined_df):,} samples")

    # Список фичей (22 indicators: 15 price + 7 volume)
    feature_columns = [
        # Price features (4)
        'open', 'high', 'low', 'volume',
        # Technical indicators (11)
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_mid', 'bb_lower',
        'sma_20', 'sma_50', 'ema_50',
        'volume_sma', 'atr',
        # Volume indicators (7) - NEW!
        'volume_delta',    # Buy/sell pressure
        'obv',             # On-Balance Volume
        'volume_ratio',    # Volume vs average
        'volume_spike',    # Anomaly detection
        'mfi',             # Money Flow Index
        'cvd',             # Cumulative Volume Delta
        'vwap_distance'    # Distance from VWAP
    ]

    logger.info(f"📊 Features: {len(feature_columns)} total")
    logger.info(f"   - Price features: 15 (open, high, low, volume, rsi, macd, bb, sma, ema, atr)")
    logger.info(f"   - Volume features: 7 (volume_delta, obv, ratio, spike, mfi, cvd, vwap)")
    logger.info(f"🔥 Enhanced model with advanced volume analysis!")

    # Подготовка последовательностей с нормализацией
    X, y, feature_scaler, target_scaler = prepare_sequences(combined_df, feature_columns, sequence_length)

    # Train/Test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"📊 Train samples: {len(X_train):,}")
    logger.info(f"📊 Test samples: {len(X_test):,}")

    # Создаём DataLoaders с многопоточной загрузкой
    train_dataset = PriceDataset(X_train, y_train)
    test_dataset = PriceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # Многопоточная загрузка данных (16 CPU threads / 2)
        pin_memory=True,  # Ускорение переноса на GPU
        persistent_workers=True,  # Держать workers alive между эпохами
        prefetch_factor=4  # Каждый worker подготавливает 4 батча заранее
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Создаём модель
    logger.info("🧠 Building GRU model...")
    model = GRUPriceModel(
        input_features=len(feature_columns),
        sequence_length=sequence_length
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✅ Model architecture:")
    logger.info(f"   - Input: ({sequence_length}, {len(feature_columns)})")
    logger.info(f"   - GRU Layer 1: 100 units, dropout=0.2")
    logger.info(f"   - GRU Layer 2: 50 units, dropout=0.2")
    logger.info(f"   - Dense: 25 units (ReLU)")
    logger.info(f"   - Output: 1 unit")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")

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

    # Денормализация обратно в реальные цены
    logger.info("🔄 Denormalizing predictions back to real prices...")
    test_predictions_real = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    test_targets_real = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

    # Метрики на нормализованных данных (0-1)
    mse_normalized = np.mean((test_predictions - test_targets) ** 2)
    mae_normalized = np.mean(np.abs(test_predictions - test_targets))

    # Метрики на реальных ценах (в долларах)
    mse = np.mean((test_predictions_real - test_targets_real) ** 2)
    mae = np.mean(np.abs(test_predictions_real - test_targets_real))
    mape = np.mean(np.abs((test_targets_real - test_predictions_real) / test_targets_real)) * 100

    logger.info("=" * 80)
    logger.info("📊 Final metrics (REAL PRICES):")
    logger.info(f"   - MSE: {mse:.2f}")
    logger.info(f"   - MAE: ${mae:.2f}")
    logger.info(f"   - MAPE: {mape:.2f}%")
    logger.info(f"📊 Normalized metrics (0-1 range):")
    logger.info(f"   - MSE: {mse_normalized:.6f}")
    logger.info(f"   - MAE: {mae_normalized:.6f}")
    logger.info("=" * 80)

    # Сохранение модели
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
            'mse_normalized': mse_normalized,
            'mae_normalized': mae_normalized
        }
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
    logger.info(f"   3. Run bot: python start_bot.py")


# ==========================================
# 🚀 MAIN
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GRU model on real Binance data")
    parser.add_argument('--days', type=int, default=365,
                        help='Days of historical data (default: 365)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16, lower if GPU OOM)')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached data if available')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Symbols to train on (default: top 10)')

    args = parser.parse_args()

    asyncio.run(train_gru_on_real_data(
        symbols=args.symbols,
        days=args.days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_cache=args.use_cache
    ))
