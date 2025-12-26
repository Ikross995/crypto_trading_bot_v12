#!/usr/bin/env python3
"""
🧠 GRU Model Training on REAL Binance Data
==========================================

Загружает реальные исторические данные из Binance Futures API
и обучает GRU модель для прогнозирования цены.

Особенности:
- Реальные данные из Binance Futures
- Правильная обработка rate limits (2400 weight/min)
- Пагинация для загрузки больших объёмов
- Расчёт реальных технических индикаторов
- Поддержка GPU через TensorFlow
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
    import tensorflow as tf
    from models.gru_predictor import GRUPricePredictor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure TensorFlow is installed: pip install tensorflow")
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
    Настройка TensorFlow для использования GPU.
    """
    logger.info("🎮 Configuring GPU...")

    # Проверка доступных GPU
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Разрешаем динамическое выделение памяти
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"✅ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                logger.info(f"   GPU {i}: {gpu.name}")

            # Устанавливаем GPU как устройство по умолчанию
            tf.config.set_visible_devices(gpus[0], 'GPU')

        except RuntimeError as e:
            logger.warning(f"⚠️  GPU configuration error: {e}")
            logger.info("📊 Will use CPU instead")
    else:
        logger.info("📊 No GPU found, using CPU")
        logger.info("💡 To use GPU, install: pip install tensorflow-gpu")


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
    WEIGHT_PER_REQUEST = {
        100: 1,
        500: 2,
        1000: 5,
        1500: 10
    }

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

        Args:
            symbol: Торговая пара (BTCUSDT)
            interval: Таймфрейм (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_time: Unix timestamp в миллисекундах
            end_time: Unix timestamp в миллисекундах
            limit: Количество свечей (макс 1500)

        Returns:
            List of klines: [timestamp, open, high, low, close, volume, ...]
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

        Args:
            symbol: Торговая пара
            interval: Таймфрейм
            days: Количество дней истории

        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
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
        logger.info(f"⏱️  Estimated time: ~{total_requests * 0.5:.0f}s")

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

            # Следующий интервал начинается с последней свечи
            current_start = klines[-1][0] + interval_ms

            # Небольшая задержка чтобы не превысить rate limit
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

        # Удаляем дубликаты по timestamp
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

    Индикаторы:
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2)
    - SMA (20, 50)
    - EMA (50)
    - Volume SMA (20)
    - ATR (14)
    """
    logger.info("📊 Calculating technical indicators...")

    df = df.copy()

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

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()

    # ATR (14) - Average True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # Удаляем NaN
    df = df.dropna()

    logger.info(f"✅ Indicators calculated, {len(df):,} samples remaining")

    return df


# ==========================================
# 🎓 TRAINING PIPELINE
# ==========================================

async def train_gru_on_real_data(
    symbols: List[str] = None,
    days: int = 365,
    interval: str = "1m",
    save_path: str = "models/checkpoints/gru_model_real.keras"
):
    """
    Обучить GRU модель на реальных данных Binance.

    Args:
        symbols: Список торговых пар (если None - используются default)
        days: Количество дней истории
        interval: Таймфрейм
        save_path: Путь для сохранения модели
    """
    # Default symbols
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'LINKUSDT', 'MATICUSDT'
        ]

    logger.info("=" * 70)
    logger.info("🚀 GRU Model Training on REAL Binance Data")
    logger.info("=" * 70)
    logger.info(f"📊 Symbols: {', '.join(symbols)}")
    logger.info(f"📅 History: {days} days")
    logger.info(f"⏰ Interval: {interval}")
    logger.info("=" * 70)

    # Настройка GPU
    configure_gpu()

    # Инициализация загрузчика
    downloader = BinanceDataDownloader()

    # Загрузка данных для всех пар
    all_data = []

    for i, symbol in enumerate(symbols, 1):
        logger.info("")
        logger.info(f"📥 [{i}/{len(symbols)}] Processing {symbol}...")

        try:
            # Загружаем данные
            df = await downloader.download_historical_data(
                symbol=symbol,
                interval=interval,
                days=days
            )

            # Рассчитываем индикаторы
            df = calculate_technical_indicators(df)

            # Добавляем к общему набору
            all_data.append(df)

            logger.info(f"✅ {symbol}: {len(df):,} samples ready")

        except Exception as e:
            logger.error(f"❌ Failed to process {symbol}: {e}")
            continue

    if not all_data:
        logger.error("❌ No data collected! Exiting...")
        return None

    # Объединяем все данные
    logger.info("")
    logger.info("🔗 Combining data from all symbols...")
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ Total samples: {len(combined_df):,}")

    # Выбираем признаки для обучения
    feature_columns = [
        'open', 'high', 'low', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_mid', 'bb_lower',
        'sma_20', 'sma_50', 'ema_50',
        'volume_sma', 'atr'
    ]

    # Проверяем наличие всех признаков
    missing_features = [col for col in feature_columns if col not in combined_df.columns]
    if missing_features:
        logger.error(f"❌ Missing features: {missing_features}")
        return None

    # Подготовка данных для обучения
    df_features = combined_df[feature_columns + ['close']]

    logger.info("")
    logger.info("🧠 Initializing GRU model...")
    logger.info(f"   Features: {len(feature_columns)}")
    logger.info(f"   Sequence length: 60")
    logger.info(f"   GRU units: 100")

    # Инициализация модели
    predictor = GRUPricePredictor(
        sequence_length=60,
        features=len(feature_columns),
        gru_units=100,
        dropout_rate=0.2,
        learning_rate=0.001  # Меньший LR для стабильности
    )

    # Подготовка данных
    logger.info("")
    logger.info("📊 Preparing training/test split...")
    X_train, y_train, X_test, y_test = predictor.prepare_data(
        data=df_features,
        target_column='close',
        train_split=0.8
    )

    logger.info(f"   Training set: {len(X_train):,} samples")
    logger.info(f"   Test set: {len(X_test):,} samples")

    # Обучение модели
    logger.info("")
    logger.info("🚀 Starting training...")
    logger.info("=" * 70)

    history = await predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # Оценка модели
    logger.info("")
    logger.info("📊 Evaluating model on test set...")
    metrics = await predictor.evaluate(X_test, y_test)

    logger.info("=" * 70)
    logger.info("📈 Final Results:")
    logger.info(f"   MAPE: {metrics.get('mape', 0):.2f}%")
    logger.info(f"   RMSE: {metrics.get('rmse', 0):.2f}")
    logger.info(f"   MAE: {metrics.get('mae', 0):.2f}")
    logger.info(f"   R²: {metrics.get('r2', 0):.4f}")
    logger.info("=" * 70)

    # Сохранение модели
    logger.info("")
    logger.info(f"💾 Saving model to {save_path}...")
    predictor.save(save_path)

    logger.info("")
    logger.info("✅ Training completed successfully!")
    logger.info("")
    logger.info("📝 Next steps:")
    logger.info("   1. Set GRU_ENABLE=true in .env")
    logger.info("   2. Run the bot: python cli.py live --timeframe 30m --testnet")
    logger.info("   3. Look for: [PHASE 2] GRU price predictor initialized")

    return predictor, metrics


# ==========================================
# 🎯 MAIN
# ==========================================

async def main():
    """Main entry point"""

    # Конфигурация из .env или default
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
        'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
        'LINKUSDT', 'MATICUSDT'
    ]

    try:
        await train_gru_on_real_data(
            symbols=symbols,
            days=365,  # 1 год данных
            interval="1m",
            save_path="models/checkpoints/gru_model_real.keras"
        )
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
