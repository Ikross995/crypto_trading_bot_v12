#!/usr/bin/env python3
"""
💾 Data Caching Utilities for GRU Training
==========================================

Сохранение и загрузка скачанных данных для быстрого переиспользования.

Использование:
    # Скачать и сохранить данные
    python scripts/download_and_cache_data.py --days 365

    # Использовать кэшированные данные при обучении
    python examples/gru_training_pytorch.py --use-cache

Преимущества:
    - Не нужно скачивать данные каждый раз (экономия 5-10 минут)
    - Можно экспериментировать с разными параметрами обучения
    - Данные сохраняются в data/cached_training_data/
"""

import asyncio
import pickle
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Optional
import sys

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.gru_training_pytorch import (
    BinanceDataDownloader,
    calculate_technical_indicators
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cached_training_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_filename(symbols: List[str], days: int, interval: str) -> str:
    """
    Генерировать имя файла кэша.

    Args:
        symbols: Список символов
        days: Количество дней
        interval: Таймфрейм

    Returns:
        Имя файла
    """
    symbols_str = "_".join(sorted(symbols))
    if len(symbols_str) > 50:
        # Если слишком длинное - используем hash
        import hashlib
        symbols_str = hashlib.md5(symbols_str.encode()).hexdigest()[:16]

    return f"training_data_{symbols_str}_{days}d_{interval}.pkl"


def save_cached_data(
    combined_df: pd.DataFrame,
    symbols: List[str],
    days: int,
    interval: str,
    metadata: Optional[dict] = None
):
    """
    Сохранить данные в кэш.

    Args:
        combined_df: Объединённый DataFrame
        symbols: Список символов
        days: Количество дней
        interval: Таймфрейм
        metadata: Дополнительная информация
    """
    filename = get_cache_filename(symbols, days, interval)
    cache_path = CACHE_DIR / filename

    cache_data = {
        'data': combined_df,
        'symbols': symbols,
        'days': days,
        'interval': interval,
        'cached_at': datetime.now().isoformat(),
        'samples_count': len(combined_df),
        'metadata': metadata or {}
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    file_size_mb = cache_path.stat().st_size / 1024 / 1024
    logger.info(f"💾 Data cached to: {cache_path}")
    logger.info(f"   File size: {file_size_mb:.1f} MB")
    logger.info(f"   Samples: {len(combined_df):,}")


def load_cached_data(
    symbols: List[str],
    days: int,
    interval: str
) -> Optional[pd.DataFrame]:
    """
    Загрузить данные из кэша.

    Args:
        symbols: Список символов
        days: Количество дней
        interval: Таймфрейм

    Returns:
        DataFrame или None если кэш не найден
    """
    filename = get_cache_filename(symbols, days, interval)
    cache_path = CACHE_DIR / filename

    if not cache_path.exists():
        logger.info(f"📂 Cache not found: {filename}")
        return None

    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        cached_at = datetime.fromisoformat(cache_data['cached_at'])
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600

        logger.info(f"✅ Loaded cached data: {filename}")
        logger.info(f"   Cached at: {cached_at.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Age: {age_hours:.1f} hours")
        logger.info(f"   Samples: {cache_data['samples_count']:,}")

        return cache_data['data']

    except Exception as e:
        logger.error(f"❌ Failed to load cache: {e}")
        return None


def list_cached_data():
    """Показать список всех кэшированных данных"""
    logger.info("=" * 80)
    logger.info("📚 Cached Training Data")
    logger.info("=" * 80)

    cache_files = sorted(CACHE_DIR.glob("training_data_*.pkl"))

    if not cache_files:
        logger.info("   No cached data found.")
        logger.info("   Run: python scripts/download_and_cache_data.py --days 365")
        return

    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            file_size_mb = cache_file.stat().st_size / 1024 / 1024
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600

            logger.info(f"\n📁 {cache_file.name}")
            logger.info(f"   Symbols: {', '.join(cache_data['symbols'])}")
            logger.info(f"   Days: {cache_data['days']}")
            logger.info(f"   Interval: {cache_data['interval']}")
            logger.info(f"   Samples: {cache_data['samples_count']:,}")
            logger.info(f"   Size: {file_size_mb:.1f} MB")
            logger.info(f"   Cached: {cached_at.strftime('%Y-%m-%d %H:%M:%S')} ({age_hours:.1f}h ago)")

        except Exception as e:
            logger.error(f"   ❌ Error reading {cache_file.name}: {e}")

    logger.info("=" * 80)


async def download_and_cache(
    symbols: List[str] = None,
    days: int = 365,
    interval: str = "1m",
    force: bool = False
):
    """
    Скачать данные и сохранить в кэш.

    Args:
        symbols: Список символов
        days: Количество дней
        interval: Таймфрейм
        force: Перезаписать существующий кэш
    """
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'LINKUSDT', 'MATICUSDT'
        ]

    logger.info("=" * 80)
    logger.info("💾 Download and Cache Training Data")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    logger.info(f"   Days: {days}")
    logger.info(f"   Interval: {interval}")
    logger.info("=" * 80)

    # Проверяем существующий кэш
    if not force:
        existing_data = load_cached_data(symbols, days, interval)
        if existing_data is not None:
            logger.info("✅ Data already cached! Use --force to re-download.")
            return

    # Загрузка данных
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

    # Сохраняем в кэш
    metadata = {
        'downloader': 'BinanceDataDownloader',
        'indicators': 15,
        'total_requests': downloader.request_count
    }

    save_cached_data(combined_df, symbols, days, interval, metadata)

    logger.info("=" * 80)
    logger.info("🎉 Data cached successfully!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 Next steps:")
    logger.info(f"   python examples/gru_training_pytorch.py --use-cache")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and cache training data")
    parser.add_argument('--days', type=int, default=365,
                        help='Days of historical data (default: 365)')
    parser.add_argument('--interval', type=str, default='1m',
                        help='Timeframe (default: 1m)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Symbols to download (default: top 10)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if cached')
    parser.add_argument('--list', action='store_true',
                        help='List all cached data')

    args = parser.parse_args()

    if args.list:
        list_cached_data()
    else:
        asyncio.run(download_and_cache(
            symbols=args.symbols,
            days=args.days,
            interval=args.interval,
            force=args.force
        ))


if __name__ == "__main__":
    main()
