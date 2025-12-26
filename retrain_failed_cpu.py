"""
Переобучение упавших символов на CPU (безопасно, без CUDA)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only

import asyncio
import logging
from run_full_combo_system_multi import train_single_symbol

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


async def retrain_failed_symbols():
    """Переобучить символы, которые упали с CUDA ошибкой."""

    failed_symbols = [
        'XRPUSDT',
        'DOGEUSDT',
        'AVAXUSDT',
        'LINKUSDT',
        'APTUSDT'
    ]

    logger.info("="*80)
    logger.info("🔄 ПЕРЕОБУЧЕНИЕ УПАВШИХ СИМВОЛОВ НА CPU")
    logger.info("="*80)
    logger.info(f"Символов для переобучения: {len(failed_symbols)}")
    logger.info(f"Режим: CPU только (без CUDA)")
    logger.info("")

    results = {}

    for i, symbol in enumerate(failed_symbols, 1):
        logger.info("")
        logger.info("="*80)
        logger.info(f"📊 [{i}/{len(failed_symbols)}] Обучение {symbol} на CPU...")
        logger.info("="*80)
        logger.info("")

        try:
            result = await train_single_symbol(
                symbol=symbol,
                days=365,
                interval='30m',
                quick_mode=True  # Быстрый режим для CPU
            )

            results[symbol] = {
                'status': 'SUCCESS',
                'result': result
            }

            logger.info(f"\n✅ {symbol} - УСПЕШНО обучен на CPU\n")

        except Exception as e:
            results[symbol] = {
                'status': 'FAILED',
                'error': str(e)
            }

            logger.error(f"\n❌ {symbol} - ОШИБКА: {e}\n")
            import traceback
            logger.error(traceback.format_exc())

    # Итоговый отчет
    logger.info("")
    logger.info("="*80)
    logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("="*80)

    success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    failed_count = len(results) - success_count

    logger.info(f"\nУспешно: {success_count}/{len(failed_symbols)}")
    logger.info(f"Ошибок: {failed_count}/{len(failed_symbols)}")
    logger.info("")

    for symbol, result in results.items():
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        logger.info(f"{status_icon} {symbol}: {result['status']}")

    logger.info("")
    logger.info("="*80)


if __name__ == '__main__':
    asyncio.run(retrain_failed_symbols())
