#!/usr/bin/env python3
"""
⏰ Automatic GRU Model Retraining
=================================

Автоматически переобучает GRU модель по расписанию.

Запуск:
    # Запустить в фоне (переобучает каждое воскресенье в 02:00)
    python scripts/auto_retrain_gru.py

    # Запустить с другим расписанием
    python scripts/auto_retrain_gru.py --schedule daily --time 03:00

    # Запустить один раз сейчас
    python scripts/auto_retrain_gru.py --run-now

Расписание:
    - daily: Каждый день в указанное время
    - weekly: Каждое воскресенье
    - monthly: 1-го числа каждого месяца

Как работает:
    1. Загружает последние 30 дней данных
    2. Дотренировывает модель 5 эпох
    3. Сохраняет обновлённую модель
    4. Создаёт резервную копию старой модели
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import schedule
import shutil

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорт функции дотренировки
from scripts.finetune_gru import finetune_model

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/auto_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoRetrainer:
    """
    Автоматический переобучатель GRU модели.
    """

    def __init__(
        self,
        model_path: str = "models/checkpoints/gru_model_pytorch.pt",
        days: int = 30,
        epochs: int = 5,
        backup_dir: str = "models/backups"
    ):
        self.model_path = Path(model_path)
        self.days = days
        self.epochs = epochs
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self):
        """Создать резервную копию текущей модели"""
        if not self.model_path.exists():
            logger.warning(f"⚠️  Model not found: {self.model_path}")
            return False

        # Название бэкапа с timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"gru_model_backup_{timestamp}.pt"

        # Копируем
        shutil.copy2(self.model_path, backup_path)
        logger.info(f"💾 Backup created: {backup_path}")

        # Удаляем старые бэкапы (оставляем последние 5)
        self.cleanup_old_backups(keep=5)

        return True

    def cleanup_old_backups(self, keep: int = 5):
        """Удалить старые бэкапы, оставить последние N"""
        backups = sorted(self.backup_dir.glob("gru_model_backup_*.pt"))

        if len(backups) > keep:
            to_remove = backups[:-keep]
            for backup in to_remove:
                backup.unlink()
                logger.info(f"🗑️  Removed old backup: {backup.name}")

    async def retrain(self):
        """Выполнить переобучение модели"""
        logger.info("=" * 80)
        logger.info("⏰ AUTO-RETRAIN: Starting scheduled retraining")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        try:
            # Создаём бэкап
            logger.info("💾 Creating backup of current model...")
            self.create_backup()

            # Дотренировываем модель
            logger.info(f"🔄 Fine-tuning on last {self.days} days ({self.epochs} epochs)...")
            await finetune_model(
                model_path=str(self.model_path),
                symbols=None,  # Используем те же что и раньше
                days=self.days,
                interval="1m",
                epochs=self.epochs,
                batch_size=32,
                learning_rate=0.0001,
                save_path=None  # Перезаписываем исходную модель
            )

            logger.info("=" * 80)
            logger.info("✅ AUTO-RETRAIN: Completed successfully!")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"❌ AUTO-RETRAIN: Failed with error: {e}")
            logger.error("   Model backup is available in backups directory")
            return False

    def run_sync(self):
        """Синхронная обёртка для asyncio"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.retrain())
        loop.close()
        return result


def schedule_retraining(
    schedule_type: str = "weekly",
    time_str: str = "02:00",
    model_path: str = "models/checkpoints/gru_model_pytorch.pt",
    days: int = 30,
    epochs: int = 5
):
    """
    Настроить расписание автоматического переобучения.

    Args:
        schedule_type: 'daily', 'weekly', 'monthly'
        time_str: Время в формате HH:MM
        model_path: Путь к модели
        days: Дней свежих данных для дотренировки
        epochs: Эпох дотренировки
    """
    logger.info("=" * 80)
    logger.info("⏰ AUTO-RETRAIN: Scheduler Started")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Schedule: {schedule_type}")
    logger.info(f"   Time: {time_str}")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Fresh data: {days} days")
    logger.info(f"   Epochs: {epochs}")
    logger.info("=" * 80)

    # Создаём retrainer
    retrainer = AutoRetrainer(
        model_path=model_path,
        days=days,
        epochs=epochs
    )

    # Настраиваем расписание
    if schedule_type == "daily":
        schedule.every().day.at(time_str).do(retrainer.run_sync)
        logger.info(f"📅 Scheduled: Daily at {time_str}")

    elif schedule_type == "weekly":
        schedule.every().sunday.at(time_str).do(retrainer.run_sync)
        logger.info(f"📅 Scheduled: Every Sunday at {time_str}")

    elif schedule_type == "monthly":
        # Будем проверять 1-е число каждый день в указанное время
        def monthly_task():
            if datetime.now().day == 1:
                retrainer.run_sync()

        schedule.every().day.at(time_str).do(monthly_task)
        logger.info(f"📅 Scheduled: 1st of every month at {time_str}")

    else:
        logger.error(f"❌ Unknown schedule type: {schedule_type}")
        return

    logger.info("")
    logger.info("🚀 Scheduler is running... (Press Ctrl+C to stop)")
    logger.info(f"   Next run: {schedule.next_run()}")
    logger.info("")

    # Запускаем scheduler loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Проверяем каждую минуту

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 80)
        logger.info("⏸️  AUTO-RETRAIN: Scheduler stopped by user")
        logger.info("=" * 80)


def run_now(model_path: str, days: int, epochs: int):
    """Запустить переобучение прямо сейчас"""
    logger.info("🚀 Running retraining now...")

    retrainer = AutoRetrainer(
        model_path=model_path,
        days=days,
        epochs=epochs
    )

    success = retrainer.run_sync()

    if success:
        logger.info("✅ Retraining completed successfully!")
    else:
        logger.error("❌ Retraining failed!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Automatic GRU model retraining")

    parser.add_argument('--schedule', type=str, default='weekly',
                        choices=['daily', 'weekly', 'monthly'],
                        help='Retraining schedule (default: weekly)')
    parser.add_argument('--time', type=str, default='02:00',
                        help='Time to run (HH:MM format, default: 02:00)')
    parser.add_argument('--model', type=str, default='models/checkpoints/gru_model_pytorch.pt',
                        help='Path to model')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of fresh data (default: 30)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs for fine-tuning (default: 5)')
    parser.add_argument('--run-now', action='store_true',
                        help='Run retraining immediately instead of scheduling')

    args = parser.parse_args()

    # Создаём директорию для логов
    Path('logs').mkdir(exist_ok=True)

    if args.run_now:
        # Запустить сейчас
        run_now(args.model, args.days, args.epochs)
    else:
        # Запустить по расписанию
        schedule_retraining(
            schedule_type=args.schedule,
            time_str=args.time,
            model_path=args.model,
            days=args.days,
            epochs=args.epochs
        )


if __name__ == "__main__":
    main()
