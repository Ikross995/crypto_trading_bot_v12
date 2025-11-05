"""
🧠 Online Learning System for GRU Model
=======================================

Система онлайн-обучения позволяет модели учиться на реальных сделках.

Принцип работы:
    1. Бот открывает сделку с прогнозом GRU
    2. Сделка закрывается
    3. Модель получает фидбек: насколько точным был прогноз
    4. Модель корректирует веса на основе ошибки

Преимущества:
    - Адаптация к реальным условиям вашего трейдинга
    - Учёт вашего стиля и стратегии
    - Постоянное улучшение без ручного переобучения

Настройка в .env:
    ONLINE_LEARNING_ENABLE=true
    ONLINE_LEARNING_LR=0.00001  # Очень низкий LR для стабильности
    ONLINE_LEARNING_MIN_TRADES=10  # Минимум сделок перед сохранением
    ONLINE_LEARNING_SAVE_INTERVAL=50  # Сохранять каждые N сделок
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Система онлайн-обучения для GRU модели.

    Модель учится на каждой закрытой сделке, постепенно
    адаптируясь к реальным рыночным условиям.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.00001,
        buffer_size: int = 100,
        min_trades_to_save: int = 10,
        save_interval: int = 50,
        model_save_path: Optional[str] = None
    ):
        """
        Args:
            model: GRU модель
            device: cuda или cpu
            learning_rate: LR для онлайн обучения (ОЧЕНЬ низкий!)
            buffer_size: Размер буфера опыта
            min_trades_to_save: Минимум сделок перед первым сохранением
            save_interval: Сохранять каждые N сделок
            model_save_path: Путь для сохранения модели
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate

        # Оптимизатор с очень низким LR для стабильности
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.buffer_size = buffer_size
        self.experience_buffer = deque(maxlen=buffer_size)

        # Статистика
        self.total_updates = 0
        self.min_trades_to_save = min_trades_to_save
        self.save_interval = save_interval
        self.model_save_path = model_save_path

        # Метрики
        self.recent_losses = deque(maxlen=50)
        self.recent_maes = deque(maxlen=50)

        logger.info("🧠 Online Learning System initialized")
        logger.info(f"   Learning rate: {learning_rate} (very low for stability)")
        logger.info(f"   Buffer size: {buffer_size} trades")
        logger.info(f"   Save interval: {save_interval} trades")

    def add_experience(
        self,
        input_sequence: np.ndarray,
        predicted_price: float,
        actual_price: float,
        profit: float,
        trade_info: Optional[Dict] = None
    ):
        """
        Добавить опыт из закрытой сделки.

        Args:
            input_sequence: Входная последовательность (60, features)
            predicted_price: Что модель предсказала
            actual_price: Реальная цена
            profit: Прибыль/убыток от сделки
            trade_info: Дополнительная информация о сделке
        """
        experience = {
            'input_sequence': input_sequence,
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'profit': profit,
            'timestamp': datetime.now(),
            'trade_info': trade_info or {}
        }

        self.experience_buffer.append(experience)

        logger.debug(
            f"📝 Experience added: "
            f"Pred={predicted_price:.2f}, "
            f"Actual={actual_price:.2f}, "
            f"Error={abs(predicted_price - actual_price):.2f}, "
            f"Profit=${profit:.2f}"
        )

    async def learn_from_experience(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Обучить модель на накопленном опыте.

        Args:
            batch_size: Размер батча для обучения

        Returns:
            Метрики обучения
        """
        if len(self.experience_buffer) < batch_size:
            logger.debug(f"⏸️  Not enough experience yet: {len(self.experience_buffer)}/{batch_size}")
            return {}

        # Выбираем случайный батч из буфера
        import random
        batch = random.sample(list(self.experience_buffer), batch_size)

        # Подготовка данных
        X_batch = []
        y_batch = []

        for exp in batch:
            X_batch.append(exp['input_sequence'])
            y_batch.append(exp['actual_price'])

        X_batch = torch.FloatTensor(np.array(X_batch)).to(self.device)
        y_batch = torch.FloatTensor(y_batch).to(self.device)

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(X_batch)
        loss = self.criterion(predictions.squeeze(), y_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Метрики
        mae = torch.mean(torch.abs(predictions.squeeze() - y_batch)).item()

        self.recent_losses.append(loss.item())
        self.recent_maes.append(mae)
        self.total_updates += 1

        # Логирование
        logger.info(
            f"🎓 Online learning update #{self.total_updates}: "
            f"Loss={loss.item():.6f}, MAE={mae:.2f}, "
            f"Batch size={batch_size}"
        )

        # Автосохранение
        if (self.total_updates >= self.min_trades_to_save and
            self.total_updates % self.save_interval == 0):
            await self.save_model()

        return {
            'loss': loss.item(),
            'mae': mae,
            'updates': self.total_updates
        }

    async def save_model(self):
        """Сохранить обновлённую модель"""
        if not self.model_save_path:
            logger.warning("⚠️  Model save path not set, skipping save")
            return

        try:
            # Создаём директорию если нужно
            save_path = Path(self.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Загружаем существующий checkpoint чтобы сохранить config
            if save_path.exists():
                checkpoint = torch.load(save_path, map_location=self.device)
            else:
                checkpoint = {}

            # Обновляем
            checkpoint['model_state_dict'] = self.model.state_dict()
            checkpoint['online_learning_info'] = {
                'total_updates': self.total_updates,
                'last_update': datetime.now().isoformat(),
                'avg_recent_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
                'avg_recent_mae': np.mean(self.recent_maes) if self.recent_maes else 0,
                'buffer_size': len(self.experience_buffer)
            }

            torch.save(checkpoint, save_path)

            logger.info(
                f"💾 Model auto-saved after {self.total_updates} online updates "
                f"(Avg MAE: {np.mean(self.recent_maes):.2f})"
            )

        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику онлайн-обучения"""
        return {
            'total_updates': self.total_updates,
            'buffer_size': len(self.experience_buffer),
            'avg_recent_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
            'avg_recent_mae': np.mean(self.recent_maes) if self.recent_maes else 0,
            'learning_rate': self.learning_rate
        }


class TradeExperienceCollector:
    """
    Коллектор опыта для интеграции с ботом.

    Использование в runner/live.py:
        # При инициализации бота
        self.experience_collector = TradeExperienceCollector(
            online_learner=self.online_learner,
            gru_predictor=self.gru_predictor
        )

        # При открытии сделки
        await self.experience_collector.on_trade_opened(
            symbol=symbol,
            entry_price=entry_price,
            side=side,
            input_sequence=input_sequence,
            gru_prediction=gru_prediction
        )

        # При закрытии сделки
        await self.experience_collector.on_trade_closed(
            symbol=symbol,
            exit_price=exit_price,
            pnl=pnl
        )
    """

    def __init__(
        self,
        online_learner: Optional[OnlineLearner] = None,
        gru_predictor: Optional[Any] = None
    ):
        """
        Args:
            online_learner: Система онлайн-обучения
            gru_predictor: GRU предсказатель (для получения input sequence)
        """
        self.online_learner = online_learner
        self.gru_predictor = gru_predictor

        # Активные сделки (ждут закрытия)
        self.open_trades: Dict[str, Dict] = {}

        logger.info("📊 Trade Experience Collector initialized")

    async def on_trade_opened(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        input_sequence: Optional[np.ndarray] = None,
        gru_prediction: Optional[Dict] = None,
        trade_id: Optional[str] = None
    ):
        """
        Зарегистрировать открытие сделки.

        Args:
            symbol: Торговая пара
            entry_price: Цена входа
            side: LONG/SHORT
            input_sequence: Входная последовательность для GRU
            gru_prediction: Прогноз GRU модели
            trade_id: ID сделки (опционально)
        """
        if not self.online_learner:
            return

        trade_key = trade_id or f"{symbol}_{entry_price}_{datetime.now().timestamp()}"

        self.open_trades[trade_key] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'side': side,
            'input_sequence': input_sequence,
            'gru_prediction': gru_prediction,
            'opened_at': datetime.now()
        }

        logger.debug(f"📝 Trade opened registered: {symbol} @ ${entry_price} ({side})")

    async def on_trade_closed(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        trade_id: Optional[str] = None
    ):
        """
        Зарегистрировать закрытие сделки и обучить модель.

        Args:
            symbol: Торговая пара
            exit_price: Цена выхода
            pnl: Прибыль/убыток
            trade_id: ID сделки
        """
        if not self.online_learner:
            return

        # Находим соответствующую открытую сделку
        trade_key = trade_id
        if trade_key not in self.open_trades:
            # Ищем по символу (берём последнюю)
            matching_trades = [k for k in self.open_trades.keys() if symbol in k]
            if not matching_trades:
                logger.warning(f"⚠️  No open trade found for {symbol}")
                return
            trade_key = matching_trades[-1]

        trade_data = self.open_trades.pop(trade_key)

        # Добавляем опыт
        if trade_data['input_sequence'] is not None:
            predicted_price = (
                trade_data['gru_prediction']['predicted_price']
                if trade_data['gru_prediction']
                else trade_data['entry_price']
            )

            self.online_learner.add_experience(
                input_sequence=trade_data['input_sequence'],
                predicted_price=predicted_price,
                actual_price=exit_price,
                profit=pnl,
                trade_info={
                    'symbol': symbol,
                    'side': trade_data['side'],
                    'entry_price': trade_data['entry_price'],
                    'exit_price': exit_price
                }
            )

            # Обучаемся на опыте
            metrics = await self.online_learner.learn_from_experience(batch_size=16)

            if metrics:
                logger.info(
                    f"🎓 Learned from trade: {symbol} "
                    f"PnL=${pnl:.2f}, "
                    f"Updates={metrics['updates']}, "
                    f"MAE={metrics['mae']:.2f}"
                )

        logger.debug(f"✅ Trade closed and processed: {symbol} @ ${exit_price}, PnL=${pnl:.2f}")
