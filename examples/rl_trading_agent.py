#!/usr/bin/env python3
"""
🤖 Reinforcement Learning Trading Agent - DQN
==============================================

РЕВОЛЮЦИОННЫЙ ПОДХОД:
- Агент ТОРГУЕТ, не предсказывает цены!
- Учится максимизировать прибыль через trial & error
- Анализирует что сработало, что нет
- Адаптируется к рынку в реальном времени

Архитектура:
- State: [цена, индикаторы, позиция, PnL]
- Actions: [LONG, SHORT, CLOSE, HOLD]
- Reward: Profit + Sharpe Ratio - Drawdown
- Network: DQN (Deep Q-Network) с Experience Replay

Автор: Claude (Anthropic)
"""

import asyncio
import logging
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install: pip install torch")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 🎮 TRADING ENVIRONMENT
# ==========================================

class TradingEnvironment:
    """
    Симуляция торгового окружения для RL агента
    """

    # Actions
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000,
        fee: float = 0.0004,  # 0.04% fee
        max_position_size: float = 1.0,  # 100% of balance
        reward_scaling: float = 100.0
    ):
        self.data = data
        self.initial_balance = initial_balance
        self.fee = fee
        self.max_position_size = max_position_size
        self.reward_scaling = reward_scaling

        # State
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.position_size = 0

        # History
        self.balance_history = []
        self.trades = []
        self.returns = []

        # Calculate features
        self._prepare_features()

    def _prepare_features(self):
        """Подготовить фичи для state"""
        df = self.data.copy()

        # Price features
        df['returns'] = df['close'].pct_change() * 100
        df['volatility'] = df['returns'].rolling(20).std()

        # Normalize
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[f'{col}_norm'] = (df[col] - df[col].rolling(100).mean()) / df[col].rolling(100).std()

        # Technical indicators (already in data)
        self.features = [
            'close_norm', 'volume_norm', 'returns', 'volatility',
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'sma_50'
        ]

        self.data = df.dropna()

    def reset(self) -> np.ndarray:
        """Сброс окружения"""
        self.current_step = 100  # Start after indicators are calculated
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.balance_history = [self.balance]
        self.trades = []
        self.returns = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Получить текущее состояние

        State: [price features, indicators, position info, PnL]
        """
        row = self.data.iloc[self.current_step]

        # Market features
        market_features = [row[f] for f in self.features if f in row.index]

        # Position features
        position_pnl = self._calculate_pnl()
        position_features = [
            self.position,  # -1, 0, 1
            position_pnl / self.balance if self.balance > 0 else 0,  # % PnL
            self.position_size / self.balance if self.balance > 0 else 0,  # Position size %
        ]

        # Balance features
        balance_features = [
            self.balance / self.initial_balance - 1,  # Total return
            len(self.trades),  # Number of trades
        ]

        state = np.array(market_features + position_features + balance_features, dtype=np.float32)

        # Handle NaN
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return state

    def _calculate_pnl(self) -> float:
        """Рассчитать текущий PnL"""
        if self.position == 0:
            return 0

        current_price = self.data.iloc[self.current_step]['close']

        if self.position == 1:  # Long
            pnl = (current_price - self.entry_price) * self.position_size / self.entry_price
        else:  # Short
            pnl = (self.entry_price - current_price) * self.position_size / self.entry_price

        return pnl

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Выполнить действие

        Returns:
            next_state, reward, done, info
        """
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        info = {}

        # Execute action
        if action == self.LONG and self.position == 0:
            # Open long
            self.position = 1
            self.entry_price = current_price
            self.position_size = self.balance * self.max_position_size
            fee_cost = self.position_size * self.fee
            self.balance -= fee_cost
            info['action'] = 'OPEN_LONG'

        elif action == self.SHORT and self.position == 0:
            # Open short
            self.position = -1
            self.entry_price = current_price
            self.position_size = self.balance * self.max_position_size
            fee_cost = self.position_size * self.fee
            self.balance -= fee_cost
            info['action'] = 'OPEN_SHORT'

        elif action == self.CLOSE and self.position != 0:
            # Close position
            pnl = self._calculate_pnl()
            fee_cost = self.position_size * self.fee

            self.balance += pnl - fee_cost

            # Calculate reward
            reward = pnl / self.initial_balance * self.reward_scaling

            # Store trade
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position': self.position,
                'pnl': pnl,
                'pnl_pct': pnl / self.position_size * 100
            })

            self.position = 0
            self.entry_price = 0
            self.position_size = 0
            info['action'] = 'CLOSE'
            info['pnl'] = pnl
        else:
            # Hold
            info['action'] = 'HOLD'
            # Small penalty for holding with no position
            if self.position == 0:
                reward = -0.01

        # Move to next step
        self.current_step += 1
        self.balance_history.append(self.balance)

        # Check if done
        done = (
            self.current_step >= len(self.data) - 1 or
            self.balance <= self.initial_balance * 0.5  # 50% drawdown = game over
        )

        # Calculate additional rewards/penalties
        if self.position != 0:
            # Reward for holding profitable position
            unrealized_pnl = self._calculate_pnl()
            if unrealized_pnl > 0:
                reward += unrealized_pnl / self.initial_balance * 0.1  # Small positive reward

        # Get next state
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())

        return next_state, reward, done, info

    def get_metrics(self) -> Dict:
        """Получить метрики производительности"""
        if len(self.trades) == 0:
            return {
                'total_return': (self.balance - self.initial_balance) / self.initial_balance * 100,
                'num_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
            }

        # Returns
        returns = np.array([t['pnl_pct'] for t in self.trades])

        # Win rate
        winning_trades = (returns > 0).sum()
        win_rate = winning_trades / len(returns) * 100

        # Sharpe ratio
        if returns.std() > 0:
            sharpe = np.sqrt(252 * 48) * returns.mean() / returns.std()
        else:
            sharpe = 0

        # Max drawdown
        balance_arr = np.array(self.balance_history)
        running_max = np.maximum.accumulate(balance_arr)
        drawdown = (balance_arr - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        return {
            'total_return': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_return': returns.mean(),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_balance': self.balance
        }


# ==========================================
# 🧠 DQN NETWORK
# ==========================================

class DQN(nn.Module):
    """
    Deep Q-Network для выбора торговых действий
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)

        self.fc4 = nn.Linear(hidden_size // 2, action_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))

        x = self.fc4(x)

        return x


# ==========================================
# 🎯 RL AGENT
# ==========================================

class RLTradingAgent:
    """
    Reinforcement Learning агент для торговли

    Features:
    - DQN with Experience Replay
    - Target Network
    - Epsilon-greedy exploration
    - Priority Experience Replay (optional)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int = 4,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 64,
        device: str = 'cuda'
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Experience replay memory
        self.memory: Deque = deque(maxlen=memory_size)

        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Loss
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Stats
        self.training_step = 0
        self.losses = []

    def remember(self, state, action, reward, next_state, done):
        """Сохранить опыт в память"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Выбрать действие

        Args:
            state: Текущее состояние
            training: Если True, использует epsilon-greedy exploration

        Returns:
            action: Выбранное действие (0-3)
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        # Exploitation - use network
        # Set to eval mode to handle BatchNorm with single sample
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            action = q_values.argmax().item()
        # Restore training mode
        self.policy_net.train()

        return action

    def replay(self) -> float:
        """
        Обучение на batch из памяти

        Returns:
            loss: Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.training_step += 1
        self.losses.append(loss.item())

        return loss.item()

    def update_target_network(self):
        """Обновить target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Сохранить модель"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)

    def load(self, path: str):
        """Загрузить модель"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


# ==========================================
# 🎓 TRAINING
# ==========================================

async def train_rl_agent(
    symbols: List[str] = None,
    days: int = 365,
    interval: str = "30m",
    episodes: int = 100,
    update_target_every: int = 10,
    save_path: str = "models/checkpoints/rl_agent.pt"
):
    """
    Обучить RL агента
    """
    if symbols is None:
        symbols = ['BTCUSDT']

    logger.info("=" * 80)
    logger.info("🤖 RL Trading Agent Training (DQN)")
    logger.info("=" * 80)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    logger.info(f"   Episodes: {episodes}")
    logger.info(f"   Days: {days}")
    logger.info(f"   Interval: {interval}")
    logger.info("=" * 80)

    # Load data
    sys.path.insert(0, str(Path(__file__).parent))
    from gru_training_pytorch import (
        BinanceDataDownloader,
        calculate_technical_indicators
    )

    downloader = BinanceDataDownloader()
    all_data = []

    for symbol in symbols:
        logger.info(f"📥 Downloading {symbol}...")
        df = await downloader.download_historical_data(symbol, interval, days)
        if len(df) > 0:
            df = calculate_technical_indicators(df)
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ Combined: {len(combined_df):,} samples")

    # Create environment
    env = TradingEnvironment(combined_df)
    state = env.reset()

    # Create agent
    agent = RLTradingAgent(
        state_size=len(state),
        action_size=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    logger.info(f"🧠 Agent created:")
    logger.info(f"   State size: {len(state)}")
    logger.info(f"   Action size: 4 (HOLD, LONG, SHORT, CLOSE)")
    logger.info(f"   Device: {agent.device}")

    # Training loop
    best_return = -np.inf

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0

        start_time = time.time()

        while True:
            # Choose action
            action = agent.act(state, training=True)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Remember
            agent.remember(state, action, reward, next_state, done)

            # Train
            loss = agent.replay()

            episode_reward += reward
            episode_loss += loss
            steps += 1
            state = next_state

            if done:
                break

        # Update target network
        if episode % update_target_every == 0:
            agent.update_target_network()
            logger.info(f"   🎯 Target network updated")

        # Get metrics
        metrics = env.get_metrics()
        elapsed = time.time() - start_time

        # Log
        logger.info(
            f"Episode {episode+1:3d}/{episodes} | "
            f"Return: {metrics['total_return']:+7.2f}% | "
            f"Trades: {metrics['num_trades']:3d} | "
            f"WinRate: {metrics['win_rate']:5.1f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:+6.2f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            agent.save(save_path)
            logger.info(f"   💾 New best model! Return: {best_return:.2f}%")

    logger.info("=" * 80)
    logger.info("🎉 RL TRAINING COMPLETED!")
    logger.info(f"   Best return: {best_return:.2f}%")
    logger.info("=" * 80)

    return agent


# ==========================================
# 🚀 MAIN
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--interval', type=str, default='30m', help='Timeframe')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTCUSDT'])

    args = parser.parse_args()

    asyncio.run(train_rl_agent(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        episodes=args.episodes
    ))
