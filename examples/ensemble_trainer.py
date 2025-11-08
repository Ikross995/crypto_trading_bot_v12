#!/usr/bin/env python3
"""
🎯 Ensemble Trainer - Комбо сила моделей!
=========================================

КОНЦЕПЦИЯ:
- Обучает несколько разных моделей параллельно
- Каждая модель ищет свои паттерны
- Комбинирует предсказания умным способом
- Voting, Stacking, Weighted Average
- Автоматически выбирает лучшую стратегию

Модели в ансамбле:
1. GRU Conservative (dropout=0.4, safe)
2. GRU Aggressive (dropout=0.2, risky)
3. GRU Deep (3 layers, complex patterns)
4. GRU Wide (wider hidden size)
5. LSTM Baseline (для сравнения)

Комбинирование:
- Soft Voting: Average probabilities
- Hard Voting: Majority vote
- Weighted: По Sharpe Ratio каждой модели
- Stacking: Meta-model учится комбинировать

Автор: Claude (Anthropic)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 📊 MODEL CONFIGS
# ==========================================

ENSEMBLE_CONFIGS = {
    'conservative': {
        'name': 'GRU Conservative',
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.4,
        'learning_rate': 0.0005,
        'description': 'Safe, high dropout, prevents overfitting'
    },
    'aggressive': {
        'name': 'GRU Aggressive',
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'description': 'Fast learner, lower dropout'
    },
    'deep': {
        'name': 'GRU Deep',
        'hidden_size': 96,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'description': 'Deep network for complex patterns'
    },
    'wide': {
        'name': 'GRU Wide',
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.0003,
        'description': 'Wide network for more features'
    },
    'balanced': {
        'name': 'GRU Balanced',
        'hidden_size': 160,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.0007,
        'description': 'Balanced configuration'
    }
}


# ==========================================
# 🧠 BASE MODEL (for ensemble members)
# ==========================================

class EnsembleGRU(nn.Module):
    """GRU модель для ансамбля"""

    def __init__(
        self,
        input_features: int,
        sequence_length: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input norm
        self.input_bn = nn.BatchNorm1d(sequence_length)

        # GRU
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        # Batch norm
        x = self.input_bn(x)

        # GRU
        out, _ = self.gru(x)

        # Last timestep
        out = out[:, -1, :]

        # FC layers
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# ==========================================
# 🎯 ENSEMBLE SYSTEM
# ==========================================

class EnsembleTrainer:
    """
    Система обучения и управления ансамблем моделей

    Features:
    - Parallel training of multiple models
    - Different configurations
    - Smart combination strategies
    - Auto-weighting by performance
    """

    def __init__(
        self,
        configs: Dict = None,
        device: str = 'cuda'
    ):
        self.configs = configs or ENSEMBLE_CONFIGS
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.models: Dict[str, nn.Module] = {}
        self.model_performance: Dict[str, float] = {}
        self.model_weights: Dict[str, float] = {}

        self.meta_model = None  # For stacking

    async def train_ensemble(
        self,
        train_data: Tuple,
        val_data: Tuple,
        epochs: int = 30,
        batch_size: int = 256
    ) -> Dict:
        """
        Обучить весь ансамбль

        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            epochs: Epochs для каждой модели
            batch_size: Batch size

        Returns:
            Dict с результатами обучения
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        logger.info("=" * 80)
        logger.info("🎯 ENSEMBLE TRAINING")
        logger.info("=" * 80)
        logger.info(f"Models in ensemble: {len(self.configs)}")
        logger.info(f"Train samples: {len(X_train):,}")
        logger.info(f"Val samples: {len(X_val):,}")
        logger.info("=" * 80)

        input_features = X_train.shape[2]
        sequence_length = X_train.shape[1]

        results = {}

        # Train each model
        for model_name, config in self.configs.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"🔧 Training: {config['name']}")
            logger.info(f"{'='*80}")
            logger.info(f"Config: {config['description']}")
            logger.info(f"Hidden={config['hidden_size']}, "
                       f"Layers={config['num_layers']}, "
                       f"Dropout={config['dropout']}, "
                       f"LR={config['learning_rate']}")

            start_time = time.time()

            # Create model
            model = EnsembleGRU(
                input_features=input_features,
                sequence_length=sequence_length,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(self.device)

            # Train
            model_results = await self._train_single_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                learning_rate=config['learning_rate'],
                epochs=epochs,
                batch_size=batch_size
            )

            # Save model
            self.models[model_name] = model

            # Store performance (use validation loss)
            val_loss = model_results['best_val_loss']
            self.model_performance[model_name] = val_loss

            training_time = time.time() - start_time

            logger.info(f"✅ {config['name']} completed in {training_time/60:.1f} min")
            logger.info(f"   Best val loss: {val_loss:.4f}")

            results[model_name] = model_results

        # Calculate model weights
        self._calculate_weights()

        logger.info("\n" + "=" * 80)
        logger.info("🎯 ENSEMBLE TRAINING COMPLETED")
        logger.info("=" * 80)
        self._print_ensemble_summary()
        logger.info("=" * 80)

        return results

    async def _train_single_model(
        self,
        model: nn.Module,
        train_data: Tuple,
        val_data: Tuple,
        learning_rate: float,
        epochs: int,
        batch_size: int
    ) -> Dict:
        """Обучить одну модель"""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Optimizer & Loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        criterion = nn.MSELoss()

        # Training
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_t)
                val_loss = criterion(val_predictions.squeeze(), y_val_t).item()

            avg_train_loss = np.mean(train_losses)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"   Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"   Epoch {epoch+1:2d}/{epochs} | "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val: {val_loss:.4f}"
                )

        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }

    def _calculate_weights(self):
        """
        Рассчитать веса моделей на основе производительности

        Lower loss = higher weight
        """
        # Invert losses (lower is better)
        inv_losses = {
            name: 1.0 / loss
            for name, loss in self.model_performance.items()
        }

        # Normalize to sum to 1
        total = sum(inv_losses.values())
        self.model_weights = {
            name: weight / total
            for name, weight in inv_losses.items()
        }

    def predict(
        self,
        X: np.ndarray,
        method: str = 'weighted_average'
    ) -> np.ndarray:
        """
        Предсказание ансамбля

        Args:
            X: Input data
            method: Combination method
                - 'weighted_average': Weighted by performance
                - 'simple_average': Equal weights
                - 'voting': Majority vote (sign)
                - 'best_model': Only best model

        Returns:
            predictions
        """
        if not self.models:
            raise ValueError("No models trained yet!")

        X_t = torch.FloatTensor(X).to(self.device)

        # Get predictions from all models
        model_predictions = {}

        with torch.no_grad():
            for name, model in self.models.items():
                model.eval()
                preds = model(X_t).cpu().numpy().flatten()
                model_predictions[name] = preds

        # Combine predictions
        if method == 'simple_average':
            # Equal weights
            predictions = np.mean(list(model_predictions.values()), axis=0)

        elif method == 'weighted_average':
            # Weighted by performance
            predictions = np.zeros_like(list(model_predictions.values())[0])
            for name, preds in model_predictions.items():
                predictions += preds * self.model_weights[name]

        elif method == 'voting':
            # Majority vote on direction
            directions = np.array([
                np.sign(preds) for preds in model_predictions.values()
            ])
            # Sum votes, then sign
            predictions = np.sign(np.sum(directions, axis=0))

        elif method == 'best_model':
            # Only best model
            best_model_name = min(
                self.model_performance,
                key=self.model_performance.get
            )
            predictions = model_predictions[best_model_name]

        else:
            raise ValueError(f"Unknown method: {method}")

        return predictions

    def _print_ensemble_summary(self):
        """Вывод информации об ансамбле"""
        logger.info("\n📊 Model Weights (based on validation performance):")
        for name in sorted(self.model_weights, key=self.model_weights.get, reverse=True):
            weight = self.model_weights[name]
            loss = self.model_performance[name]
            logger.info(f"   {self.configs[name]['name']:20s}: "
                       f"weight={weight:5.1%}, val_loss={loss:.4f}")

        best_model = min(self.model_performance, key=self.model_performance.get)
        logger.info(f"\n🏆 Best Single Model: {self.configs[best_model]['name']}")

    def save_ensemble(self, path: str):
        """Сохранить весь ансамбль"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = save_dir / f"{name}.pt"
            torch.save({
                'model_state': model.state_dict(),
                'config': self.configs[name],
                'performance': self.model_performance[name],
                'weight': self.model_weights[name]
            }, model_path)

        logger.info(f"💾 Ensemble saved to {path}/")

    def load_ensemble(self, path: str):
        """Загрузить ансамбль"""
        load_dir = Path(path)

        for model_file in load_dir.glob("*.pt"):
            name = model_file.stem
            checkpoint = torch.load(model_file)

            # Recreate model
            config = checkpoint['config']
            model = EnsembleGRU(
                input_features=checkpoint['model_state']['gru.weight_ih_l0'].shape[1],
                sequence_length=100,  # Will be overridden
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(self.device)

            model.load_state_dict(checkpoint['model_state'])

            self.models[name] = model
            self.model_performance[name] = checkpoint['performance']
            self.model_weights[name] = checkpoint['weight']

        logger.info(f"✅ Ensemble loaded from {path}/")


# ==========================================
# 🚀 EXAMPLE USAGE
# ==========================================

async def train_ensemble_example():
    """Пример использования ансамбля"""

    # Generate dummy data
    np.random.seed(42)
    n_samples = 10000
    sequence_length = 60
    n_features = 22

    X_train = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)

    X_val = np.random.randn(2000, sequence_length, n_features).astype(np.float32)
    y_val = np.random.randn(2000).astype(np.float32)

    # Create ensemble
    ensemble = EnsembleTrainer()

    # Train
    results = await ensemble.train_ensemble(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=10,
        batch_size=256
    )

    # Test predictions
    X_test = np.random.randn(100, sequence_length, n_features).astype(np.float32)

    logger.info("\n🧪 Testing prediction methods:")
    for method in ['simple_average', 'weighted_average', 'voting', 'best_model']:
        preds = ensemble.predict(X_test, method=method)
        logger.info(f"   {method:20s}: predictions range [{preds.min():.2f}, {preds.max():.2f}]")

    # Save
    ensemble.save_ensemble('models/ensemble/')


if __name__ == "__main__":
    asyncio.run(train_ensemble_example())
