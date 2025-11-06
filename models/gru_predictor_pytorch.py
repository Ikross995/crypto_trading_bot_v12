"""
🧠 PyTorch GRU Price Predictor
================================

PyTorch-based GRU model for real-time cryptocurrency price prediction.

Features:
- GPU acceleration (CUDA support)
- Real-time inference (<50ms)
- 22 features (price + volume indicators)
- Trained on real Binance data
- MinMaxScaler normalization (0-1)

Performance:
- MAE: $2.04
- MAPE: 30.50%
- Val Loss: 0.000000
"""

import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class GRUPriceModel(nn.Module):
    """
    PyTorch GRU model architecture.

    Architecture:
    - GRU Layer 1: 100 units, dropout=0.2
    - GRU Layer 2: 50 units, dropout=0.2
    - Dense: 25 units (ReLU)
    - Output: 1 unit (price prediction)
    """

    def __init__(self, input_features: int, sequence_length: int):
        super(GRUPriceModel, self).__init__()

        self.input_features = input_features
        self.sequence_length = sequence_length

        # GRU layers
        self.gru1 = nn.GRU(
            input_size=input_features,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(0.2)

        self.gru2 = nn.GRU(
            input_size=100,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(0.2)

        # Dense layers
        self.fc1 = nn.Linear(50, 25)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)

        # Output
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        # x shape: (batch, sequence_length, features)

        # GRU layer 1
        out, _ = self.gru1(x)
        out = self.dropout1(out)

        # GRU layer 2
        out, hidden = self.gru2(out)
        out = self.dropout2(out)

        # Take last output
        out = out[:, -1, :]  # (batch, 50)

        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)

        # Output
        out = self.fc2(out)

        return out


class GRUPredictorPyTorch:
    """
    PyTorch GRU predictor for real-time price prediction.

    Usage:
        predictor = GRUPredictorPyTorch()
        predictor.load("models/checkpoints/gru_model_pytorch.pt")

        # Predict next price
        price = predictor.predict(recent_data)
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize PyTorch GRU predictor.

        Args:
            device: "auto", "cuda", or "cpu"
        """
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model components
        self.model: Optional[GRUPriceModel] = None
        self.feature_scaler = None
        self.target_scaler = None

        # Model config
        self.config: Dict[str, Any] = {}
        self.feature_columns: List[str] = []

        logger.info(f"🧠 GRUPredictorPyTorch initialized on device: {self.device}")

    def load(self, model_path: str):
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to .pt checkpoint file
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"📂 Loading model from: {model_path}")

        # Load checkpoint (weights_only=False to allow sklearn scalers)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract config
        self.config = checkpoint['model_config']
        self.feature_columns = self.config['feature_columns']

        # 🔥 Check model version and load appropriate architecture
        model_version = self.config.get('model_version', 'v1_absolute')
        target_type = self.config.get('target_type', 'absolute_price')

        if model_version == 'v2_percentage' or target_type == 'percentage_change':
            # 🔥 NEW MODEL: Enhanced architecture
            logger.info("🔥 Loading ENHANCED GRU model (% change prediction)")
            try:
                from models.gru_model_enhanced import EnhancedGRUModel
                self.model = EnhancedGRUModel(
                    input_features=self.config['input_features'],
                    sequence_length=self.config['sequence_length']
                )
            except ImportError:
                logger.warning("⚠️ EnhancedGRUModel not found, using default")
                self.model = GRUPriceModel(
                    input_features=self.config['input_features'],
                    sequence_length=self.config['sequence_length']
                )
        else:
            # 🔴 OLD MODEL: Original architecture
            logger.info("📊 Loading standard GRU model (absolute price prediction)")
            self.model = GRUPriceModel(
                input_features=self.config['input_features'],
                sequence_length=self.config['sequence_length']
            )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Load scalers
        self.feature_scaler = checkpoint['scalers']['feature_scaler']
        self.target_scaler = checkpoint['scalers']['target_scaler']

        # Log scaler ranges (important for debugging!)
        if hasattr(self.target_scaler, 'data_min_') and hasattr(self.target_scaler, 'data_max_'):
            if target_type == 'percentage_change':
                logger.info(f"🎯 Target scaler range: {self.target_scaler.data_min_[0]:.2f}% - {self.target_scaler.data_max_[0]:.2f}%")
            else:
                logger.info(f"🎯 Target scaler range: ${self.target_scaler.data_min_[0]:.2f} - ${self.target_scaler.data_max_[0]:.2f}")
        else:
            logger.warning("⚠️ Target scaler doesn't have data_min_/data_max_ attributes")

        # Log info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   Model version: {model_version}")
        logger.info(f"   Target type: {target_type}")
        logger.info(f"   Input features: {self.config['input_features']}")
        logger.info(f"   Sequence length: {self.config['sequence_length']}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Device: {self.device}")

        if 'final_metrics' in checkpoint:
            metrics = checkpoint['final_metrics']
            logger.info(f"   MAE: ${metrics['mae']:.2f}")
            logger.info(f"   MAPE: {metrics['mape']:.2f}%")

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate technical indicators for prediction.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Feature array (normalized, ready for model)
        """
        df = df.copy()

        # Price features (4)
        # Already have: open, high, low, volume

        # Technical indicators (11)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_mid'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)

        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        # ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        # Volume indicators (7)
        # 1. Volume Delta
        df['volume_delta'] = df['volume'] * np.where(
            df['close'] > df['open'], 1, -1
        )

        # 2. OBV
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

        # 3. Volume Ratio
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        # 4. Volume Spike
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(float)

        # 5. MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(df['close'] <= df['close'].shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-10)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))

        # 6. CVD
        df['cvd'] = df['volume_delta'].cumsum()

        # 7. VWAP Distance
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

        # Extract feature columns
        features = df[self.feature_columns].values

        # Fill NaN with 0 (first rows will have NaN from indicators)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predict next price based on recent data.

        Args:
            df: DataFrame with recent OHLCV data (at least sequence_length rows)

        Returns:
            Predicted price (denormalized)
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load() first.")

        sequence_length = self.config['sequence_length']

        if len(df) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} candles, got {len(df)}"
            )

        # Take last sequence_length candles
        df_recent = df.tail(sequence_length).copy()

        # Calculate features
        features = self.prepare_features(df_recent)

        # Normalize
        features_normalized = self.feature_scaler.transform(features)

        # Take last sequence_length rows
        sequence = features_normalized[-sequence_length:]

        # Convert to tensor
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        # Shape: (1, sequence_length, features)

        # Predict
        with torch.no_grad():
            prediction_normalized = self.model(X).cpu().numpy()

        # Debug: Log normalized prediction
        logger.debug(f"🔢 Normalized prediction: {prediction_normalized[0, 0]:.6f}")

        # Denormalize
        prediction_or_pct = self.target_scaler.inverse_transform(
            prediction_normalized.reshape(-1, 1)
        )[0, 0]

        # 🔥 Check model version
        current_price = float(df_recent['close'].iloc[-1])
        model_version = self.config.get('model_version', 'v1_absolute')
        target_type = self.config.get('target_type', 'absolute_price')

        if target_type == 'percentage_change' or model_version == 'v2_percentage':
            # 🔥 NEW MODEL: Prediction is % change
            pct_change = prediction_or_pct
            predicted_price = current_price * (1 + pct_change / 100)

            logger.debug(f"🔢 Denormalized % change: {pct_change:+.2f}%")
            logger.debug(f"🔢 Predicted price: ${predicted_price:.2f} (current: ${current_price:.2f})")

            return float(predicted_price)
        else:
            # 🔴 OLD MODEL: Prediction is absolute price
            predicted_price = prediction_or_pct

            logger.debug(f"🔢 Denormalized prediction: ${predicted_price:.2f} (current: ${current_price:.2f})")

            return float(predicted_price)

    def predict_batch(self, df_list: List[pd.DataFrame]) -> List[float]:
        """
        Predict multiple prices in batch (faster for multiple symbols).

        Args:
            df_list: List of DataFrames with recent OHLCV data

        Returns:
            List of predicted prices
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load() first.")

        sequence_length = self.config['sequence_length']

        # Prepare all sequences
        sequences = []
        for df in df_list:
            if len(df) < sequence_length:
                raise ValueError(f"Need at least {sequence_length} candles")

            df_recent = df.tail(sequence_length).copy()
            features = self.prepare_features(df_recent)
            features_normalized = self.feature_scaler.transform(features)
            sequence = features_normalized[-sequence_length:]
            sequences.append(sequence)

        # Stack into batch
        X = torch.FloatTensor(np.array(sequences)).to(self.device)
        # Shape: (batch_size, sequence_length, features)

        # Predict
        with torch.no_grad():
            predictions_normalized = self.model(X).cpu().numpy()

        # Denormalize
        predictions = self.target_scaler.inverse_transform(
            predictions_normalized.reshape(-1, 1)
        ).flatten()

        return predictions.tolist()

    def get_feature_columns(self) -> List[str]:
        """Get list of required feature columns."""
        return self.feature_columns.copy()

    def get_sequence_length(self) -> int:
        """Get required sequence length."""
        return self.config.get('sequence_length', 60)


__all__ = ['GRUPredictorPyTorch', 'GRUPriceModel']
