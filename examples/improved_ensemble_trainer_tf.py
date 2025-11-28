#!/usr/bin/env python3
"""
🚀 IMPROVED ENSEMBLE TRAINER - TensorFlow/Keras Version
========================================================

Современная архитектура 2025 на TensorFlow/Keras:
- Multi-Head Attention (Transformer-like)
- Positional Encoding
- Bidirectional GRU
- Layer Normalization
- Residual Connections
- Mixed Precision Training
- Advanced Callbacks

Оптимизировано для RTX 5070 Ti с TensorFlow GPU.

Автор: Claude (Anthropic) - 2025
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install: pip install tensorflow[and-cuda]")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 🎯 POSITIONAL ENCODING
# ==========================================

class PositionalEncoding(layers.Layer):
    """Positional Encoding для временных последовательностей"""

    def __init__(self, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.pe = None

    def build(self, input_shape):
        _, seq_len, d_model = input_shape

        # Create positional encoding matrix
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((self.max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Convert to TensorFlow variable
        self.pe = tf.Variable(
            initial_value=pe.astype(np.float32),
            trainable=False,
            name='positional_encoding'
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]


# ==========================================
# 🧠 MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(layers.Layer):
    """Multi-Head Attention как в Transformer"""

    def __init__(self, d_model, num_heads=8, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
        self.W_o = layers.Dense(d_model)

        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()

    def split_heads(self, x, batch_size):
        """Split last dimension into (num_heads, d_k)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]

        # Linear projections
        Q = self.split_heads(self.W_q(x), batch_size)
        K = self.split_heads(self.W_k(x), batch_size)
        V = self.split_heads(self.W_v(x), batch_size)

        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, tf.float32))
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, training=training)

        # Apply attention to values
        context = tf.matmul(attn, V)

        # Concatenate heads
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.W_o(context)

        # Residual connection + LayerNorm
        output = self.layer_norm(x + self.dropout(output, training=training))

        return output


# ==========================================
# 🔥 IMPROVED GRU MODEL
# ==========================================

def create_improved_gru_model(
    input_shape: Tuple[int, int],
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.3,
    use_attention: bool = True,
    bidirectional: bool = True,
    name: str = 'ImprovedGRU'
) -> keras.Model:
    """
    Создать улучшенную GRU модель с современной архитектурой

    Args:
        input_shape: (sequence_length, num_features)
        hidden_size: Размер скрытого слоя
        num_layers: Количество GRU слоёв
        num_heads: Количество attention heads
        dropout: Dropout rate
        use_attention: Использовать Multi-Head Attention
        bidirectional: Использовать Bidirectional GRU
        name: Имя модели

    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape, name='input')

    # Input projection to hidden_size
    x = layers.Dense(hidden_size, name='input_proj')(inputs)
    x = layers.LayerNormalization(name='input_norm')(x)

    # Positional encoding
    x = PositionalEncoding(name='pos_encoding')(x)

    # Multi-Head Attention (опционально)
    if use_attention:
        x = MultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            name='multi_head_attention'
        )(x)

    # Bidirectional GRU layers
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)  # Last layer doesn't return sequences

        gru_layer = layers.GRU(
            hidden_size,
            return_sequences=return_sequences,
            dropout=dropout if num_layers > 1 else 0,
            name=f'gru_{i+1}'
        )

        if bidirectional:
            x = layers.Bidirectional(gru_layer, name=f'bidirectional_gru_{i+1}')(x)
            # After bidirectional, we have 2*hidden_size
            if i < num_layers - 1:
                # Project back to hidden_size for next layer
                x = layers.Dense(hidden_size, name=f'proj_{i+1}')(x)
                x = layers.LayerNormalization(name=f'norm_{i+1}')(x)
        else:
            x = gru_layer(x)

    # Post-GRU normalization
    gru_output_size = hidden_size * 2 if bidirectional and num_layers == 1 else hidden_size
    x = layers.LayerNormalization(name='gru_norm')(x)

    # Dense layers with residual
    x = layers.Dense(hidden_size, activation='relu', name='fc1')(x)
    x = layers.LayerNormalization(name='fc1_norm')(x)
    x = layers.Dropout(dropout, name='dropout')(x)

    # Output layer
    outputs = layers.Dense(1, name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model


# ==========================================
# 📊 IMPROVED LOSS FUNCTION
# ==========================================

class ImprovedLoss(keras.losses.Loss):
    """
    Улучшенная функция потерь:
    - Huber Loss для устойчивости к выбросам
    - Directional Accuracy penalty
    """

    def __init__(self, delta=1.0, direction_weight=0.3, name='improved_loss'):
        super().__init__(name=name)
        self.delta = delta
        self.direction_weight = direction_weight
        self.huber = keras.losses.Huber(delta=delta)

    def call(self, y_true, y_pred):
        # Huber loss
        huber_loss = self.huber(y_true, y_pred)

        # Directional accuracy penalty
        pred_sign = tf.sign(y_pred)
        true_sign = tf.sign(y_true)
        direction_correct = tf.cast(tf.equal(pred_sign, true_sign), tf.float32)
        direction_loss = 1.0 - tf.reduce_mean(direction_correct)

        # Combined loss
        total_loss = huber_loss + self.direction_weight * direction_loss

        return total_loss


# ==========================================
# 🎲 DATA AUGMENTATION
# ==========================================

class TimeSeriesAugmentation:
    """Аугментация временных рядов для TensorFlow"""

    @staticmethod
    def add_noise(x, noise_level=0.01):
        """Добавить гауссов шум"""
        noise = tf.random.normal(tf.shape(x), stddev=noise_level)
        return x + noise

    @staticmethod
    def scale(x, scale_range=(0.95, 1.05)):
        """Масштабирование"""
        scale_factor = tf.random.uniform(
            (tf.shape(x)[0], 1, 1),
            minval=scale_range[0],
            maxval=scale_range[1]
        )
        return x * scale_factor

    @staticmethod
    def augment(x, training=True):
        """Применить аугментацию"""
        if not training:
            return x

        # 50% вероятность добавить шум
        if tf.random.uniform(()) < 0.5:
            x = TimeSeriesAugmentation.add_noise(x, 0.01)

        # 50% вероятность масштабировать
        if tf.random.uniform(()) < 0.5:
            x = TimeSeriesAugmentation.scale(x, (0.98, 1.02))

        return x


# ==========================================
# 📊 MODEL CONFIGS
# ==========================================

IMPROVED_ENSEMBLE_CONFIGS_TF = {
    'attention_deep': {
        'name': 'TF Attention Deep',
        'hidden_size': 128,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'use_attention': True,
        'bidirectional': True,
        'description': '3-layer bidirectional GRU with multi-head attention'
    },
    'attention_wide': {
        'name': 'TF Attention Wide',
        'hidden_size': 256,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 0.0003,
        'use_attention': True,
        'bidirectional': True,
        'description': 'Wide model with 256 units and attention'
    },
    'gru_conservative': {
        'name': 'TF GRU Conservative',
        'hidden_size': 128,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.4,
        'learning_rate': 0.0005,
        'use_attention': False,
        'bidirectional': True,
        'description': 'Safe bidirectional GRU without attention'
    },
    'attention_aggressive': {
        'name': 'TF Attention Aggressive',
        'hidden_size': 160,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'use_attention': True,
        'bidirectional': True,
        'description': 'Fast learner with attention'
    },
    'attention_balanced': {
        'name': 'TF Attention Balanced',
        'hidden_size': 192,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 0.0007,
        'use_attention': True,
        'bidirectional': True,
        'description': 'Balanced configuration with attention'
    }
}


# ==========================================
# 🎯 IMPROVED ENSEMBLE TRAINER
# ==========================================

class ImprovedEnsembleTrainerTF:
    """Улучшенный тренер ансамбля на TensorFlow"""

    def __init__(self, configs: Dict = None):
        self.configs = configs or IMPROVED_ENSEMBLE_CONFIGS_TF

        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                logger.info(f"🎮 GPU: {gpus[0].name}")
                logger.info(f"🚀 Mixed Precision: Enabled")

                # Enable mixed precision
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)

            except RuntimeError as e:
                logger.warning(f"⚠️ GPU setup error: {e}")
        else:
            logger.warning("⚠️ No GPU found - using CPU")

        self.models = {}
        self.histories = {}

    def train_ensemble(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=512
    ):
        """Обучить весь ансамбль"""

        logger.info("=" * 80)
        logger.info("🚀 IMPROVED ENSEMBLE TRAINING (TensorFlow)")
        logger.info("=" * 80)
        logger.info(f"Models: {len(self.configs)}")
        logger.info(f"Train: {len(X_train):,} samples")
        logger.info(f"Val: {len(X_val):,} samples")
        logger.info(f"Batch size: {batch_size}")
        logger.info("=" * 80)

        input_shape = (X_train.shape[1], X_train.shape[2])

        for model_name, config in self.configs.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"🔧 Training: {config['name']}")
            logger.info(f"{'='*80}")
            logger.info(f"Config: {config['description']}")

            start_time = time.time()

            # Create model
            model = create_improved_gru_model(
                input_shape=input_shape,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                num_heads=config.get('num_heads', 8),
                dropout=config['dropout'],
                use_attention=config.get('use_attention', True),
                bidirectional=config.get('bidirectional', True),
                name=model_name
            )

            # Count parameters
            total_params = model.count_params()
            logger.info(f"📊 Parameters: {total_params:,}")

            # Compile model
            optimizer = keras.optimizers.AdamW(
                learning_rate=config['learning_rate'],
                weight_decay=1e-4
            )

            model.compile(
                optimizer=optimizer,
                loss=ImprovedLoss(delta=1.0, direction_weight=0.3),
                metrics=['mae']
            )

            # Callbacks
            callbacks = [
                # Early stopping
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),

                # Learning rate reduction
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=config['learning_rate'] * 0.01,
                    verbose=1
                ),

                # Model checkpoint
                ModelCheckpoint(
                    f'models/checkpoints/{model_name}_best.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
            ]

            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Save
            self.models[model_name] = model
            self.histories[model_name] = history

            training_time = time.time() - start_time
            best_val_loss = min(history.history['val_loss'])

            logger.info(f"✅ {config['name']} completed in {training_time/60:.1f} min")
            logger.info(f"   Best val loss: {best_val_loss:.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("🎯 ENSEMBLE TRAINING COMPLETED")
        logger.info("=" * 80)

        return self.histories

    def save_ensemble(self, save_dir: str):
        """Сохранить ансамбль"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = save_path / f"{name}_tf.keras"
            model.save(model_path)
            logger.info(f"✅ Saved {name} to {model_path}")

    def load_ensemble(self, load_dir: str):
        """Загрузить ансамбль"""
        load_path = Path(load_dir)

        for model_file in load_path.glob("*_tf.keras"):
            model_name = model_file.stem.replace('_tf', '')
            self.models[model_name] = keras.models.load_model(
                model_file,
                custom_objects={'ImprovedLoss': ImprovedLoss}
            )
            logger.info(f"✅ Loaded {model_name}")


# ==========================================
# 🧪 EXAMPLE
# ==========================================

if __name__ == "__main__":
    logger.info("🚀 Improved Ensemble Trainer TF - Ready!")
    logger.info("=" * 80)
    logger.info("GPU Check:")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"  ✅ {len(gpus)} GPU(s) found")
        for gpu in gpus:
            logger.info(f"     - {gpu.name}")
    else:
        logger.info("  ⚠️ No GPU found")

    logger.info("=" * 80)
