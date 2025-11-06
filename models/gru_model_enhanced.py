"""
🔥 Enhanced GRU Model Architecture
===================================

УСИЛЕННАЯ версия GRU для крипто-трейдинга:
- 3x больше нейронов
- 3 GRU слоя вместо 2
- Batch Normalization
- Dropout для регуляризации
- Attention mechanism (опционально)

Total parameters: ~400K (вместо 61K)
"""

import torch
import torch.nn as nn


class EnhancedGRUModel(nn.Module):
    """
    🔥 УСИЛЕННАЯ GRU модель для прогнозирования % изменения цены.

    Architecture (POWERFUL):
    - Input: (batch, sequence_length, features)
    - GRU Layer 1: 256 units, dropout=0.3
    - GRU Layer 2: 128 units, dropout=0.3
    - GRU Layer 3: 64 units, dropout=0.2
    - Dense 1: 128 units, ReLU + BatchNorm
    - Dense 2: 64 units, ReLU + BatchNorm
    - Dense 3: 32 units, ReLU
    - Output: 1 unit (% price change)

    Total params: ~400,000 (vs 61,000 in old model)
    """

    def __init__(self, input_features: int, sequence_length: int):
        super(EnhancedGRUModel, self).__init__()

        self.input_features = input_features
        self.sequence_length = sequence_length

        # 🔥 GRU Layers (УВЕЛИЧЕНЫ!)
        self.gru1 = nn.GRU(
            input_size=input_features,
            hidden_size=256,  # 100 → 256
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.gru2 = nn.GRU(
            input_size=256,
            hidden_size=128,  # 50 → 128
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.gru3 = nn.GRU(  # 🔥 Новый 3-й слой!
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # 🔥 Dropout layers (УСИЛЕНЫ!)
        self.dropout1 = nn.Dropout(0.3)  # 0.2 → 0.3
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)

        # 🔥 Dense layers (БОЛЬШЕ!)
        self.fc1 = nn.Linear(64, 128)  # 50→25  стало  64→128
        self.bn1 = nn.BatchNorm1d(128)  # 🔥 Batch Normalization

        self.fc2 = nn.Linear(128, 64)  # 🔥 Новый слой!
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)  # 🔥 Новый слой!

        self.fc_out = nn.Linear(32, 1)  # Output

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, sequence_length, features)

        # GRU Layer 1
        out, _ = self.gru1(x)
        out = self.dropout1(out)

        # GRU Layer 2
        out, _ = self.gru2(out)
        out = self.dropout2(out)

        # 🔥 GRU Layer 3 (NEW!)
        out, _ = self.gru3(out)
        out = self.dropout3(out)

        # Берём последний timestep
        out = out[:, -1, :]  # (batch, 64)

        # 🔥 Dense layers (ENHANCED!)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc_out(out)

        return out


if __name__ == "__main__":
    # Test model
    model = EnhancedGRUModel(input_features=22, sequence_length=60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Enhanced GRU Model")
    print(f"   Total parameters: {total_params:,}")
    print(f"   ~{total_params / 61301:.1f}x bigger than old model")

    # Test forward pass
    x = torch.randn(32, 60, 22)  # (batch=32, seq=60, features=22)
    y = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
