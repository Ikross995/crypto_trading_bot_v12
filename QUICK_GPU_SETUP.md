# ⚡ Быстрая настройка GPU (у вас уже есть CUDA)

## Шаг 1: Проверьте версию CUDA (в PowerShell)

```powershell
nvcc --version
```

Должно показать что-то вроде:
```
Cuda compilation tools, release 12.3, V12.3.103
```

Если команда не найдена, но CUDA установлен, добавьте в PATH:
```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
```

---

## Шаг 2: Переустановите TensorFlow с CUDA поддержкой

```powershell
# 1. Удалите старый TensorFlow
pip uninstall tensorflow tensorflow-gpu tensorflow-directml -y

# 2. Установите TensorFlow с CUDA (версия 2.15+)
pip install tensorflow[and-cuda]

# ИЛИ, если предыдущая команда не работает:
pip install tensorflow==2.15.0
```

---

## Шаг 3: Проверьте GPU

```powershell
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

Должно показать:
```
TensorFlow: 2.15.0
Built with CUDA: True
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Шаг 4: Тест GPU производительности

```powershell
python -c "import tensorflow as tf; import time; print('Testing GPU...'); with tf.device('/GPU:0'): x = tf.random.normal([5000, 5000]); start = time.time(); y = tf.matmul(x, x); print('GPU Time:', round(time.time()-start, 3), 'sec'); print('Device:', y.device)"
```

Должно быть < 0.5 секунды на RTX 5070 Ti.

---

## Шаг 5: Запустите обучение GRU

```powershell
cd C:\Users\User\crypto_trading_bot_v12
python examples\gru_training_real_data.py
```

### Ожидаемое время:
- **Загрузка данных**: 5-10 минут (10 пар × 525,600 свечей)
- **Обучение на GPU**: 15-20 минут (20 эпох, 99,601 параметр)
- **Итого**: ~25-30 минут

### Логи обучения:
```
🎮 Configuring GPU...
✅ GPU available: 1 device(s)
   GPU 0: /physical_device:GPU:0

📥 Downloading BTCUSDT (1/10)...
⏱️  Rate limit check: 5/2400 weight used
✅ Downloaded 525,600 candles (365.0 days)

📥 Downloading ETHUSDT (2/10)...
...

📊 Combined dataset: 5,255,500 samples
📊 Features: 15 (open, high, low, volume, rsi, macd, ...)
📊 Training samples: 4,204,055
📊 Testing samples: 1,051,013

🧠 Building GRU model...
✅ Model architecture:
   - Input: (60, 15)
   - GRU Layer 1: 100 units, dropout=0.2
   - GRU Layer 2: 50 units, dropout=0.2
   - Dense: 25 units
   - Output: 1 unit
   Total parameters: 99,601

🎯 Training started...

Epoch 1/20
 131376/131376 [==============================] - 45s 343us/step - loss: 0.0234 - mae: 0.0891 - val_loss: 0.0198 - val_mae: 0.0812
Epoch 2/20
 131376/131376 [==============================] - 43s 327us/step - loss: 0.0156 - mae: 0.0723 - val_loss: 0.0142 - val_mae: 0.0689
Epoch 3/20
 131376/131376 [==============================] - 43s 327us/step - loss: 0.0128 - mae: 0.0654 - val_loss: 0.0119 - val_mae: 0.0621
...
Epoch 20/20
 131376/131376 [==============================] - 42s 320us/step - loss: 0.0089 - mae: 0.0512 - val_loss: 0.0091 - val_mae: 0.0518

📊 Training completed in 14.2 minutes
📊 Final metrics:
   - Training Loss: 0.0089
   - Training MAE: 5.12%
   - Validation Loss: 0.0091
   - Validation MAE: 5.18%

✅ Model saved to: models/checkpoints/gru_model_real.keras
✅ Model size: 1.2 MB
```

---

## Шаг 6: Включите GRU в боте

Откройте `.env` и измените:

```bash
# Было:
GRU_ENABLE=false

# Станет:
GRU_ENABLE=true
GRU_MODEL_PATH=models/checkpoints/gru_model_real.keras
```

Также понизьте порог ML (чтобы ML не блокировала сделки):

```bash
ML_MIN_CONFIDENCE=1.2
ML_COLD_START_CONFIDENCE=1.2
```

---

## Шаг 7: Проверьте модель

```powershell
python -c "from models.gru_predictor import GRUPricePredictor; p = GRUPricePredictor(); p.load('models/checkpoints/gru_model_real.keras'); print('✅ Model loaded successfully')"
```

---

## Шаг 8: Запустите бота

```powershell
python start_bot.py
```

Или с verbose логами:
```powershell
python cli.py live --timeframe 30m --testnet --use-imba --verbose
```

### Проверьте логи бота:

Должны увидеть:
```
✅ [PHASE 2] GRU Predictor initialized (MAPE: 5.12%)
🧠 [GRU] Symbol=BTCUSDT, Predicted=43521.34, Current=43500.00, Change=+0.05%
```

---

## 🆘 Если возникли проблемы

### Проблема 1: TensorFlow не видит GPU после установки

```powershell
# Проверьте переменные окружения
echo $env:PATH | Select-String "CUDA"

# Должны быть пути к CUDA:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\libnvvp

# Если нет, добавьте:
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
```

### Проблема 2: Ошибка "Could not find cuda drivers"

Убедитесь что CUDA Toolkit совпадает с TensorFlow:
- **TensorFlow 2.15-2.16**: CUDA 12.2 или 12.3
- **TensorFlow 2.17+**: CUDA 12.3

Скачайте: https://developer.nvidia.com/cuda-12-3-0-download-archive

### Проблема 3: Ошибка "failed call to cuInit"

```powershell
# Перезагрузите компьютер после установки CUDA
# Или перезапустите драйвер:
nvidia-smi
```

### Проблема 4: GPU Training слишком медленный

```powershell
# Проверьте что TensorFlow использует GPU:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Если показывает [], значит TensorFlow не видит GPU
# Вернитесь к Шагу 2 и переустановите
```

---

## ⚡ Быстрая команда для полной проверки

```powershell
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU Count:', len(gpus)); [print(f'  {i}: {gpu.name}') for i, gpu in enumerate(gpus)]; print('CUDA:', tf.test.is_built_with_cuda())"
```

Ожидается:
```
GPU Count: 1
  0: /physical_device:GPU:0
CUDA: True
```

---

## 🎯 Итого:

1. ✅ CUDA у вас установлено
2. ⚙️ Переустановите TensorFlow: `pip install tensorflow[and-cuda]`
3. ✅ Проверьте GPU: должен показать 1 устройство
4. 🚀 Запустите обучение: `python examples\gru_training_real_data.py`
5. ⏱️ Подождите ~25-30 минут
6. ⚙️ Включите в `.env`: `GRU_ENABLE=true`
7. 🚀 Запустите бота: `python start_bot.py`

**Вперёд! Ваша RTX 5070 Ti готова обучить модель! 🔥**
