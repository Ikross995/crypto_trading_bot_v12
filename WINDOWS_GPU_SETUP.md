# 🎮 Настройка GPU для обучения GRU на Windows

## Ваше оборудование
- ✅ NVIDIA GeForce RTX 5070 Ti Laptop GPU
- ✅ Driver Version: 32.0.15.8097

## Проблема
TensorFlow не видит вашу GPU из-за отсутствия CUDA драйверов.

---

## 🚀 Решение 1: TensorFlow с DirectML (БЫСТРО, РЕКОМЕНДУЕТСЯ)

DirectML работает с любыми GPU на Windows (AMD и NVIDIA) без установки CUDA.

### Установка (5 минут):

```powershell
# 1. Удалите старый TensorFlow
pip uninstall tensorflow tensorflow-gpu -y

# 2. Установите TensorFlow с DirectML
pip install tensorflow-directml

# 3. Проверьте GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### Запуск обучения:
```powershell
cd C:\Users\User\crypto_trading_bot_v12
python examples\gru_training_real_data.py
```

---

## 🔧 Решение 2: TensorFlow с CUDA (СЛОЖНЕЕ, БЫСТРЕЕ)

Для максимальной производительности на NVIDIA GPU.

### Требования:
1. **CUDA Toolkit 12.x** - https://developer.nvidia.com/cuda-downloads
2. **cuDNN 8.9.x** - https://developer.nvidia.com/cudnn

### Установка (15-20 минут):

#### Шаг 1: Установите CUDA Toolkit
```powershell
# Скачайте с:
https://developer.nvidia.com/cuda-12-3-0-download-archive

# Выберите:
# - Windows
# - x86_64
# - 11
# - exe (network)

# Запустите установщик
```

#### Шаг 2: Установите cuDNN
```powershell
# 1. Скачайте cuDNN 8.9 для CUDA 12.x
https://developer.nvidia.com/rdp/cudnn-download

# 2. Распакуйте в папку CUDA:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\

# 3. Добавьте в PATH:
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
```

#### Шаг 3: Установите TensorFlow
```powershell
pip uninstall tensorflow tensorflow-gpu -y
pip install tensorflow[and-cuda]
```

#### Шаг 4: Проверьте
```powershell
python -c "import tensorflow as tf; print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 📊 Решение 3: Обучение на CPU (ЗАПАСНОЙ ВАРИАНТ)

Если GPU не работает, можно обучить на CPU с уменьшенным датасетом.

### Изменения в скрипте:
Откройте `examples/gru_training_real_data.py` и измените:

```python
# Было:
days = 365  # 1 год данных

# Станет:
days = 180  # 6 месяцев данных (в 2 раза быстрее)
```

### Время обучения:
- **GPU (RTX 5070 Ti)**: ~15-20 минут
- **CPU (6 месяцев)**: ~25-30 минут
- **CPU (1 год)**: ~45-60 минут

---

## ✅ Проверка после установки

### Тест 1: Проверьте TensorFlow
```powershell
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Ожидается:
```
TensorFlow: 2.20.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Тест 2: Простой расчёт на GPU
```powershell
python -c "import tensorflow as tf; import time; x = tf.random.normal([1000, 1000]); start = time.time(); y = tf.matmul(x, x); print('Time:', time.time()-start, 'sec'); print('Device:', y.device)"
```

Ожидается:
```
Time: 0.01 sec
Device: /job:localhost/replica:0/task:0/device:GPU:0
```

---

## 🚀 Запуск обучения

После настройки GPU:

```powershell
# 1. Перейдите в папку проекта
cd C:\Users\User\crypto_trading_bot_v12

# 2. Запустите обучение
python examples\gru_training_real_data.py
```

### Что произойдёт:
1. **Загрузка данных** (5-10 мин): 10 пар × ~525,600 свечей = 5.25M данных
2. **Расчёт индикаторов** (2-3 мин): RSI, MACD, BB, SMA, EMA, ATR
3. **Обучение** (15-20 мин на GPU): 20 эпох, 99,601 параметр
4. **Сохранение**: `models/checkpoints/gru_model_real.keras`

### Логи обучения:
```
Epoch 1/20
 52123/52123 [======] - 45s 862us/step - loss: 0.0234 - mae: 0.0891
Epoch 2/20
 52123/52123 [======] - 43s 825us/step - loss: 0.0156 - mae: 0.0723
...
Epoch 20/20
 52123/52123 [======] - 42s 806us/step - loss: 0.0089 - mae: 0.0512
```

---

## 🔥 После обучения

### 1. Включите GRU в боте
Отредактируйте `.env`:
```bash
GRU_ENABLE=true
GRU_MODEL_PATH=models/checkpoints/gru_model_real.keras
```

### 2. Проверьте модель
```powershell
python -c "from models.gru_predictor import GRUPricePredictor; p = GRUPricePredictor(); p.load('models/checkpoints/gru_model_real.keras'); print('Model loaded OK')"
```

### 3. Запустите бота
```powershell
python start_bot.py
```

Или:
```powershell
python cli.py live --timeframe 30m --testnet --use-imba --verbose
```

---

## 🆘 Если что-то не работает

### DirectML не видит GPU:
```powershell
# Обновите драйвера через Device Manager или GeForce Experience
```

### CUDA ошибки:
```powershell
# Убедитесь что установлена правильная версия:
nvcc --version  # Должна быть 12.x
```

### Медленно качает данные:
```
# Это нормально - 10 пар × 365 дней = большой объём
# Binance rate limit: 2400 requests/min
# Ожидайте 5-10 минут для загрузки
```

---

## 💡 Рекомендация

**Для быстрого старта**: Используйте **DirectML** (Решение 1)
- Работает из коробки
- Не требует CUDA
- Достаточно быстро для обучения
- GPU будет использоваться автоматически

**Для максимальной скорости**: Используйте **CUDA** (Решение 2)
- На 20-30% быстрее DirectML
- Но требует установки CUDA Toolkit и cuDNN

---

## 📞 Поддержка

Если возникнут проблемы, пришлите вывод:

```powershell
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU')); print('CUDA:', tf.test.is_built_with_cuda())"
```
