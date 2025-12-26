# 🚀 Локальное обучение GRU модели - Пошаговая инструкция

## 📋 Системные требования

### Минимальные:
- **Python:** 3.8+
- **RAM:** 8 GB
- **Место на диске:** 5 GB свободного
- **Интернет:** для загрузки данных с Binance

### Рекомендуемые (для быстрого обучения):
- **GPU:** NVIDIA GTX 1060+ (6GB+ VRAM)
- **RAM:** 16 GB
- **Место на диске:** 10 GB

---

## 🔧 Шаг 1: Установка зависимостей

### Windows (с GPU NVIDIA):

```powershell
# 1. Установите PyTorch с CUDA поддержкой
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Установите остальные зависимости
pip install scikit-learn pandas numpy aiohttp

# 3. Проверьте GPU
python check_gpu.py
```

### Windows (без GPU / CPU only):

```powershell
# 1. Установите PyTorch CPU версию (быстрее установка)
pip install torch torchvision

# 2. Установите остальные зависимости
pip install scikit-learn pandas numpy aiohttp
```

### Linux / Mac:

```bash
# С GPU (NVIDIA):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Без GPU (CPU only):
pip3 install torch torchvision

# Остальные зависимости:
pip3 install scikit-learn pandas numpy aiohttp
```

---

## ✅ Шаг 2: Проверка установки

Запустите проверку GPU (если есть):

```bash
python check_gpu.py
```

**Ожидаемый вывод (с GPU):**
```
✅ GPU available: NVIDIA GeForce RTX 5070 Ti
   GPU Memory: 16.0 GB
   CUDA Version: 12.1
```

**Ожидаемый вывод (CPU only):**
```
📊 Using CPU
   Training will be slower, but works fine!
```

---

## 🎯 Шаг 3: Запуск обучения

### Вариант 1: Улучшенная версия (РЕКОМЕНДУЕТСЯ) ⭐

Это **лучшая версия** с исправлением всех багов ML!

```bash
# С настройками по умолчанию (365 дней, 50 эпох):
python examples/gru_training_improved.py

# С кастомными параметрами (180 дней, 30м таймфрейм):
python examples/gru_training_improved.py --days 180 --interval 30m --epochs 50 --batch-size 128
```

**Параметры:**
- `--days 180` - Использовать 180 дней данных (полгода)
- `--interval 30m` - Таймфрейм 30 минут
- `--epochs 50` - Максимум 50 эпох (остановится раньше с early stopping)
- `--batch-size 128` - Размер батча (зависит от GPU)

**Оптимальные batch sizes:**
- **RTX 5070 Ti (16GB):** `--batch-size 256` ⚡
- **RTX 4090 (24GB):** `--batch-size 512` ⚡⚡
- **RTX 3080 (10GB):** `--batch-size 128`
- **GTX 1080 (8GB):** `--batch-size 64`
- **CPU only:** `--batch-size 32`

---

### Вариант 2: Percentage-based версия

Использует % изменения вместо абсолютных цен (универсально для всех монет):

```bash
python examples/gru_training_pytorch_v2_percentage.py --days 180 --interval 30m --epochs 30 --batch-size 1024
```

---

### Вариант 3: Финальная версия (комбинированная)

```bash
python train_gru_final.py --days 180 --epochs 30 --batch-size 1024
```

---

## ⏱️ Ожидаемое время обучения

### С GPU:
| GPU | Batch Size | 180 дней, 50 эпох | 365 дней, 50 эпох |
|-----|------------|-------------------|-------------------|
| **RTX 5070 Ti (16GB)** | 256 | ~30-40 мин | ~1-2 часа |
| **RTX 4090 (24GB)** | 512 | ~20-30 мин | ~40-60 мин |
| **RTX 3080 (10GB)** | 128 | ~1-1.5 часа | ~2-3 часа |
| **GTX 1080 (8GB)** | 64 | ~2-3 часа | ~4-6 часов |

### С CPU:
| CPU | 180 дней | 365 дней |
|-----|----------|----------|
| **i7/i9 (8+ cores)** | ~4-6 часов | ~8-12 часов |
| **i5 (4-6 cores)** | ~8-10 часов | ~16-20 часов |

---

## 📊 Мониторинг процесса обучения

Во время обучения вы увидите:

```
================================================================================
🚀 IMPROVED GRU Model Training (NO BUGS!)
================================================================================
📋 Configuration:
   Symbols: BTCUSDT, ETHUSDT, BNBUSDT, ...
   Days: 180 (last 180 days of FRESH data)
   Sequence: 60
   Epochs: 50 (max, with early stopping)
   Batch size: 128
================================================================================

🎮 Configuring GPU...
✅ GPU: NVIDIA GeForce RTX 5070 Ti (16.0 GB)
   CUDA: 12.1

📥 Downloading BTCUSDT (1/10)...
   ✅ BTCUSDT: 8,640 candles

📦 Preparing sequences (NO LEAKAGE!)...
   Sequence length: 60
   Train: 70%, Val: 15%, Test: 15%

✅ Temporal split:
   Train: 6,048 samples
   Val:   1,296 samples
   Test:  1,296 samples

🧠 Building IMPROVED GRU model...
✅ Model parameters: 150,284

🎯 Training IMPROVED model...
   Epochs: 50 (max)
   Initial LR: 0.001
   Early stopping patience: 7

Epoch   1/50 | Train: 0.001234 | Val: 0.001456 | LR: 0.001000 | Time: 12.3s
Epoch   2/50 | Train: 0.001089 | Val: 0.001234 | LR: 0.001000 | Time: 24.8s
   💾 New best model! Val Loss: 0.001234
Epoch   3/50 | Train: 0.000987 | Val: 0.001123 | LR: 0.001000 | Time: 37.2s
   💾 New best model! Val Loss: 0.001123
...
   ⚠️  EarlyStopping counter: 1/7
...
✅ Early stopping at epoch 28
   Best validation loss: 0.000856

================================================================================
📊 Final Evaluation on Test Set
================================================================================
📊 Test Metrics (Real Prices):
   MSE:  125.45
   MAE:  $8.32
   MAPE: 0.15%

📊 Win Rate Analysis:
   Overall: 57.23% (742/1296)
   Significant moves (>0.0001): 58.91%

✅ Model saved: models/checkpoints/gru_improved.pt
   Size: 2.3 MB

================================================================================
🎉 IMPROVED TRAINING COMPLETED!
================================================================================
```

---

## 🎯 После обучения

Модель сохранится в `models/checkpoints/gru_improved.pt`

### Обновите .env файл:

```env
# Включите GRU модель
GRU_ENABLE=true

# Укажите путь к модели
GRU_MODEL_PATH=models/checkpoints/gru_improved.pt
```

### Запустите бота:

```bash
python start_bot.py
```

или

```bash
python cli.py live --timeframe 30m --use-imba
```

---

## 🐛 Troubleshooting

### Ошибка: "CUDA out of memory"

**Решение:** Уменьшите batch size:
```bash
python examples/gru_training_improved.py --batch-size 64
# Или ещё меньше: --batch-size 32
```

---

### Ошибка: "No module named 'torch'"

**Решение:** Установите PyTorch:
```bash
# С GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Без GPU:
pip install torch torchvision
```

---

### Ошибка: "Failed to download data from Binance"

**Причины:**
1. Нет интернета
2. Binance API недоступен в вашей стране
3. Rate limit (слишком много запросов)

**Решение:**
1. Проверьте интернет
2. Используйте VPN (если Binance заблокирован)
3. Подождите 1 минуту и попробуйте снова

---

### Обучение слишком долгое на CPU

**Решение:** Уменьшите объём данных:
```bash
python examples/gru_training_improved.py --days 90 --epochs 20 --batch-size 32
```

---

### Ошибка: "No GPU found, using CPU"

Это **не ошибка**, просто PyTorch не видит GPU.

**Проверьте:**
1. У вас NVIDIA GPU? (AMD не поддерживается PyTorch CUDA)
2. Установлены драйверы NVIDIA?
3. Установили PyTorch с CUDA?

**Установка PyTorch с CUDA:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📈 Сравнение версий обучения

| Версия | Файл | Преимущества | Недостатки |
|--------|------|--------------|------------|
| **Improved** ⭐ | `examples/gru_training_improved.py` | ✅ NO data leakage<br>✅ Temporal split<br>✅ Early stopping<br>✅ LR scheduler<br>✅ RobustScaler | Предсказывает абсолютную цену |
| **Percentage** | `examples/gru_training_pytorch_v2_percentage.py` | ✅ Универсально для всех монет<br>✅ Большая архитектура (400K params) | ❌ Есть data leakage<br>❌ Нет early stopping |
| **Final** | `train_gru_final.py` | ✅ Комбинирует percentage + enhanced | ❌ Есть data leakage |

**Рекомендация:** Используйте **Improved** версию для production!

---

## 💡 Советы по оптимизации

### 1. Быстрое тестирование:
```bash
# Маленький датасет, быстрое обучение:
python examples/gru_training_improved.py --days 30 --epochs 10 --batch-size 64
```

### 2. Полное обучение (максимальное качество):
```bash
# Год данных, 50 эпох, большой batch:
python examples/gru_training_improved.py --days 365 --epochs 50 --batch-size 256
```

### 3. Для слабого GPU (8GB):
```bash
# Небольшой batch, меньше данных:
python examples/gru_training_improved.py --days 180 --epochs 30 --batch-size 64
```

### 4. Для мощного GPU (24GB+):
```bash
# Максимальная скорость:
python examples/gru_training_improved.py --days 365 --epochs 50 --batch-size 512
```

---

## 🎓 Что дальше?

После успешного обучения модели:

1. ✅ **Обновите .env** - включите GRU_ENABLE=true
2. ✅ **Запустите бота** - `python start_bot.py`
3. ✅ **Мониторьте результаты** - смотрите логи и метрики
4. ✅ **Переобучайте периодически** - каждые 2-4 недели со свежими данными

---

## 📚 Дополнительная информация

- **Документация PyTorch:** https://pytorch.org/docs/stable/index.html
- **Документация Binance API:** https://binance-docs.github.io/apidocs/
- **Проблемы с обучением?** Создайте issue в GitHub

---

## ✨ Ключевые улучшения в `gru_training_improved.py`

1. **NO data leakage** - scaler fit только на train данных
2. **Temporal split** - train/val/test по времени (70/15/15)
3. **shuffle=False** - сохраняет временной порядок
4. **Early stopping** - останавливается при plateau (patience=7)
5. **Learning rate scheduler** - уменьшает LR автоматически
6. **Gradient clipping** - предотвращает exploding gradients
7. **RobustScaler** - устойчив к выбросам
8. **AdamW optimizer** - лучшая регуляризация
9. **Dropout 0.4** - предотвращает overfitting
10. **Batch Normalization** - стабильное обучение

---

🎉 **Удачи с обучением модели!** 🚀
