# 🛡️ Автоматическая очистка несовместимых ML моделей

## 🎯 Проблема которую решает этот фикс

### Что произошло у вас:
1. ✅ Бот работал 42 часа - все было ок
2. 🔄 Вы сделали `git pull` - скачали новый код с clipping
3. ❌ Бот загрузил **СТАРЫЕ** .pkl модели (без clipping)
4. 💥 Предсказания снова абсурдные: `+10318233232.94%`

### Почему так произошло:
- Старые модели обучены **БЕЗ** clipping (±бесконечность)
- Новый код **С** clipping (±20% max)
- Модели несовместимы, но бот не знал об этом
- Загрузил старые модели → WARM START добавил данные → предсказания сломаны

---

## ✅ Автоматическое решение (уже работает!)

### Commit: `4a6a2c1` - Add ML model versioning and automatic validation

Теперь бот **АВТОМАТИЧЕСКИ**:

### 1. Проверяет версию модели
```python
MODEL_VERSION = 2

# v1: Старые модели без clipping
# v2: Новые модели с clipping (±20% PnL, 0-24h hold_time, 0-10% risk)
```

### 2. Валидирует предсказания при загрузке
```python
def _validate_model_sanity():
    """Проверяет что модель не предсказывает абсурд"""

    # Тестовый прогон с нормальными данными
    test_prediction = model.predict(normal_features)

    # Проверка диапазонов:
    ✅ pnl_predictor: должен быть ±50% max (не миллиарды!)
    ✅ win_probability: должен быть 0-1
    ✅ hold_time_predictor: должен быть 0-48 часов
    ✅ risk_estimator: должен быть 0-20%

    # Если модель предсказывает абсурд → ОТКЛОНИТЬ
```

### 3. Удаляет несовместимые модели
```python
# Если модель:
# - Не той версии (v1 вместо v2)
# - Или предсказывает абсурдные значения
#
# ТО → автоматически удалить .pkl файлы
# И → начать обучение с нуля
```

---

## 🚀 Что нужно сделать вам

### Вариант A: Автоматическая очистка (РЕКОМЕНДУЕТСЯ)

Просто скачайте обновление и перезапустите:

```powershell
# 1. Скачать новый код с автоматической валидацией
git pull origin claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV

# 2. Перезапустить бота
python cli.py live --timeframe 30m --testnet --use-combo --verbose --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,APTUSDT
```

**Что произойдет:**
```
🧠 [ML_LOAD] Loading model 'pnl_predictor'...
⚠️ [ML_LOAD] Model 'pnl_predictor' version mismatch: saved v1, current v2
⚠️ [ML_LOAD] Model 'win_probability' version mismatch: saved v1, current v2
⚠️ [ML_LOAD] Model 'hold_time_predictor' version mismatch: saved v1, current v2
⚠️ [ML_LOAD] Model 'risk_estimator' version mismatch: saved v1, current v2

🗑️ [ML_CLEANUP] Deleting 4 incompatible models: ['pnl_predictor', 'win_probability', 'hold_time_predictor', 'risk_estimator']

📚 [ML_LOAD] No saved models found - starting from scratch
🧠 [COLD_START] Learning mode: 0/50 samples
```

---

### Вариант B: Ручная очистка (если хотите убедиться)

```powershell
# Удалить все старые модели вручную
Remove-Item -Recurse -Force ml_learning_data\* -ErrorAction SilentlyContinue

# Проверить что папка пустая
Get-ChildItem ml_learning_data

# Запустить бота
python cli.py live --timeframe 30m --testnet --use-combo --verbose --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,APTUSDT
```

---

## 📊 Ожидаемые логи после очистки

### ✅ ХОРОШО (нормальные предсказания):
```
📚 [ML_LOAD] No saved models found - starting from scratch
🧠 [COLD_START] Learning mode: 0/50 samples

📉 Expected PnL: +0.0%  ← НОРМАЛЬНО! Модель еще не обучена
📊 Win Probability: 50.0%
⏱️ Hold Time: 0.5 hours
⚠️ Risk: 0.0%

# После 10 сделок:
📉 Expected PnL: +1.2%  ← НОРМАЛЬНО!
📊 Win Probability: 65.0%
⏱️ Hold Time: 2.3 hours
⚠️ Risk: 1.8%
```

### ❌ ПЛОХО (старые модели не удалились):
```
✅ [ML_LOAD] Loaded model 'pnl_predictor': 268 samples seen

📉 Expected PnL: +10318233232.94%  ← АБСУРД! Старые модели!
```

Если видите **АБСУРД** → что-то пошло не так, используйте **Вариант B** (ручная очистка).

---

## 🔧 Как работает защита

### При каждом запуске бота:

```
1. Загрузить .pkl файлы
   ↓
2. Прочитать metadata.json
   ↓
3. Проверить model_version
   ↓
4. Если версия не совпадает → ОТКЛОНИТЬ
   ↓
5. Если версия ок → сделать тестовый прогноз
   ↓
6. Проверить что прогноз вменяемый
   ↓
7. Если прогноз абсурдный → ОТКЛОНИТЬ
   ↓
8. Удалить отклоненные модели
   ↓
9. Начать обучение с нуля (с правильным clipping)
```

---

## 🎯 Будущие обновления

Теперь при любых изменениях в логике обучения я буду:

1. Увеличивать `MODEL_VERSION` в коде
2. Старые модели автоматически удалятся
3. Обучение начнется с нуля с новой логикой

**Пример:**
```python
# Если добавлю separate models per symbol:
MODEL_VERSION = 3  # v3: Per-symbol models

# Старые v2 модели (общие для всех символов) → auto-delete
# Новые v3 модели (отдельные BTC, ETH, etc) → start fresh
```

---

## 🆘 Если проблемы остались

### 1. Проверить что новый код скачался:
```powershell
git log --oneline -1
# Должно быть: 4a6a2c1 Add ML model versioning and automatic validation
```

### 2. Проверить что MODEL_VERSION = 2:
```powershell
Select-String -Path strategy/ml_learning_system.py -Pattern "MODEL_VERSION"
# Должно показать: MODEL_VERSION = 2
```

### 3. Удалить модели вручную:
```powershell
Remove-Item -Recurse -Force ml_learning_data\*
```

### 4. Перезапустить бота и показать первые 50 строк лога:
```powershell
python cli.py live --timeframe 30m --testnet --use-combo --verbose --symbols BTCUSDT,ETHUSDT 2>&1 | Select-Object -First 50
```

---

## ✅ Итог

**Раньше:** Каждый раз после `git pull` нужно было **ВРУЧНУЮ** удалять старые модели

**Теперь:** Бот **АВТОМАТИЧЕСКИ** обнаружит несовместимые модели и удалит их

**Действия:**
1. `git pull` (скачать новый код)
2. Перезапустить бота
3. Все! 🚀

Система сама обнаружит версию v1, увидит что нужна v2, удалит старые модели и начнет обучение с нуля.
