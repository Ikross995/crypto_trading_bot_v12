# 📋 Ожидаемые логи ML системы

## 🚀 При запуске бота (первый раз)

```
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | 🧠 [ADVANCED_ML] System initialized
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | 📚 [ML_LOAD] No saved models found - starting from scratch

2025-11-19 10:00:50 | strategy.signals | INFO | ✅ LSTM predictor initialized successfully
2025-11-19 10:00:50 | strategy.enhanced_adaptive_learning | INFO | 🧠 [ENHANCED_ML] Advanced adaptive learning system initialized
```

---

## 🔄 При запуске бота (с сохраненными моделями)

```
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | 🧠 [ADVANCED_ML] System initialized
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | ✅ [ML_LOAD] Loaded model 'pnl_predictor': 150 samples seen
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | ✅ [ML_LOAD] Loaded model 'win_probability': 150 samples seen
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | ✅ [ML_LOAD] Loaded model 'hold_time_predictor': 150 samples seen
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | ✅ [ML_LOAD] Loaded model 'risk_estimator': 150 samples seen
2025-11-19 10:00:45 | strategy.ml_learning_system | INFO | 🧠 [ML_LOAD] Successfully loaded 4/4 ML models

2025-11-19 10:00:50 | strategy.signals | INFO | ✅ LSTM predictor initialized successfully
```

---

## 📚 Во время торговли (COLD START фаза)

```
2025-11-19 10:01:40 | strategy.enhanced_adaptive_learning | INFO | 🧠 [COLD_START] Learning mode: 15/50 samples - ML не блокирует
2025-11-19 10:01:40 | strategy.enhanced_adaptive_learning | INFO | 📚 [LEARNING_MODE] Пропускаем сигнал 0.63 - ML учится на реальных данных
2025-11-19 10:01:40 | strategy.ml_learning_system | INFO | 🎯 [ML_PREDICTION] Expected: +0.00% PnL, 10% win prob, 0.10 confidence

# После закрытия сделки:
2025-11-19 10:15:30 | strategy.enhanced_adaptive_learning | INFO | 🧠 [ML_LEARNING] Learned from ETHUSDT: +0.75% PnL in 13.8 min
2025-11-19 10:15:30 | strategy.enhanced_adaptive_learning | INFO | 📚 [ML_SAMPLES] 16/50 samples collected for ML training
```

---

## 🎓 LEARNING фаза (50-200 samples)

```
2025-11-19 11:30:00 | strategy.enhanced_adaptive_learning | INFO | 🎓 [LEARNING] Learning mode: 120/200 samples, progress: 46.7%
2025-11-19 11:30:00 | strategy.enhanced_adaptive_learning | INFO | 🎓 [LEARNING] Signal: 1.35>=1.31? True, ML weight: 0.14, Decision: TRADE
2025-11-19 11:30:00 | strategy.ml_learning_system | INFO | 🎯 [ML_PREDICTION] Expected: +0.45% PnL, 62% win prob, 0.24 confidence

# После сделки:
2025-11-19 11:45:00 | strategy.enhanced_adaptive_learning | INFO | 🧠 [ML_LEARNING] Learned from BTCUSDT: +0.52% PnL in 15.0 min
2025-11-19 11:45:00 | strategy.enhanced_adaptive_learning | INFO | 📚 [ML_SAMPLES] 121/200 samples collected for ML training
```

---

## 🧠 FULL ML фаза (200+ samples)

```
2025-11-19 14:20:00 | strategy.enhanced_adaptive_learning | DEBUG | 🧠 [FULL_ML] Full ML mode: 250 samples
2025-11-19 14:20:00 | strategy.ml_learning_system | INFO | 🎯 [ML_PREDICTION] Expected: +0.85% PnL, 72% win prob, 0.45 confidence
2025-11-19 14:20:00 | strategy.enhanced_adaptive_learning | INFO | 🎯 [ENHANCED_ANALYSIS] BTCUSDT: Expected +0.85% PnL, 72% win prob, Confidence: 0.45

# ML теперь активно фильтрует:
2025-11-19 14:25:00 | strategy.enhanced_adaptive_learning | DEBUG | 🧠 [FULL_ML] Signal 1.15 rejected: ML confidence too low (0.25 < 0.40)
```

---

## 💾 Автосохранение (каждые 10 сделок)

```
2025-11-19 12:00:00 | strategy.enhanced_adaptive_learning | INFO | 💾 [AUTO_SAVE] Saving ML models at 130 samples...
2025-11-19 12:00:00 | strategy.ml_learning_system | INFO | 💾 [ML_SAVE] Saved 4 ML models with metadata
2025-11-19 12:00:00 | strategy.ml_learning_system | INFO | 💾 [ML_SAVE] Saved ML data: 130 contexts, 130 outcomes
2025-11-19 12:00:00 | strategy.enhanced_adaptive_learning | INFO | ✅ [AUTO_SAVE] ML models saved successfully
```

---

## 🛑 При остановке бота (Ctrl+C)

```
2025-11-19 16:00:00 | runner.live | INFO | 🛑 [SHUTDOWN] Stopping trading engine...
2025-11-19 16:00:00 | runner.live | INFO | 💾 [SHUTDOWN] Saving dashboard history...
2025-11-19 16:00:00 | runner.live | INFO | ✅ [SHUTDOWN] Dashboard saved

2025-11-19 16:00:00 | runner.live | INFO | 💾 [SHUTDOWN] Saving ML models and learning data...
2025-11-19 16:00:00 | strategy.ml_learning_system | INFO | 💾 [ML_SAVE] Saved 4 ML models with metadata
2025-11-19 16:00:00 | strategy.ml_learning_system | INFO | 💾 [ML_SAVE] Saved ML data: 185 contexts, 185 outcomes
2025-11-19 16:00:00 | runner.live | INFO | ✅ [SHUTDOWN] ML models saved (185 samples trained)

2025-11-19 16:00:00 | runner.live | INFO | 💾 [SHUTDOWN] Active positions tracked: 0
2025-11-19 16:00:00 | runner.live | INFO | ✅ [SHUTDOWN] Trading engine stopped cleanly
```

---

## 🔍 Прогресс обучения (примерный timeline)

### День 1 (0-50 samples)
```
10:00 - Bot started, 0 samples
10:30 - First trade learned, 1 sample
12:00 - 10 samples, AUTO_SAVE
14:00 - 20 samples, AUTO_SAVE
16:00 - 30 samples, AUTO_SAVE
18:00 - 40 samples, AUTO_SAVE
20:00 - 50 samples, AUTO_SAVE → Переход в LEARNING фазу
```

### День 2-3 (50-200 samples)
```
ML постепенно увеличивает влияние на решения
Progress: 50 → 100 → 150 → 200 samples
```

### День 4+ (200+ samples)
```
ML полностью активна
Continuous improvement с каждой сделкой
```

---

## ⚠️ Возможные предупреждения (нормальные)

```
2025-11-19 10:00:50 | strategy.signals | WARNING | LSTM predictor initialization failed: No trained model found
```
**Решение**: Это нормально при первом запуске. Запустите обучение командой из документации.

```
2025-11-19 10:01:40 | strategy.enhanced_adaptive_learning | WARNING | ⚠️ [AUTO_SAVE] Failed to save models: [Errno 28] No space left
```
**Решение**: Очистите место на диске или удалите старые `ml_learning_data/`.

---

## ✅ Успешные индикаторы

- ✅ `Successfully loaded 4/4 ML models` - модели загрузились
- ✅ `AUTO_SAVE ML models saved successfully` - автосохранение работает
- ✅ `ML models saved (X samples trained)` - прогресс сохраняется при остановке
- ✅ `Phase: FULL ML` - система полностью обучена

---

## 📊 Как отследить прогресс

### Вариант 1: Логи
```bash
tail -f bot.log | grep "ML_SAMPLES"
```

### Вариант 2: Status скрипт
```bash
python3 check_ml_status.py
```

### Вариант 3: Файлы
```bash
cat ml_learning_data/pnl_predictor_metadata.json
```
