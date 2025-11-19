# 🚀 Быстрый старт для Windows

## 1️⃣ Скачать все исправления

```powershell
# Проверить текущую ветку
git branch

# Если НЕ на ветке claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV:
git checkout claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV

# Скачать все изменения
git pull origin claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV

# Проверить что файлы появились
dir test_ml_persistence.py
dir check_ml_status.py
dir INSTALL_WINDOWS.md
```

---

## 2️⃣ Исправить конфликт виртуальных окружений

У вас активированы **ДВА** окружения: `(venv)` и `(ai_trading)`. Нужно оставить одно!

### Вариант A: Использовать ai_trading (рекомендуется, если там уже есть пакеты)

```powershell
# Деактивировать venv
deactivate

# Теперь должно остаться только (ai_trading)
# Проверить Python
python --version
where python

# Установить зависимости в ai_trading
pip install -r requirements_fixed.txt

# Проверить TensorFlow
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"
```

### Вариант B: Использовать venv (чистое окружение)

```powershell
# Деактивировать ai_trading
conda deactivate

# Деактивировать venv тоже
deactivate

# Удалить старый venv
Remove-Item -Recurse -Force venv

# Создать новый чистый venv
python -m venv venv

# Активировать
.\venv\Scripts\Activate.ps1

# Если ошибка execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# Обновить pip
python -m pip install --upgrade pip setuptools wheel

# Установить зависимости
pip install -r requirements_fixed.txt
```

---

## 3️⃣ Проверить установку

```powershell
# Проверить TensorFlow
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"

# Проверить все библиотеки
python -c "import tensorflow, numpy, pandas, sklearn; print('✅ Все библиотеки работают')"

# Тест персистентности
python test_ml_persistence.py
```

---

## 4️⃣ Запустить бота

```powershell
# Проверить ML статус (должен показать "No saved models - starting fresh")
python check_ml_status.py

# Запустить бота на testnet
python cli.py live --timeframe 30m --testnet --use-combo --verbose --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,APTUSDT
```

---

## 🔍 Диагностика проблем

### Проверить что все файлы скачались:

```powershell
# Должны быть все эти файлы:
dir test_ml_persistence.py
dir check_ml_status.py
dir install_local.sh
dir START_TRADING.sh
dir INSTALL_WINDOWS.md
dir ML_FIX_SUMMARY.md
dir EXPECTED_LOGS.md
```

### Проверить текущую ветку и коммиты:

```powershell
git branch
git log --oneline -5
```

Должно показать:
```
* claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV
aefe5df Add comprehensive Windows installation guide for TensorFlow
70ea6b0 Add local installation script with TensorFlow setup
...
```

---

## ⚠️ Важно

### Проблема 1: Файлы не появились после git pull

```powershell
# Проверить удаленные ветки
git fetch --all
git branch -r | Select-String "claude/fix"

# Принудительно переключиться на ветку
git fetch origin
git checkout -B claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV origin/claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV

# Проверить файлы
dir *.py
```

### Проблема 2: "git command not found"

Установить Git для Windows: https://git-scm.com/download/win

### Проблема 3: Два виртуальных окружения одновременно

```powershell
# Деактивировать все
conda deactivate
deactivate

# Закрыть PowerShell и открыть заново

# Активировать ТОЛЬКО ОДНО окружение
.\venv\Scripts\Activate.ps1

# ИЛИ
conda activate ai_trading
```

---

## 📋 Контрольный список

- [ ] Скачаны изменения: `git pull origin claude/fix-lstm-tensorflow-dependency-011HYLKrz2PEqxC6NQowAgKV`
- [ ] Файлы существуют: `dir test_ml_persistence.py`
- [ ] Активировано ОДНО виртуальное окружение
- [ ] TensorFlow установлен: `python -c "import tensorflow; print('OK')"`
- [ ] Тест пройден: `python test_ml_persistence.py`

---

## 🆘 Если ничего не работает

**Запустите эту команду и покажите результат:**

```powershell
Write-Host "=== Git Status ===" -ForegroundColor Cyan
git branch
git log --oneline -3

Write-Host "`n=== Files ===" -ForegroundColor Cyan
dir *.py | Select-Object Name

Write-Host "`n=== Python ===" -ForegroundColor Cyan
python --version
where python

Write-Host "`n=== Activated Environments ===" -ForegroundColor Cyan
$env:VIRTUAL_ENV
$env:CONDA_DEFAULT_ENV
```

Это покажет всю нужную информацию для диагностики!
