# 🪟 Установка на Windows

## ✅ TensorFlow РАБОТАЕТ на Windows!

TensorFlow 2.16+ полностью поддерживает Windows 10/11.

---

## 📋 Требования

### 1. Python версия
TensorFlow 2.16+ требует **Python 3.9 - 3.12**

**Проверить версию:**
```cmd
python --version
```

**Если Python старше или новее:**
- Скачать Python 3.11: https://www.python.org/downloads/
- При установке отметить: ✅ "Add Python to PATH"

---

### 2. Visual C++ Redistributable (обычно уже установлен)

TensorFlow требует Microsoft Visual C++ 2015-2022 Redistributable.

**Скачать если нужно:**
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## 🚀 Установка (Windows)

### Вариант 1: PowerShell (рекомендуется)

```powershell
# 1. Открыть PowerShell (НЕ обязательно от администратора)

# 2. Перейти в папку проекта
cd C:\path\to\crypto_trading_bot_v12

# 3. Создать виртуальное окружение (рекомендуется)
python -m venv venv

# 4. Активировать venv
.\venv\Scripts\Activate.ps1

# Если выдает ошибку "execution policy", выполните:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Потом снова:
.\venv\Scripts\Activate.ps1

# 5. Обновить pip
python -m pip install --upgrade pip setuptools wheel

# 6. Установить зависимости
pip install -r requirements_fixed.txt

# 7. Проверить TensorFlow
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"
```

---

### Вариант 2: CMD (Command Prompt)

```cmd
# 1. Открыть CMD (cmd.exe)

# 2. Перейти в папку проекта
cd C:\path\to\crypto_trading_bot_v12

# 3. Создать виртуальное окружение
python -m venv venv

# 4. Активировать venv
venv\Scripts\activate.bat

# 5. Обновить pip
python -m pip install --upgrade pip setuptools wheel

# 6. Установить зависимости
pip install -r requirements_fixed.txt

# 7. Проверить TensorFlow
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"
```

---

### Вариант 3: Git Bash (если используете)

```bash
# 1. Открыть Git Bash

# 2. Перейти в папку проекта
cd /c/path/to/crypto_trading_bot_v12

# 3. Создать виртуальное окружение
python -m venv venv

# 4. Активировать venv
source venv/Scripts/activate

# 5. Обновить pip
python -m pip install --upgrade pip setuptools wheel

# 6. Установить зависимости
pip install -r requirements_fixed.txt

# 7. Проверить TensorFlow
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"
```

---

## ⚠️ Частые проблемы на Windows

### Проблема 1: "python не является внутренней или внешней командой"

**Решение:**
```cmd
# Использовать полный путь к Python
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements_fixed.txt

# ИЛИ добавить Python в PATH:
# Панель управления → Система → Дополнительные параметры → Переменные среды
# Добавить в PATH: C:\Users\YourName\AppData\Local\Programs\Python\Python311
```

---

### Проблема 2: "cannot be loaded because running scripts is disabled"

**Решение (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Альтернатива:**
Используйте CMD вместо PowerShell:
```cmd
venv\Scripts\activate.bat
```

---

### Проблема 3: "ERROR: Could not find a version that satisfies the requirement tensorflow"

**Причина:** Python слишком старый (< 3.9) или новый (> 3.12)

**Решение:**
```cmd
# Проверить версию
python --version

# Должно быть: Python 3.9.x, 3.10.x, 3.11.x или 3.12.x
# Если нет - установить Python 3.11 с python.org
```

---

### Проблема 4: "DLL load failed" при импорте TensorFlow

**Решение:**
Установить Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

### Проблема 5: Долгая установка / зависание

**Это нормально!** TensorFlow большой (~500 MB), установка может занять 5-15 минут.

```cmd
# Установка с прогрессом
pip install --progress-bar on -r requirements_fixed.txt
```

---

## 🧪 Проверка установки

### Полный тест:

```cmd
# Активировать venv (если еще не активирован)
venv\Scripts\activate.bat

# Запустить тест
python test_ml_persistence.py
```

**Ожидаемый результат:**
```
✅ TensorFlow 2.20.0
✅ NumPy 2.3.5
✅ Pandas 2.3.3
✅ scikit-learn 1.7.2
✅ SUCCESS! Loaded 60 samples (same as saved 60)
✅ Models are persistent across restarts!
```

---

## 🚀 Запуск бота на Windows

### PowerShell:
```powershell
# Активировать venv
.\venv\Scripts\Activate.ps1

# Запустить
python cli.py live --timeframe 30m --testnet --use-combo --verbose --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,APTUSDT
```

### CMD:
```cmd
# Активировать venv
venv\Scripts\activate.bat

# Запустить
python cli.py live --timeframe 30m --testnet --use-combo --verbose --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,APTUSDT
```

### Интерактивный режим:
```cmd
venv\Scripts\activate.bat
python start_bot.py
```

---

## 📊 Мониторинг на Windows

### Проверить ML статус:
```cmd
python check_ml_status.py
```

### Автообновление (PowerShell):
```powershell
while ($true) {
    cls;
    python check_ml_status.py;
    Start-Sleep -Seconds 10
}
```

### Смотреть логи:
```powershell
Get-Content bot.log -Tail 50 -Wait
```

---

## 💾 Backup моделей (Windows)

### Создать backup:
```powershell
$date = Get-Date -Format "yyyyMMdd_HHmmss"
Compress-Archive -Path ml_learning_data -DestinationPath "ml_backup_$date.zip"
```

### ИЛИ в CMD:
```cmd
tar -czf ml_backup_%date:~0,4%%date:~5,2%%date:~8,2%.tar.gz ml_learning_data
```

---

## 🔑 Важные отличия Windows vs Linux

| Операция | Linux/Mac | Windows (CMD) | Windows (PowerShell) |
|----------|-----------|---------------|----------------------|
| Активация venv | `source venv/bin/activate` | `venv\Scripts\activate.bat` | `.\venv\Scripts\Activate.ps1` |
| Деактивация | `deactivate` | `deactivate` | `deactivate` |
| Путь к Python | `/usr/bin/python3` | `C:\Python311\python.exe` | `C:\Python311\python.exe` |
| Разделитель путей | `/` | `\` | `\` (но `/` тоже работает) |
| Переменные окружения | `export VAR=value` | `set VAR=value` | `$env:VAR="value"` |

---

## ✅ Контрольный список

- [ ] Python 3.9-3.12 установлен
- [ ] Python добавлен в PATH
- [ ] Visual C++ Redistributable установлен (обычно уже есть)
- [ ] Виртуальное окружение создано: `python -m venv venv`
- [ ] Venv активирован: `venv\Scripts\activate.bat`
- [ ] pip обновлен: `python -m pip install --upgrade pip`
- [ ] Зависимости установлены: `pip install -r requirements_fixed.txt`
- [ ] TensorFlow работает: `python -c "import tensorflow as tf; print(tf.__version__)"`
- [ ] Тест пройден: `python test_ml_persistence.py`

---

## 🆘 Если ничего не помогает

### Переустановка с нуля:

```cmd
# 1. Удалить старый venv
rmdir /s /q venv

# 2. Очистить pip cache
pip cache purge

# 3. Создать новый venv
python -m venv venv

# 4. Активировать
venv\Scripts\activate.bat

# 5. Установить только TensorFlow сначала
pip install tensorflow==2.20.0

# 6. Проверить
python -c "import tensorflow; print('OK')"

# 7. Если OK - установить остальное
pip install -r requirements_fixed.txt
```

---

## 📞 Диагностика

Если проблемы остаются, соберите информацию:

```cmd
python --version
pip --version
pip list | findstr tensorflow
python -c "import sys; print(sys.executable)"
```

И отправьте вывод этих команд.

---

## 🎯 Итог

**TensorFlow 100% работает на Windows!**

Основные требования:
- ✅ Python 3.9-3.12
- ✅ Visual C++ Redistributable (обычно уже установлен)
- ✅ Виртуальное окружение (рекомендуется)

После установки используйте ваши команды как обычно! 🚀
