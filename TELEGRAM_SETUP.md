# 🤖 Telegram Bot Setup - Dashboard Integration

## Быстрый старт (3 минуты)

### Шаг 1: Создай Telegram бота

1. Открой Telegram и найди **@BotFather**
2. Отправь команду `/newbot`
3. Придумай имя бота (например: `My Trading Bot`)
4. Придумай username (должен заканчиваться на `bot`, например: `my_trading_bot`)
5. **Скопируй токен** который даст BotFather (примерно так: `6754312890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw`)

### Шаг 2: Получи Chat ID группы

**Вариант 1: Через @userinfobot (самый простой)**
1. Добавь своего бота в группу
2. Добавь [@userinfobot](https://t.me/userinfobot) в ту же группу
3. @userinfobot покажет Chat ID группы (например: `-1001234567890`)
4. Удали @userinfobot из группы

**Вариант 2: Через @getidsbot**
1. Перешли любое сообщение из группы боту [@getidsbot](https://t.me/getidsbot)
2. Он покажет Chat ID группы

**Вариант 3: Через API (если первые 2 не работают)**
1. Добавь своего бота в группу
2. Напиши в группе любое сообщение
3. Открой в браузере: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Найди `"chat":{"id":-1001234567890}` - это твой Chat ID

### Шаг 3: Настрой конфигурацию

Открой файл `.env` или создай его в корне проекта:

```bash
# Telegram Settings
TG_BOT_TOKEN=6754312890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw
TG_CHAT_ID=-1001234567890
TG_DASHBOARD_ENABLED=true
TG_DASHBOARD_INTERVAL=300  # Обновлять каждые 5 минут (300 секунд)
```

### Шаг 4: Протестируй бота

Запусти тест скрипт:

```bash
python infra/telegram_bot.py
```

Если видишь `✅ Bot connected: @my_trading_bot` - **всё работает!**

---

## 📊 Что будет отправляться в группу

Бот будет отправлять красивые обновления дашборда каждые 5 минут (настраивается):

```
🚀 Trading Dashboard Update
2025-11-12 20:30:45 UTC

━━━━━━━━━━━━━━━━━━━━
💰 ACCOUNT BALANCE
━━━━━━━━━━━━━━━━━━━━

Balance: $1,075.50
Equity: $1,100.25
Total P&L: 💰 +$75.50 (+7.55%)
Hourly P&L: +$30.20/hr

━━━━━━━━━━━━━━━━━━━━
📊 TRADING STATS
━━━━━━━━━━━━━━━━━━━━

Total Trades: 25
Win Rate: 68.0% (17W/8L)
Profit Factor: 2.5x
Sharpe Ratio: 1.85

Best Trade: 💎 $45.20
Worst Trade: 💔 -$12.50

🔥 Win Streak: 5
Best Win Streak: 🏆 8
Worst Loss Streak: 💀 3

━━━━━━━━━━━━━━━━━━━━
⚠️ RISK METRICS
━━━━━━━━━━━━━━━━━━━━

Risk Score: 🟢 25/100
Margin Used: $43.00 (4.0%)
Free Margin: $1,032.50

━━━━━━━━━━━━━━━━━━━━
📈 OPEN POSITIONS (2)
━━━━━━━━━━━━━━━━━━━━

BTCUSDT 🟢 10x
Entry: $40,250 → $40,780
P&L: 💚 +$12.50 (+3.2%)
Margin: $20.39

ETHUSDT 🟢 5x
Entry: $2,180 → $2,245
P&L: 💚 +$8.20 (+2.5%)
Margin: $22.45

━━━━━━━━━━━━━━━━━━━━
```

---

## 🔧 Настройки и параметры

### В `.env` файле:

```bash
# Токен бота от @BotFather
TG_BOT_TOKEN=your_bot_token_here

# ID группы/канала (начинается с минуса для групп)
TG_CHAT_ID=-1001234567890

# Включить отправку обновлений (true/false)
TG_DASHBOARD_ENABLED=true

# Интервал обновлений в секундах
# 60 = каждую минуту
# 300 = каждые 5 минут (рекомендуется)
# 600 = каждые 10 минут
# 1800 = каждые 30 минут
# 3600 = каждый час
TG_DASHBOARD_INTERVAL=300
```

### Или через CLI аргументы:

```bash
python start_bot.py --tg-enabled --tg-interval 300
```

---

## 🚀 Запуск с Telegram интеграцией

### Вариант 1: Через .env (рекомендуется)

1. Настрой `.env` файл (см. выше)
2. Запусти бота как обычно:
   ```bash
   python start_bot.py
   ```

### Вариант 2: Через аргументы командной строки

```bash
python start_bot.py \
  --tg-bot-token "YOUR_TOKEN" \
  --tg-chat-id "-1001234567890" \
  --tg-enabled \
  --tg-interval 300
```

---

## 🎯 Примеры использования

### Обновления каждые 5 минут (по умолчанию)
```bash
TG_DASHBOARD_INTERVAL=300
```

### Обновления каждый час (для менее активной торговли)
```bash
TG_DASHBOARD_INTERVAL=3600
```

### Обновления каждую минуту (для активной торговли)
```bash
TG_DASHBOARD_INTERVAL=60
```

### Отключить Telegram (только локальный дашборд)
```bash
TG_DASHBOARD_ENABLED=false
```

---

## 📱 Что можно отправлять

Telegram бот поддерживает:

### 1. Текстовые обновления (по умолчанию)
- Красиво форматированные сообщения с HTML
- Эмодзи для визуализации
- Автообновление каждые N минут

### 2. HTML файл дашборда
```python
await bot.send_document(
    Path("data/learning_reports/enhanced_dashboard.html"),
    caption="📊 Full Dashboard Report"
)
```

### 3. Кастомные сообщения
```python
await bot.send_message("🚀 Bot started successfully!")
```

---

## 🔐 Безопасность

⚠️ **ВАЖНО**: НЕ комитьте `.env` файл в Git!

Добавь в `.gitignore`:
```
.env
.env.local
.env.*.local
```

Храни токены в безопасности:
- Используй `.env` для локальной разработки
- Используй переменные окружения на сервере
- Не шарь токены в публичных местах

---

## 🐛 Troubleshooting

### Бот не отправляет сообщения

**Проблема**: "Failed to send message: Forbidden: bot was kicked from the group"
**Решение**:
1. Убедись что бот добавлен в группу
2. Дай боту права администратора (или права на отправку сообщений)

**Проблема**: "Failed to send message: Bad Request: chat not found"
**Решение**:
1. Проверь Chat ID - должен начинаться с `-100` для групп
2. Убедись что бот был добавлен в группу

### Неправильный Chat ID

**Для групп**: Chat ID начинается с `-100` (например: `-1001234567890`)
**Для личных чатов**: Chat ID - положительное число (например: `123456789`)

### Бот не видит обновления

**Проблема**: Bot token неправильный
**Решение**:
1. Перепроверь токен от @BotFather
2. Токен должен быть вида: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`

---

## 💡 Дополнительные возможности

### Отправка алертов о сделках

Можешь добавить отправку уведомлений о каждой сделке:

```python
# После открытия позиции
await telegram_bot.send_message(
    f"🟢 <b>LONG {symbol}</b> opened\n"
    f"Entry: ${entry_price:.2f}\n"
    f"Size: {quantity} @ {leverage}x"
)

# После закрытия позиции
await telegram_bot.send_message(
    f"✅ <b>{symbol} closed</b>\n"
    f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
)
```

### Команды для бота

Можешь добавить команды для управления ботом через Telegram:

- `/status` - текущий статус
- `/positions` - открытые позиции
- `/stats` - статистика
- `/pause` - приостановить торговлю
- `/resume` - возобновить торговлю

---

## 📚 API Reference

### TelegramDashboardBot

```python
from infra.telegram_bot import TelegramDashboardBot

# Инициализация
bot = TelegramDashboardBot(token="YOUR_TOKEN", chat_id="YOUR_CHAT_ID")

# Тест соединения
if await bot.test_connection():
    print("Connected!")

# Отправить текст
await bot.send_message("Hello from bot!")

# Отправить файл
await bot.send_document(Path("report.html"), caption="Report")

# Отправить обновление дашборда
await bot.send_dashboard_update(dashboard_data)
```

---

## 🎮 Интерактивное управление через Telegram

### 📱 Команды бота

После запуска бота вы можете использовать команды прямо в Telegram:

```
/start      - Приветственное сообщение
/menu       - Показать главное меню с кнопками  👈 НАЧНИ ОТСЮДА!
/status     - Текущий статус бота и баланс
/positions  - Список открытых позиций
/stats      - Статистика торговли
/help       - Справка по командам
```

### 🎯 Главное меню

Отправь `/menu` чтобы увидеть интерактивное меню с кнопками:

```
🤖 Trading Bot Menu

┌──────────────┬──────────────┐
│ 📊 Портфолио │ 📈 Статистика│
├──────────────┼──────────────┤
│📝 Активные   │ 📜 История   │
│   сделки     │              │
├──────────────┼──────────────┤
│⚙️ Настройки  │ 💰 Кошелек   │
└──────────────┴──────────────┘
          🔄 Обновить
```

### 💼 Портфолио (📊 кнопка)

Нажми **📊 Портфолио** чтобы увидеть:
- 💵 Текущий баланс в USDT
- 💎 Equity (баланс + unrealized P&L)
- 💰 Общий P&L и ROI
- 🟢/🔴 Индикаторы прибыли/убытка

```
💼 ПОРТФОЛИО

💵 Баланс: $1,075.50 USDT
💎 Equity: $1,100.25 USDT
💰 P&L: +$75.50 (+7.55%)
🟢 ROI: +7.55%

━━━━━━━━━━━━━━━━━━━━
[📊 Детали] [📈 График]
       [🔙 Назад]
```

### 📊 Статистика (📈 кнопка)

Нажми **📈 Статистика** для детальной статистики:
- 🔢 Количество сделок
- 📈 Win Rate (процент прибыльных сделок)
- 💹 Profit Factor
- 📉 Sharpe Ratio

```
📊 СТАТИСТИКА

🔢 Всего сделок: 25
📈 Win Rate: 68.0%
💹 Profit Factor: 2.50
📉 Sharpe Ratio: 1.85

━━━━━━━━━━━━━━━━━━━━
[🏆 Лучшие] [💔 Худшие]
[📅 По дням] [📆 По неделям]
       [🔙 Назад]
```

### 📝 Активные сделки

Нажми **📝 Активные сделки** чтобы увидеть все открытые позиции:

```
📝 ACTIVE POSITIONS

• BTCUSDT
• ETHUSDT
• BNBUSDT

       [🔙 Назад]
```

### 💰 Кошелек

Нажми **💰 Кошелек** для информации о балансе:

```
💰 WALLET

Balance: $1,075.50 USDT

More features coming soon...

       [🔙 Назад]
```

---

## 🚀 Примеры использования

### Вариант 1: Быстрый старт с .env

```bash
# 1. Настрой .env файл
TG_BOT_TOKEN=your_token_here
TG_CHAT_ID=your_chat_id_here
TG_DASHBOARD_ENABLED=true
TG_DASHBOARD_INTERVAL=300

# 2. Запусти бота
python cli.py live --tg-enabled

# 3. В Telegram отправь: /menu
# 4. Нажимай кнопки и управляй ботом! 🎯
```

### Вариант 2: Полная настройка через CLI

```bash
python cli.py live \
  --tg-enabled \
  --tg-bot-token "6754312890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" \
  --tg-chat-id "-1001234567890" \
  --tg-interval 300 \
  --tg-trades

# В Telegram отправь: /start
# Потом: /menu
# Наслаждайся интерактивным управлением! 🎉
```

### Вариант 3: Тестирование без торговли

```bash
# Запусти только Telegram бот для теста
python infra/telegram_bot.py

# Если всё работает, увидишь:
# ✅ Bot connected: @your_bot
# ✅ Test message sent successfully!
```

---

## 🎨 Что происходит автоматически

### 📨 Автоматические уведомления

1. **При открытии позиции:**
```
╔════════════════════════╗
║  ⚡ NEW TRADE LIVE!  ║
╚════════════════════════╝

🟢 BTCUSDT  │  LONG 📈

Entry Price: $40,250.00
Quantity: 0.025
Position Size: $1,006.25
⚡ Leverage: 10x | Margin: $100.63

🛡️ PROTECTION SETUP
💎 Take Profit: $40,750.00
   🎯 Target: +1.24% → ROI: ~12.4%
🛡️ Stop Loss: $39,950.00
   ⚠️ Distance: -0.75% → Risk: ~7.5%

💰 ACCOUNT STATUS
💵 Balance: 1,075.50 USDT
```

2. **При закрытии позиции:**
```
╔════════════════════════╗
║  🎯 GREAT WIN  ║
╚════════════════════════╝

🟢 BTCUSDT  │  LONG

📊 PERFORMANCE
💰 Profit/Loss: +$12.50 USDT
📈 ROI: +12.43%
[█████████████░░] 12.4%

💬 Excellent profit! Keep it up! 💪
```

3. **Периодические обновления дашборда** (каждые 5 мин):
```
🚀 Trading Dashboard Update
...полная статистика...
```

---

## ✅ Checklist для настройки

- [ ] Создал бота через @BotFather
- [ ] Получил Bot Token
- [ ] Добавил бота в группу
- [ ] Получил Chat ID группы
- [ ] Настроил `.env` файл
- [ ] Протестировал бота (`python infra/telegram_bot.py`)
- [ ] Запустил торгового бота с Telegram интеграцией (`python cli.py live --tg-enabled`)
- [ ] Получил первое обновление в группе
- [ ] Попробовал команды (`/start`, `/menu`, `/status`)
- [ ] Нажал на кнопки в меню

---

## 🎯 Полезные советы

### Команды первого дня

1. **День 1 - Знакомство:**
   ```
   /start  → Приветствие
   /help   → Узнай все команды
   /menu   → Открой главное меню
   ```

2. **День 2 - Мониторинг:**
   ```
   /status     → Проверь статус каждое утро
   /positions  → Посмотри открытые сделки
   ```

3. **День 3 - Анализ:**
   ```
   /stats  → Изучи статистику
   /menu → 📊 Портфолио → Проверь ROI
   ```

### Настройка уведомлений

**Меньше спама (рекомендуется):**
```bash
TG_DASHBOARD_INTERVAL=1800  # Раз в 30 минут
TG_TRADE_NOTIFICATIONS=true  # Только сделки
```

**Максимум информации:**
```bash
TG_DASHBOARD_INTERVAL=300    # Каждые 5 минут
TG_TRADE_NOTIFICATIONS=true  # Все сделки
TG_POSITION_UPDATES=true     # + обновления позиций
```

**Только важное:**
```bash
TG_DASHBOARD_INTERVAL=3600   # Раз в час
TG_TRADE_NOTIFICATIONS=true  # Только сделки
```

---

**🎉 Готово! Теперь ты можешь полностью управлять ботом через Telegram!**

**💡 Совет:** Начни с `/menu` - это самый удобный способ взаимодействия с ботом!
