# 📱 Telegram Web App Setup

## Интерактивный Dashboard в Telegram

Теперь ты можешь открывать полноценный интерактивный дашборд прямо в Telegram без выхода из приложения!

---

## 🎯 Что это дает:

- ✅ **Интерактивные графики** - Equity curve, P&L, позиции
- ✅ **Красивый UI** - Адаптивный дизайн в стиле Telegram
- ✅ **Real-time данные** - Обновление по кнопке
- ✅ **Нативная интеграция** - Открывается прямо в Telegram

---

## 🚀 Быстрая настройка (3 варианта):

### **Вариант 1: GitHub Pages (Рекомендуется для продакшена)**

**Шаг 1: Загрузи файл на GitHub**

```bash
# Перейди в папку проекта
cd C:\Users\User\crypto_trading_bot_v12

# Создай ветку для GitHub Pages
git checkout -b gh-pages

# Скопируй Web App в корень
copy telegram_webapp\dashboard.html index.html

# Закоммить
git add index.html
git commit -m "Add Telegram Web App dashboard"

# Запушить
git push origin gh-pages
```

**Шаг 2: Включи GitHub Pages**

1. Открой репозиторий на GitHub
2. Settings → Pages
3. Source: `gh-pages` branch
4. Save

**Шаг 3: Получи URL**

URL будет: `https://<username>.github.io/<repo-name>/index.html`

Например: `https://Ikross995.github.io/crypto_trading_bot_v12/index.html`

---

### **Вариант 2: Локальный сервер с ngrok (Для тестирования)**

**Шаг 1: Запусти локальный сервер**

```powershell
# В папке проекта
cd telegram_webapp
python -m http.server 8000
```

**Шаг 2: Установи ngrok**

Скачай: https://ngrok.com/download

```powershell
# Запусти ngrok
ngrok http 8000
```

**Шаг 3: Скопируй HTTPS URL**

ngrok покажет URL вида: `https://abc123.ngrok.io`

Твой Web App URL: `https://abc123.ngrok.io/dashboard.html`

⚠️ **Важно:** ngrok URL меняется при перезапуске!

---

### **Вариант 3: Netlify (Самый простой)**

**Шаг 1: Создай аккаунт на Netlify.com**

**Шаг 2: Перетащи папку `telegram_webapp` на Netlify**

Drag & Drop в интерфейс Netlify

**Шаг 3: Получи URL**

Netlify даст постоянный URL вида: `https://your-app.netlify.app/dashboard.html`

---

## ⚙️ Настройка в боте

### **Вариант 1: Через .env файл**

```bash
# Добавь в .env
TG_WEBAPP_URL=https://your-url.com/dashboard.html
```

### **Вариант 2: Через код**

Отредактируй `infra/telegram_bot.py`:

```python
# В методе __init__ класса TelegramDashboardBot
self.webapp_url = "https://your-url.com/dashboard.html"
```

---

## 🎮 Как использовать

После настройки:

1. Открой бота в Telegram
2. Отправь `/menu`
3. Нажми кнопку **📱 Интерактивный Dashboard**
4. Откроется Web App с графиками и статистикой!

---

## 🔧 Кастомизация

### **Изменить цвета**

Отредактируй `telegram_webapp/dashboard.html`:

```css
.stat-card.positive .value {
    color: #4CAF50; /* Зеленый для прибыли */
}

.stat-card.negative .value {
    color: #F44336; /* Красный для убытка */
}
```

### **Добавить новые графики**

```javascript
// В функции renderDashboard добавь:
<div class="chart-container">
    <h3>💹 P&L по дням</h3>
    <canvas id="pnlChart"></canvas>
</div>
```

### **Подключить Real-Time данные**

Замени `mockData` на API call:

```javascript
async function loadData() {
    const response = await fetch('/api/dashboard');
    const data = await response.json();
    renderDashboard(data);
}
```

---

## 📡 API Endpoint (Опционально)

Если хочешь Real-Time данные, создай API endpoint:

```python
# В runner/live.py или отдельном файле
from aiohttp import web

async def get_dashboard_data(request):
    # Получить данные от trading_engine
    data = {
        'balance': trading_engine.equity_usdt,
        'totalPnl': ...,
        'positions': [...],
    }
    return web.json_response(data)

# Запустить сервер
app = web.Application()
app.router.add_get('/api/dashboard', get_dashboard_data)
web.run_app(app, port=8080)
```

---

## 🐛 Troubleshooting

### **Web App не открывается**

1. Проверь что URL доступен (открой в браузере)
2. URL должен быть **HTTPS** (не HTTP)
3. Проверь что бот имеет доступ к URL

### **Данные не обновляются**

1. Нажми кнопку 🔄 Refresh в Web App
2. Проверь что `mockData` заменен на реальные данные

### **Кнопка Web App не появилась**

1. Проверь что `TG_WEBAPP_URL` настроен в `.env`
2. Перезапусти бота
3. Отправь `/menu` снова

---

## 📝 Следующие шаги

- [ ] Выбери способ хостинга (GitHub Pages/ngrok/Netlify)
- [ ] Размести `dashboard.html` на хостинге
- [ ] Получи HTTPS URL
- [ ] Добавь URL в `.env` как `TG_WEBAPP_URL`
- [ ] Перезапусти бота
- [ ] Отправь `/menu` и нажми 📱 Интерактивный Dashboard

---

**🎉 Готово! Теперь у тебя полноценный интерактивный dashboard прямо в Telegram!**
