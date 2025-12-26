# 🚀 Enhanced Trading Dashboard

Красивый, современный дашборд с интерактивными графиками для мониторинга торгового бота.

## 🎨 Особенности

- ✨ **6 типов интерактивных графиков** (Chart.js):
  - 💰 Equity (Капитал)
  - 📊 P&L (Прибыль/Убыток)
  - 🎯 Win Rate (Винрейт)
  - 📈 Trades (Статистика сделок)
  - 📉 Drawdown (Просадка)
  - 🎛️ AI Parameters (Параметры AI)

- 🎨 **Современный дизайн**:
  - Анимированные карточки
  - Градиентные фоны
  - Hover-эффекты
  - Пульсирующий Live индикатор
  - Прогресс-бары

- 📱 **Адаптивная верстка** (Desktop + Mobile)

- 🔄 **Интерактивность**:
  - Pinch to zoom
  - Wheel zoom
  - Drag to pan
  - Reset zoom
  - Auto-refresh

## 🚀 Запуск

### Способ 1: Через существующий веб-сервер (рекомендуется)

```bash
# Запустите веб-сервер
python webapp_server.py
```

**Затем откройте в браузере:**
```
http://localhost:8080/
```

**Теперь Enhanced Dashboard - это главный дашборд!** 🎉

### Способ 2: Напрямую из файла

Откройте файл в браузере:
```
C:\Users\User\crypto_trading_bot_v12\data\learning_reports\enhanced_dashboard.html
```

## 📊 Что отображается

### Верхняя панель (Status Bar)
- Bot Status
- Uptime
- Total Trades
- Win Rate
- ROI
- Profit Factor

### ROI Card
Большая карточка с Return on Investment

### Основные метрики (3 карточки)
1. **Account Overview**
   - Balance
   - Equity
   - Initial Balance
   - Total P&L

2. **Trading Performance**
   - Total Trades
   - Win Rate
   - Profit Factor
   - Max Drawdown

3. **AI Learning Status**
   - Confidence Threshold
   - Position Multiplier
   - Adaptations
   - Learning Confidence

### Advanced Statistics
- Average Win
- Average Loss
- Best Trade
- Worst Trade
- Win Streak
- Sharpe Ratio

### Risk Metrics
- Risk Score (с индикатором)
- Margin Usage
- Performance Summary (Daily/Weekly/Monthly)

### Графики
6 интерактивных графиков с переключением по табам

## 🔧 Технологии

- **Chart.js** 4.4.0 - для графиков
- **Hammer.js** - для touch-жестов
- **Pure CSS3** - анимации и стили
- **Vanilla JS** - без фреймворков

## 📁 Файлы

- `enhanced_dashboard.html` - основной файл дашборда
- `webapp_server.py` - веб-сервер с поддержкой дашборда
- `start_dashboard.bat` - bat-файл для Windows

## 💡 Советы

1. Для лучшей производительности используйте современный браузер (Chrome, Firefox, Edge)
2. Для мобильного просмотра используйте responsive режим в DevTools
3. Графики можно зумировать колесом мыши или pinch-жестом
4. Дашборд автоматически обновляется каждые 30 секунд

## 🎯 URL-адреса

При запуске `python webapp_server.py`:

- **Enhanced Dashboard (Main): `http://localhost:8080/`** ⭐ (для Telegram)
- Original Dashboard: `http://localhost:8080/original`
- API: `http://localhost:8080/api/dashboard`
- Health: `http://localhost:8080/api/health`

**Примечание:** Теперь Enhanced Dashboard транслируется на главный URL для использования в Telegram WebApp!

## 🐛 Troubleshooting

**Проблема:** Дашборд не открывается
- Убедитесь, что запущен `webapp_server.py`
- Проверьте, что порт 8080 свободен
- Откройте DevTools (F12) для проверки ошибок

**Проблема:** Графики не отображаются
- Проверьте подключение к интернету (Chart.js загружается из CDN)
- Проверьте консоль браузера на ошибки
- Попробуйте очистить кэш браузера

**Проблема:** Данные не обновляются
- Убедитесь, что файл `data/dashboard_state.json` существует
- Проверьте, что торговый бот запущен и обновляет данные
- Откройте DevTools (F12) → Console и проверьте наличие ошибок
- Проверьте, что API доступен: http://localhost:8080/api/dashboard

## 📝 Changelog

### v3.0 (2025-12-01)
- ✅ Полная переработка дизайна
- ✅ Добавлено 6 типов графиков (Chart.js)
- ✅ Анимации и градиенты
- ✅ Интерактивные элементы
- ✅ Адаптивная верстка
- ✅ Auto-refresh
- ✅ Zoom & Pan для графиков

---

**Создано с ❤️ для трейдинга**
