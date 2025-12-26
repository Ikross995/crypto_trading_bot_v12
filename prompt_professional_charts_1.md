# ПРОМТ ДЛЯ CLAUDE CODE: Профессиональные графики для крипто-бота

## 🎯 ЗАДАЧА

Нужно создать систему визуализации для торгового бота уровня TradingView/Binance Pro. НЕ используй базовый matplotlib с дефолтными настройками — это выглядит как студенческий проект. Нужен production-ready дашборд.

---

## ⚡ КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ

### 1. БИБЛИОТЕКИ (выбери одну из стратегий)

**Вариант A — TradingView-style (рекомендую):**
```bash
pip install lightweight-charts  # TradingView Lightweight Charts для Python
```
- Выглядит ТОЧНО как TradingView
- Candlestick с объёмами из коробки
- Поддержка real-time обновлений
- Toolbox для рисования (трендлайны, уровни)
- Мультипанельные графики (Subcharts)

**Вариант B — Plotly Dash (для web-дашбордов):**
```bash
pip install plotly dash dash-bootstrap-components pandas-ta
```
- Интерактивные веб-дашборды
- Bootstrap тема для профессионального вида
- Callbacks для real-time обновлений

---

### 2. ОБЯЗАТЕЛЬНЫЙ СТИЛЬ — DARK THEME

```python
# Цветовая палитра (как у Binance/TradingView):
COLORS = {
    'background': '#131722',        # Тёмный фон
    'card_bg': '#1E222D',           # Фон карточек
    'grid': '#2A2E39',              # Линии сетки
    'text': '#D1D4DC',              # Основной текст
    'text_secondary': '#787B86',    # Вторичный текст
    'green': '#26A69A',             # Бычьи свечи / профит
    'red': '#EF5350',               # Медвежьи свечи / убыток
    'blue': '#2962FF',              # Акцентный синий
    'yellow': '#FFEB3B',            # Предупреждения
    'volume_up': '#26A69A80',       # Объём (прозрачный зелёный)
    'volume_down': '#EF535080',     # Объём (прозрачный красный)
}
```

---

### 3. СТРУКТУРА ГЛАВНОГО ДАШБОРДА

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER: Пара | Цена | 24h Change | Balance | PnL Today         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│         ГЛАВНЫЙ ГРАФИК (Candlestick + Volume)                   │
│         - MA линии (7, 25, 99)                                  │
│         - Точки входа/выхода (маркеры)                          │
│         - Volume внизу графика                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│    RSI Panel   │   MACD Panel   │   Market Regime Indicator     │
├─────────────────────────────────────────────────────────────────┤
│                      TRADE HISTORY TABLE                        │
│  Time | Side | Price | Amount | PnL | Strategy | Status         │
├─────────────────────────────────────────────────────────────────┤
│   EQUITY CURVE  │  WIN RATE  │  DRAWDOWN  │  SHARPE RATIO       │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4. ОБЯЗАТЕЛЬНЫЕ КОМПОНЕНТЫ ГРАФИКОВ

#### A. Candlestick Chart:
- OHLCV данные
- Wicks (тени) тоньше тела свечи
- Hover tooltip с ценой/объёмом/временем
- Zoom и Pan (scroll для масштаба)
- Crosshair (перекрестие)

#### B. Индикаторы на графике:
- MA линии (разные цвета, полупрозрачные)
- Bollinger Bands (fill между линиями)
- Точки сделок: 🔺 покупка (зелёный) / 🔻 продажа (красный)
- Stop-loss / Take-profit уровни (пунктирные линии)

#### C. Отдельные панели:
- RSI (0-100, линии 30/70)
- MACD (гистограмма + линии)
- Volume (цветные бары по направлению свечи)

#### D. Метрики в реальном времени:
- Текущий PnL (зелёный/красный)
- Open positions
- Balance / Equity
- Win Rate %
- Max Drawdown %

---

### 5. ПРИМЕР КОДА (lightweight-charts)

```python
from lightweight_charts import Chart
import pandas as pd

def create_trading_dashboard(df: pd.DataFrame, trades: list):
    """
    df: DataFrame с колонками [time, open, high, low, close, volume]
    trades: список сделок [{'time': ..., 'side': 'buy/sell', 'price': ...}]
    """
    
    # Основной график
    chart = Chart(
        width=1400,
        height=800,
        title='BTC/USDT Trading Bot',
        inner_width=1,
        inner_height=0.7,  # 70% для основного графика
    )
    
    # Стиль TradingView Dark
    chart.layout(
        background_color='#131722',
        text_color='#D1D4DC',
        font_size=12,
        font_family='Trebuchet MS'
    )
    
    chart.candle_style(
        up_color='#26A69A',
        down_color='#EF5350',
        border_up_color='#26A69A',
        border_down_color='#EF5350',
        wick_up_color='#26A69A',
        wick_down_color='#EF5350',
    )
    
    chart.volume_config(
        up_color='rgba(38, 166, 154, 0.5)',
        down_color='rgba(239, 83, 80, 0.5)',
    )
    
    chart.grid(
        vert_enabled=True,
        horz_enabled=True,
        color='#2A2E39',
    )
    
    # Данные
    chart.set(df)
    
    # Добавляем MA линии
    ma7 = chart.create_line('MA7', color='#F0B90B', width=1)
    ma7.set(df['close'].rolling(7).mean())
    
    ma25 = chart.create_line('MA25', color='#0ECB81', width=1)
    ma25.set(df['close'].rolling(25).mean())
    
    ma99 = chart.create_line('MA99', color='#F6465D', width=1)
    ma99.set(df['close'].rolling(99).mean())
    
    # Маркеры сделок
    for trade in trades:
        chart.marker(
            time=trade['time'],
            position='below' if trade['side'] == 'buy' else 'above',
            shape='arrowUp' if trade['side'] == 'buy' else 'arrowDown',
            color='#26A69A' if trade['side'] == 'buy' else '#EF5350',
            text=f"{trade['side'].upper()} @ {trade['price']:.2f}"
        )
    
    # RSI subplot
    rsi_chart = chart.create_subchart(height=0.15)
    rsi_line = rsi_chart.create_line('RSI', color='#BB86FC')
    rsi_line.set(calculate_rsi(df['close']))
    rsi_chart.horizontal_line(70, color='#EF5350', style='dashed')
    rsi_chart.horizontal_line(30, color='#26A69A', style='dashed')
    
    # MACD subplot
    macd_chart = chart.create_subchart(height=0.15)
    # ... MACD histogram
    
    chart.show()
```

---

### 6. АНТИ-ПАТТЕРНЫ (НЕ ДЕЛАЙ ТАК!)

❌ **Matplotlib с дефолтными настройками:**
```python
# ПЛОХО - выглядит как лабораторная работа
plt.plot(df['close'])
plt.show()
```

❌ **Белый фон** — трейдеры ВСЕГДА используют тёмную тему

❌ **Статичные PNG без интерактивности** — в 2024+ нужны zoom/pan/hover

❌ **Один график без индикаторов** — нужен мультипанельный layout

❌ **Базовые линии вместо свечей** — OHLC candlestick обязателен

❌ **Нечитаемые шрифты и оси** — используй контрастные цвета

---

### 7. ЭКСПОРТ И ИНТЕГРАЦИЯ

Графики должны:
1. **Сохраняться как HTML** (интерактивные, можно открыть в браузере)
2. **Встраиваться в Telegram бота** (как скриншоты или ссылки)
3. **Обновляться в реальном времени** (для live trading)
4. **Работать с WebSocket** данными от Binance

```python
# Сохранение интерактивного HTML
chart.save('dashboard.html')

# Для Telegram — скриншот
from playwright.sync_api import sync_playwright
# ... делаем screenshot HTML

# Real-time обновление
async def on_tick(data):
    chart.update(data)
```

---

### 8. ДОПОЛНИТЕЛЬНЫЕ ФИЧИ (если будет время)

- **Heatmap** для ликвидаций и объёмов
- **Depth chart** (стакан ордеров)
- **Correlation matrix** между активами  
- **Equity curve** с просадками
- **Trade journal** в виде таблицы
- **Alerts** визуализация (горизонтальные линии с лейблами)

---

## 📦 РЕЗЮМЕ ТЕХНОЛОГИЙ

| Задача | Библиотека | Почему |
|--------|-----------|--------|
| TradingView-style charts | `lightweight-charts` | Идентичен TradingView |
| Web Dashboard | `plotly` + `dash` | Интерактивный веб |
| Индикаторы | `pandas-ta` / `ta-lib` | 130+ индикаторов |
| Real-time | `asyncio` + `websockets` | Binance stream |
| Скриншоты для TG | `playwright` | Headless browser |

---

## ✅ CHECKLIST ПЕРЕД СДАЧЕЙ

- [ ] Dark theme (#131722 фон)
- [ ] Candlestick (не линии!)
- [ ] Volume bar внизу
- [ ] Минимум 2 индикатора (MA + RSI/MACD)
- [ ] Маркеры сделок на графике
- [ ] Hover tooltips
- [ ] Zoom/Pan
- [ ] Responsive layout
- [ ] Сохранение в HTML
- [ ] Real-time update поддержка

---

Сделай так, чтобы графики выглядели как на профессиональной торговой платформе, а не как домашнее задание по программированию! 🚀
