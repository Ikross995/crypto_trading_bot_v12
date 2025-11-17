#!/usr/bin/env python3
"""
Скрипт для обновления dashboard_state.json с актуальными данными
Используется для веб-приложения когда торговый бот не запущен
"""

import json
from pathlib import Path
from datetime import datetime


def load_portfolio_history():
    """Загрузить историю портфеля из portfolio_history.json."""
    try:
        history_file = Path('data/portfolio_history.json')
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки portfolio_history.json: {e}")
    return []


def get_latest_balance():
    """Получить последний баланс из истории портфеля."""
    history = load_portfolio_history()
    if history:
        # Получить последнюю запись с ненулевым балансом
        for entry in reversed(history):
            balance = entry.get('total_value', 0)
            if balance > 0:
                return balance
    return 0.0


def load_equity_history():
    """Загрузить equity историю из CSV файла."""
    try:
        import pandas as pd
        equity_file = Path('data/equity_15m.csv')

        if equity_file.exists():
            df = pd.read_csv(equity_file)
            if not df.empty:
                # Последние 20 точек
                df_recent = df.tail(20)
                labels = df_recent['timestamp'].astype(str).tolist()
                values = df_recent['equity'].tolist()
                return labels, values
    except Exception as e:
        print(f"Ошибка загрузки equity_15m.csv: {e}")

    return [], []


def create_dashboard_state():
    """Создать файл dashboard_state.json с актуальными данными."""

    # Получить текущий баланс
    balance = get_latest_balance()
    initial = 1836.0  # Начальный баланс (можно изменить)

    # Рассчитать метрики
    total_pnl = balance - initial
    roi_pct = ((balance - initial) / initial * 100) if initial > 0 else 0.0

    # Загрузить equity историю
    labels, values = load_equity_history()

    # Если нет истории, создать простую для демо
    if not values:
        from datetime import timedelta
        now = datetime.now()
        for i in range(10):
            time = now - timedelta(hours=10-i)
            labels.append(time.strftime('%H:%M'))
            # Линейная интерполяция от initial до balance
            progress = i / 9
            value = initial + (balance - initial) * progress
            values.append(round(value, 2))

    # Формируем состояние
    state = {
        'balance': round(balance, 2),
        'equity': round(balance, 2),
        'totalPnl': round(total_pnl, 2),
        'roiPct': round(roi_pct, 2),
        'openPositions': 0,  # Будет заполнено торговым ботом
        'totalTrades': 0,    # Будет заполнено торговым ботом
        'winRate': 0.0,      # Будет заполнено торговым ботом
        'profitFactor': 0.0, # Будет заполнено торговым ботом
        'positions': [],     # Будет заполнено торговым ботом
        'equityHistory': {
            'labels': labels[-20:],  # Последние 20 точек
            'values': values[-20:]
        },
        'lastUpdate': datetime.now().isoformat()
    }

    # Сохраняем в файл
    state_file = Path('data/dashboard_state.json')
    state_file.parent.mkdir(parents=True, exist_ok=True)

    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"✅ Файл {state_file} обновлен!")
    print(f"📊 Баланс: ${balance:.2f}")
    print(f"💰 P&L: ${total_pnl:+.2f} ({roi_pct:+.2f}%)")
    print(f"📈 История: {len(values)} точек")

    return state


if __name__ == '__main__':
    state = create_dashboard_state()

    # Вывести содержимое для проверки
    print("\n📄 Содержимое файла:")
    print(json.dumps(state, indent=2, ensure_ascii=False))
