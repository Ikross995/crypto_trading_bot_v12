#!/usr/bin/env python3
"""
Обновление dashboard_state.json с РЕАЛЬНЫМИ данными из работающего бота
БЕЗ тестовых позиций - только реальные метрики и позиции
"""

import json
from pathlib import Path
from datetime import datetime


def update_dashboard_with_real_data():
    """Обновить дашборд реальными данными из portfolio_history и bot state."""

    # 1. Загружаем реальную историю портфеля
    history_file = Path('data/portfolio_history.json')

    if not history_file.exists():
        print("❌ Файл portfolio_history.json не найден!")
        return

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    if not history:
        print("❌ История портфеля пуста!")
        return

    # 2. Берём последнюю запись
    latest = history[-1]
    balance = latest.get('balance', 0)
    unrealized_pnl = latest.get('unrealized_pnl', 0)
    total_value = latest.get('total_value', balance)

    print(f"✅ Загружены реальные данные:")
    print(f"   Balance: ${balance:.2f}")
    print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
    print(f"   Total Value: ${total_value:.2f}")
    print(f"   Timestamp: {latest.get('timestamp', 'N/A')}")

    # 3. Определяем начальный баланс из первой записи
    initial_balance = 1000.0
    for entry in history:
        bal = entry.get('balance', 0)
        total_val = entry.get('total_value', 0)
        if bal > 100:
            initial_balance = bal
            break
        elif total_val > 100:
            initial_balance = total_val
            break

    # 4. Вычисляем метрики
    total_pnl = total_value - initial_balance
    roi_pct = ((total_value - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0

    # 5. Готовим историю для графика (последние 48 точек)
    labels = []
    values = []

    for entry in history[-48:]:
        timestamp = entry.get('timestamp', '')
        try:
            dt = datetime.fromisoformat(timestamp)
            labels.append(dt.strftime('%H:%M'))
        except:
            labels.append(timestamp[:5] if len(timestamp) > 5 else timestamp)

        values.append(round(entry.get('total_value', 0), 2))

    # 6. Загружаем статистику по трейдам если есть
    trades_file = Path('data/trade_history.json')
    total_trades = 0
    win_rate = 0
    profit_factor = 0

    if trades_file.exists():
        try:
            with open(trades_file, 'r', encoding='utf-8') as f:
                trades = json.load(f)

            if trades:
                total_trades = len(trades)
                winning = sum(1 for t in trades if t.get('pnl', 0) > 0)
                losing = sum(1 for t in trades if t.get('pnl', 0) < 0)

                win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

                total_wins = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
                total_losses = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
                profit_factor = (total_wins / total_losses) if total_losses > 0 else total_wins
        except Exception as e:
            print(f"⚠️  Не удалось загрузить trade_history: {e}")

    # 7. Проверяем реальные активные позиции от бота
    # Если бот работает, он сохраняет позиции в active_positions или через DashboardStateManager
    positions = []

    # Попробуем загрузить из текущего dashboard_state если есть позиции с TP/SL
    current_state_file = Path('data/dashboard_state.json')
    if current_state_file.exists():
        try:
            with open(current_state_file, 'r', encoding='utf-8') as f:
                current_state = json.load(f)

            # Проверяем есть ли позиции с TP/SL (признак реальных позиций от бота)
            current_positions = current_state.get('positions', [])
            for pos in current_positions:
                # Если позиция имеет stop_loss и take_profit - это реальная позиция от бота
                if pos.get('stopLoss') or pos.get('takeProfit'):
                    positions.append(pos)

            if positions:
                print(f"✅ Найдено {len(positions)} реальных позиций с TP/SL от бота")
        except Exception as e:
            print(f"⚠️  Не удалось проверить текущие позиции: {e}")

    # 8. Формируем state с РЕАЛЬНЫМИ данными
    state = {
        'balance': round(balance, 2),
        'equity': round(total_value, 2),
        'initialEquity': round(initial_balance, 2),
        'totalPnl': round(total_pnl, 2),
        'roiPct': round(roi_pct, 2),
        'openPositions': len(positions),
        'totalTrades': total_trades,
        'winRate': round(win_rate, 1),
        'profitFactor': round(profit_factor, 2),
        'sharpeRatio': 0,
        'maxDrawdown': 0,
        'maxDrawdownPct': 0,
        'positions': positions,
        'equityHistory': {
            'labels': labels,
            'values': values
        },
        'lastUpdate': datetime.now().isoformat()
    }

    # 9. Сохраняем
    state_file = Path('data/dashboard_state.json')
    state_file.parent.mkdir(parents=True, exist_ok=True)

    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dashboard обновлен с РЕАЛЬНЫМИ данными!")
    print(f"💰 Balance: ${balance:.2f}")
    print(f"💎 Equity: ${total_value:.2f}")
    print(f"📊 Total P&L: ${total_pnl:+.2f} ({roi_pct:+.2f}%)")
    print(f"📈 History points: {len(values)}")
    print(f"🎯 Total Trades: {total_trades}")
    print(f"🏆 Win Rate: {win_rate:.1f}%")
    print(f"📍 Open Positions: {len(positions)}")

    if not positions:
        print("\n⚠️  Нет открытых позиций с TP/SL")
        print("   Если бот работает, позиции появятся после открытия новых сделок")
        print("   Если бот не работает, запусти его для открытия позиций")

    return state


if __name__ == '__main__':
    update_dashboard_with_real_data()
