#!/usr/bin/env python3
"""
Исправление данных дашборда - берём РЕАЛЬНЫЕ данные из portfolio_history.json
"""

import json
from pathlib import Path
from datetime import datetime

def fix_dashboard_data():
    """Создать dashboard_state.json с РЕАЛЬНЫМИ данными из portfolio_history."""

    # Загружаем реальную историю портфеля
    history_file = Path('data/portfolio_history.json')

    if not history_file.exists():
        print("❌ Файл portfolio_history.json не найден!")
        return

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    if not history:
        print("❌ История портфеля пуста!")
        return

    # Берём последнюю запись
    latest = history[-1]
    balance = latest.get('balance', 0)
    unrealized_pnl = latest.get('unrealized_pnl', 0)
    total_value = latest.get('total_value', balance)

    print(f"✅ Загружены реальные данные:")
    print(f"   Balance: ${balance:.2f}")
    print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
    print(f"   Total Value: ${total_value:.2f}")

    # Начальный баланс - ищем первую ненулевую запись
    initial_balance = 1000.0
    if len(history) > 0:
        # Ищем первую запись с ненулевым балансом
        for entry in history:
            balance_val = entry.get('balance', 0)
            total_val = entry.get('total_value', 0)
            if balance_val > 100:
                initial_balance = balance_val
                break
            elif total_val > 100:
                initial_balance = total_val
                break

    print(f"   Initial Balance: ${initial_balance:.2f}")

    # Вычисляем метрики
    total_pnl = total_value - initial_balance
    roi_pct = ((total_value - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0

    # Готовим историю для графика (последние 48 точек)
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

    # Загружаем данные о трейдах если есть
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

    # Загружаем открытые позиции если есть
    positions = []
    positions_file = Path('data/active_positions.json')

    if positions_file.exists():
        try:
            with open(positions_file, 'r', encoding='utf-8') as f:
                active_positions = json.load(f)

            # Преобразуем в формат для дашборда
            for symbol, pos_data in active_positions.items():
                if isinstance(pos_data, dict):
                    positions.append({
                        'symbol': symbol,
                        'side': pos_data.get('side', 'LONG'),
                        'entryPrice': round(pos_data.get('entry_price', 0), 4),
                        'currentPrice': round(pos_data.get('current_price', pos_data.get('entry_price', 0)), 4),
                        'quantity': round(pos_data.get('quantity', 0), 6),
                        'pnl': round(pos_data.get('pnl', 0), 2),
                        'pnlPct': round(pos_data.get('pnl_pct', 0), 2),
                        'leverage': pos_data.get('leverage', 1)
                    })
            print(f"✅ Загружено позиций: {len(positions)}")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить active_positions: {e}")

    # Формируем state с ПРАВИЛЬНЫМИ данными
    state = {
        'balance': round(balance, 2),  # РЕАЛЬНЫЙ баланс
        'equity': round(total_value, 2),  # РЕАЛЬНЫЙ total value (balance + unrealized PnL)
        'initialEquity': round(initial_balance, 2),
        'totalPnl': round(total_pnl, 2),
        'roiPct': round(roi_pct, 2),
        'openPositions': len(positions),
        'totalTrades': total_trades,
        'winRate': round(win_rate, 1),
        'profitFactor': round(profit_factor, 2),
        'sharpeRatio': 0,  # Будет рассчитано ботом
        'maxDrawdown': 0,
        'maxDrawdownPct': 0,
        'positions': positions,
        'equityHistory': {
            'labels': labels,
            'values': values
        },
        'lastUpdate': datetime.now().isoformat()
    }

    # Сохраняем
    state_file = Path('data/dashboard_state.json')
    state_file.parent.mkdir(parents=True, exist_ok=True)

    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Файл {state_file} обновлен с РЕАЛЬНЫМИ данными!")
    print(f"💰 Balance: ${balance:.2f}")
    print(f"💎 Equity: ${total_value:.2f}")
    print(f"📊 Total P&L: ${total_pnl:+.2f} ({roi_pct:+.2f}%)")
    print(f"📈 History points: {len(values)}")
    print(f"🎯 Total Trades: {total_trades}")
    print(f"🏆 Win Rate: {win_rate:.1f}%")

    return state

if __name__ == '__main__':
    fix_dashboard_data()
